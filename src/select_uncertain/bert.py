"""
Intended for scoring LLM-generated samples by probability of wrong annotations
    Previously, using average log probs on the entity annotation span

Now, explore using BERT-class models trained on the LLM generated data as a scoring function,
    indicating uncertainty/probability of wrong annotation

In particular, get the BERT CLS head output logits for the LLM-annotated spans and aggregate these to a probability
"""

import os
import json
import math
from os.path import join as os_join
from copy import deepcopy
from typing import Dict, Tuple, List, Any, Union
from collections import defaultdict
from dataclasses import dataclass, asdict

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import default_collate
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm

from stefutil import get_logger, pl, add_file_handler, now, fmt_e, round_f, group_n, punc_tokenize
from src.util import sconfig, dataset_name2data_dir, dataset_name2model_dir
from src.util.ner_example import NerReadableExample, NerBioExample
from src.data_util import prettier, dataset
from src.trainer.utils_ner import get_tag_to_id, get_chunks, NerChunk


logger = get_logger('BERT Uncertain Sample Selector')


__all__ = [
    'BertConfidence', 'BertEvalNConfidence', 'BertPredictedSample', 'EncodedEntityInfo', 'EncodeSampleOutput',
    'ner_dataset2uncertain_triples'
]


N_SPECIAL_TOKENS_COUNT = 2  # for DeBERTa V3


@dataclass
class EncodedEntityInfo:
    # the indices of the entity span in the encoded sequence
    #   i.e. index of non-subword tokens in the encoded sequence
    #   also including a token before and after the span if possible
    # TODO: also try a scoring version w/o the extra tokens on both ends?
    sample_indices: List[Tuple[int, int]] = None  # corresponding to the (start & end) indices in the original BIO sample
    encoded_indices: List[int] = None  # corresponding to the indices (1st-subword) in the encoded sequence for BERT
    tag_ids: List[int] = None  # the tag ids corresponding to the entity span
    # these should all be of the same length

    # when `include_span_edge` is True, the actual indices of th entity span, w/o potential extra tokens on both ends
    positive_span: Tuple[int, int] = None  # start inclusive, end exclusive


@dataclass
class EncodeSampleOutput:
    inputs: Dict[str, torch.Tensor] = None
    entities_info: List[EncodedEntityInfo] = None
    encoded_token_indices: List[int] = None  # the indices in the encoded sequence corresponding to actual tokens in the sample


@dataclass
class BertConfidence:
    """
    BERT confidence on a single entity span
    """
    conf: float = None
    # the indices of the tokens in the entity span that were considered for uncertainty
    #   intended to keep track of `include_span_edge`
    token_indices: List[int] = None
    # to keep track of discrepancies like:
    #   LLM generated: [O, B-Year, O]
    #   BERT predicted: [O, I-Year, O]
    pred_tags: List[str] = None
    pred_type: str = None  # the predicted entity type
    # to check if the predicted tags are consistent with the LLM-annotated tags
    llm_tags: List[str] = None  # the LLM-generated tags processed into BIO format


def chunks2tags(chunks: List[NerChunk], seq_length: int = None) -> List[str]:
    """
    Converts a list of chunks to a list of NER BIO tags
        Intended to convert BERT predictions back to human-readable tags
    """
    # note chunks are start inclusive, end exclusive
    # sanity check chunks are not overlapping
    spans = [(chunk.start, chunk.end) for chunk in chunks]
    assert all(e1 <= s2 for (_, e1), (s2, _) in zip(spans[:-1], spans[1:]))
    
    i = 0
    it_chunk = iter(chunks)
    next_chunk = next(it_chunk, None)
    ret = []
    while i < seq_length:
        if next_chunk is None or i < next_chunk.start:
            ret.append('O')
            i += 1
        elif i == next_chunk.start:
            ret.append('B-' + next_chunk.entity_type)
            i += 1
        elif i < next_chunk.end:
            ret.append('I-' + next_chunk.entity_type)
            i += 1
        else:
            assert i == next_chunk.end
            next_chunk = next(it_chunk, None)
    return ret


@dataclass
class BertEvalNConfidence:
    """
    BERT annotation prediction on a single LLM-generated sample & corresponding confidences of LLM annotation
    """
    loss: float = None  # loss on the entire sequence
    conf: float = None  # confidence as percentage for the entire annotation, based on loss on all tokens
    # confidence for each entity annotation, based on loss for all tokens in each entity span
    entity_confs: List[BertConfidence] = None  # length matches #entities in the sample
    pred_tags: List[str] = None  # predicted tags from simple logit argmax, length matches #tokens in the sample
    pred_chunks: List[NerChunk] = None  # predicted chunks by further processing the token-level tags, i.e. handling starting `I-` tags
    pred_tags_processed: List[str] = None  # i.e. ensure a `B-` tag before any `I-` tag


@dataclass
class BertPredictedSample:
    # original sample (i.e. annotated by LLM), also w/
    #   entity annotations from BERT, and it's confidence in LLM annotations
    sample: NerReadableExample = None
    bio_sample: NerBioExample = None
    bert_pred: BertEvalNConfidence = None


class Tags2Chunks:
    """
    Get chunks expects tag ids and a map from tag to id,
        this is a wrapper/syntactic sugar that just takes tags
    """
    def __init__(self, tag2id: Dict[str, int]):
        self.tag2id = tag2id

    def __call__(self, tags: List[str]) -> List[NerChunk]:
        return get_chunks(seq=[self.tag2id[t] for t in tags], tag2id=self.tag2id, sort=True)


def bert_pred_chunks2type(pred_chunks: List[NerChunk], ec: prettier.EdgeCases = None, **kwargs) -> Union[str, Dict[str, int]]:
    n_pred_chunk = len(pred_chunks)
    if n_pred_chunk == 0:
        return '__NA__'
    else:
        if n_pred_chunk > 1:
            ret = defaultdict(int)  # show each type annotated w/ the corresponding occurrence count
            for chunk in pred_chunks:
                ret[chunk.entity_type] += chunk.end - chunk.start
            if ec:
                msg_ = f'BERT predicted more than 1 entity type for an LLM-annotated entity'
                if kwargs:
                    msg_ = f'{msg_}: {pl.i(kwargs, indent=1)}'
                ec(msg=msg_, kind='multiple-type-in-llm-annotated-entity', args=dict(type_dist=ret))
            return ret
        else:
            return pred_chunks[0].entity_type


@dataclass
class TagInfo:
    tag: str = None  # the tag as string
    id: int = None  # the tag id
    logit: float = None  # the logit value for the tag


@dataclass
class ITagModificationInfo:
    index: int = None  # the index of the starting `I-` tag in the sequence, the difference/modification that we make
    entity_type: str = None  # the entity type of the tag
    i_tag: TagInfo = None  # the original `I-` tag info
    b_tag: TagInfo = None  # the original `B-` tag info


@dataclass
class ModifyStartingITagsOutput:
    found: bool = None  # whether any starting `I-` tags were found
    mods: List[ITagModificationInfo] = None  # tracks modification of starting `I-` tags to `B-` tags
    logits: torch.Tensor = None  # the modified logits where the values at modified tag indices between the `I-` and `B-` tags are swapped


def modify_starting_i_tag_in_sequence(
        tags_true: List[str] = None, tags_pred: List[str] = None, tag2id: Dict[str, int] = None, tags2chunks: Tags2Chunks = None,
        logits: torch.Tensor = None
) -> ModifyStartingITagsOutput:
    tags_true, tags_pred = list(tags_true), list(tags_pred)
    seq_len, n_tag = logits.shape
    assert len(tags_true) == len(tags_pred) == seq_len  # sanity check
    assert len(tag2id) == n_tag

    tags2chunks = tags2chunks or Tags2Chunks(tag2id=tag2id)
    chunks_true, chunks_pred = tags2chunks(tags=tags_true), tags2chunks(tags=tags_pred)
    # note: starting I-tags doesn't influence the annotated chunks, so check for starting I tag in each matched chunk
    chunks_ovl = set(chunks_true) & set(chunks_pred)
    chunks_ovl = sorted(chunks_ovl, key=lambda x: (x.start, x.end))  # sort by start index

    mods = []
    logits_mod = logits.clone()

    for chunk in chunks_ovl:
        s, e = chunk.start, chunk.end
        tags_true_, tags_pred_ = tags_true[s:e], tags_pred[s:e]
        if tags_true_ != tags_pred_:  # must be the case that the only difference is the starting `I-` tag
            tag_true, tag_pred = tags_true_[0], tags_pred_[0]
            assert tags_true_[1:] == tags_pred_[1:]  # sanity check the rest of the tags are the same

            tag_type_true, entity_type_true = tag_true.split('-')
            tag_type_pred, entity_type_pred = tag_pred.split('-')
            assert entity_type_pred == entity_type_true  # sanity check the entity type is the same
            entity_type = entity_type_true

            assert tag_type_true == 'B' and tag_type_pred == 'I'  # sanity check indeed prediction is incorrectly a starting `I-` tag
            idx_diff = s  # the index of the starting `I-` tag in the sequence

            # get the logits for the `B-` and `I-` tags
            id_b, id_i = tag2id[tag_true], tag2id[tag_pred]
            logits_mod[idx_diff, id_b], logits_mod[idx_diff, id_i] = ori_i, ori_b = logits[idx_diff, id_i], logits[idx_diff, id_b]
            assert ori_i > ori_b  # sanity check the original `I-` tag has higher logit than the `B-` tag, should be the case by argmax

            mods.append(ITagModificationInfo(
                index=idx_diff, entity_type=entity_type,
                i_tag=TagInfo(tag=tag_pred, id=id_i, logit=ori_i.item()),
                b_tag=TagInfo(tag=tag_true, id=id_b, logit=ori_b.item())
            ))
    return ModifyStartingITagsOutput(found=len(mods) > 0, mods=mods, logits=logits_mod)


class UncertainSampleEncoder:
    """
    Encoding samples for BERT as uncertainty sample scoring function
    """
    def __init__(
            self, ner_tags: List[str] = None, tag2id: Dict[str, int] = None, tokenizer: PreTrainedTokenizer = None,
            pad_token_label_id: int = -100, max_seq_length: int = 144, lowercase: bool = False
    ):
        self.ner_tags = ner_tags
        self.tag2id = tag2id or {tag: i for i, tag in enumerate(ner_tags)}

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_label_id = pad_token_label_id

        self.lowercase = lowercase

    def __call__(self, sample: NerReadableExample = None, include_span_edge: bool = True) -> EncodeSampleOutput:
        """
        Encode a sample into the format expected by BERT, also include the entity span indices for getting entity-wise uncertainty

        :param sample: the sample to encode
        :param include_span_edge: whether to include 1 token before and after the entity span if possible
        """
        # the labels are utilized in 2 ways:
        #   1) labels for the entire sequence passed into BERT forward pass:loss as per-sample annotation uncertainty
        #   2) labels near each annotated entity span: cross entropy as per-entity annotation uncertainty
        if self.lowercase:
            sample = sample.to_lower()
        bio_sample = sample.to_bio()

        entity_idxs = bio_sample.get_entity_span_indices()  # start & end inclusive
        # start & end within each entity index; start inclusive, end exclusive
        entity_spans = [(0, e - s + 1) for s, e in entity_idxs]
        if include_span_edge:  # add 1 token before and after the entity span if possible
            # note the same index may appear in multiple spans when the 2 spans are close to each other
            n_tag = len(bio_sample.ner_tags)
            entity_idxs_, entity_spans_ = [], []
            # also adjust the internal entity span indices
            for (s, e), (s_, e_) in zip(entity_idxs, entity_spans):
                if s > 0:
                    s -= 1
                    s_ += 1  # inserting token before positive span, the span of positive span also shifts
                    e_ += 1
                if e < n_tag - 1:
                    e += 1  # appending token after positive span, no change to the span of positive span
                entity_idxs_.append((s, e))
                entity_spans_.append((s_, e_))
            entity_idxs = entity_idxs_
            entity_spans = entity_spans_

            # i.e., for each positive entity annotation, get the indices of the span (+1 on both ends) in the encoded sequence,
            #   to compare with BERT logits
        # get the tags corresponding to each index in each entity span
        entity_span_tags = [[bio_sample.ner_tags[i] for i in range(s, e + 1)] for (s, e) in entity_idxs]

        model_tokens, label_ids = [], []

        # start encoding token by token, and also
        #   get the model-token-level label ids corresponding to each entity span
        # one-to-one map by position: from the original token indices to the encoded sequence indices, i.e. the 1st subword token
        lst_encoded_idx = []
        for i, (token, tag) in enumerate(zip(bio_sample.tokens, bio_sample.ner_tags)):
            model_tokens_ = self.tokenizer.tokenize(token)
            assert len(model_tokens_) > 0  # sanity check
            label = self.tag2id[tag]
            lst_encoded_idx.append(len(model_tokens))  # the index of the 1st subword token
            model_tokens += model_tokens_
            label_ids += [label] + [self.pad_token_label_id] * (len(model_tokens_) - 1)  # non-1st model token set to padding label

        # now, get the encoded sequence indices for each entity span
        encoded_entity_idxs = [lst_encoded_idx[s:e + 1] for (s, e) in entity_idxs]

        # sanity check the entity span information
        assert len(entity_idxs) == len(encoded_entity_idxs) == len(entity_span_tags)  # sanity check same #entities
        for span, encoded_idxs, span_tags in zip(entity_idxs, encoded_entity_idxs, entity_span_tags):
            s, e = span  # sanity check one-to-one map between tokens and encoded sequence indices
            assert len(encoded_idxs) == len(span_tags) == e - s + 1
        entity_info = [
            EncodedEntityInfo(sample_indices=entity_idxs, encoded_indices=encoded_idxs, tag_ids=[self.tag2id[t] for t in span_tags], positive_span=span)
            for idxs, encoded_idxs, span_tags, span in zip(entity_idxs, encoded_entity_idxs, entity_span_tags, entity_spans)
        ]

        input_ids = self.tokenizer.convert_tokens_to_ids(model_tokens)
        if len(input_ids) > self.max_seq_length - N_SPECIAL_TOKENS_COUNT:
            # TODO: also need to truncate the entity span indices
            raise NotImplementedError('a too long sentence, truncate')

        # for the BERT tokenization convention
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        label_ids = [self.pad_token_label_id] + label_ids + [self.pad_token_label_id]
        attn_msks = [1] * len(input_ids)  # in HF implementation, 1 means active attention, no masking

        # pad until max seq length
        pad_len = self.max_seq_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        label_ids += [self.pad_token_label_id] * pad_len
        attn_msks += [0] * pad_len
        assert len(input_ids) == len(label_ids) == len(attn_msks) == self.max_seq_length  # sanity check

        # for some reason, this is needed for the collate function
        input_ids, label_ids, attn_msks = map(torch.tensor, (input_ids, label_ids, attn_msks))
        inputs = dict(input_ids=input_ids, labels=label_ids, attention_mask=attn_msks)
        return EncodeSampleOutput(inputs=inputs, entities_info=entity_info, encoded_token_indices=lst_encoded_idx)


def ner_dataset2uncertain_triples(
        dataset_name: str = None, dataset_dir_name: str = None,  model_dir_name: str = None, batch_size: int = 16,
        max_seq_length: int = 144, model_type: str = 'microsoft/deberta-v3-base',
        include_span_edge: bool = True, modify_starting_i_tag: bool = False, lowercase: bool = False,
):
    """
    :param dataset_name: dataset name
    :param dataset_dir_name: dataset directory name
    :param model_dir_name: directory name of BERT model trained on LLM-generated data from directory `dataset_dir_name`
    :param batch_size: batch size for BERT model inference
    :param max_seq_length: maximum sequence length for BERT inference tokenization
    :param model_type: BERT model type
    :param include_span_edge: If True, when comparing the entity span tags between LLM and BERT,
        include 1 token before and after the entity span if available in the encoded sequence
    :param modify_starting_i_tag: If True, BERT-predicted chunks with starting `I-` tags will be modified to `B-` tags
        Corresponding log probs for the swapped tags are also swapped
            Motivation: from manual observations, many high-uncertainty samples may be due to, e.g.
                LLM-span: [O B-Title]; BERT-span: [O I-Title]
                => These may be considered false positives since the final sequence annotation will be the same
    :param lowercase: If True, the sentence will be lowercased before encoding
    """
    # For each NER sample sentence, get the BERT CLS head output logits
    #   For each entity annotated by LLM in the sentence, get the corresponding logits on the same span
    #       To better account for the exact boundary, also include 1 more token at the start and end of the span
    #   Compare the cross entropy of BERT predictions vs LLM-prediction-converted labels, as an estimate of annotation uncertainty
    #       e.g. (entity 1) annotated by LLM => (O, B-entity type 1, I-entity type 1, I-entity type 1, O) as BERT prediction comparison

    # TODO: the original LLM correction pipeline only operates on sentences **with** entities
    #   but here, we can actually look for potential missing LLM false negatives?
    #       Can do this later, ignore for now
    data_path = dataset_name2data_dir(dataset_name=dataset_name, input_dir=dataset_dir_name).path
    ner_samples = dataset.dataset_dir_name2ner_samples(dataset_name=dataset_name, dataset_dir_name=dataset_dir_name).samples

    date = now(for_path=True, fmt='short-date')
    output_dir_nm = f'{date}_BERT-Annotation-Confidence'
    post = dict()
    if include_span_edge:
        post['edge'] = True
    if modify_starting_i_tag:
        post['mod-i'] = True
    if lowercase:
        post['lc'] = True
    if post is not dict():
        output_dir_nm += f'_{pl.pa(post)}'
    # sic(output_dir_nm)
    # raise NotImplementedError
    output_path = os_join(data_path, output_dir_nm)
    add_file_handler(logger=logger, file_path=os_join(output_path, 'process.log'))

    # load model
    model_path = dataset_name2model_dir(dataset_name=dataset_name, model_dir=model_dir_name).path
    model_path = os_join(model_path, 'trained')

    entity_types = sconfig(f'datasets.{dataset_name}.readable-entity-types')
    ner_tags = sconfig(f'datasets.{dataset_name}.readable-ner-tags')
    n_tag = len(ner_tags)

    if model_type not in ['microsoft/deberta-v3-base']:
        raise NotImplementedError
    config = AutoConfig.from_pretrained(model_path, num_labels=n_tag)
    fast_tokenizer = True
    if 'deberta' in model_type:  # to disable warning, see `run_ner`
        fast_tokenizer = False
    # we use `microsoft/deberta-v3-base`, but from HF config, the model type loaded from pretrained config says `deberta-v2`
    #   If just loading from the expected model type, the tokenizers are not exactly the same by object equivalence, but looks like the same
    # model_type = 'microsoft/deberta-v3-base'
    # tokenizer2 = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_type)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, config=config, use_fast=fast_tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=model_path, config=config)
    model.eval()

    if torch.cuda.is_available():
        device = 'cuda'
    # elif torch.backends.mps.is_available():  # TODO: seems to give wrong results???
    #     device = 'mps'
    else:
        device = 'cpu'
    model.to(device)
    cel = CrossEntropyLoss(reduction='none')
    pad_token_label_id = cel.ignore_index

    d_log = {
        'dataset-name': dataset_name, 'dataset-dir-name': dataset_dir_name, 'NER-tags': ner_tags, '#NER-tags': n_tag,
        'model-dir-name': model_dir_name, 'model-path': model_path,
        'model-type-hf': model_type, 'model-type-str': config.model_type, 'model-type': type(model),
        'tokenizer-type': type(tokenizer), 'device': device,
        'batch-size': batch_size, 'pad-token-label-id': pad_token_label_id, 'max-seq-length': max_seq_length,
        'lowercase': lowercase, 'include-span-edge': include_span_edge, 'modify-starting-i-tag': modify_starting_i_tag,
        'output-dir-name': output_dir_nm, 'output-path': output_path
    }
    logger.info(f'Using downstream-trained BERT to select uncertain samples w/ {pl.i(d_log, indent=1)}')

    # batch outputs for faster model inference
    samples_loader = group_n(ner_samples, n=batch_size)
    # can't use the default ordering by dataset entity type for backward compatibility
    #   in earlier NER processing, all entity types are sorted alphabetically and then converted to BIO tags
    # tag2id = {tag: i for i, tag in enumerate(ner_tags)}
    tag2id = get_tag_to_id(path=data_path, verbose=True, dataset_name=dataset_name)
    id2tag = {i: tag for tag, i in tag2id.items()}
    tags2chunks = Tags2Chunks(tag2id=tag2id)

    it = tqdm(samples_loader, total=math.ceil(len(ner_samples) / batch_size), desc='Computing Entity-Wise BERT Uncertainty', unit='ba')
    use = UncertainSampleEncoder(
        ner_tags=ner_tags, tag2id=tag2id, tokenizer=tokenizer, pad_token_label_id=pad_token_label_id, max_seq_length=max_seq_length,
        lowercase=lowercase)

    conf_samples = []  # for each NER sample w/ sample-wise & entity-wise confidence scores
    ec = prettier.EdgeCases(logger=logger)
    for samples in it:
        outs = [use(sample=s, include_span_edge=include_span_edge) for s in samples]  # samples are in the Readable format, will convert to the tag-wise BIO format for encoding
        batch = [out.inputs for out in outs]
        # each element in the list is all entity info for 1 sample,
        #   each sample w/ K entities w/ have entity info of length K
        entity_info: List[List[EncodedEntityInfo]] = [out.entities_info for out in outs]
        enc_idxs = [out.encoded_token_indices for out in outs]
        n_entity = sum(len(entity_info_) for entity_info_ in entity_info)  # #entities total in the batch
        it.set_postfix({'#entities': pl.i(n_entity)})

        batch = default_collate(batch)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs['logits']  # logits shape (#batch, seq len, #tag)

        confs = []
        tag_ids = batch['labels']
        # get loss values & sample-level predictions for each sample
        d_mod_i_tag_ets = defaultdict(list) if modify_starting_i_tag else None
        for i_sample, enc_idxs_ in enumerate(enc_idxs):
            enc_idxs_ = [i+1 for i in enc_idxs_]  # shift by 1 for the CLS token
            logits_ = logits[i_sample, enc_idxs_, :]  # shape (#tag in sample, #tag for dataset)
            pred_ids = logits_.argmax(dim=-1).tolist()
            pred_tags = [id2tag[i] for i in pred_ids]
            chunks = get_chunks(seq=pred_ids, tag2id=tag2id, sort=True)

            targets_ = tag_ids[i_sample, enc_idxs_]  # shape (#tag in sample)
            assert (targets_ != pad_token_label_id).all()  # sanity check no padding in the target
            # this instance of cross entropy loss don't reduce to a single value, so explicitly take the mean
            loss = cel(input=logits_, target=targets_).mean().item()

            # conf = math.exp(-loss)
            conf = -loss  # just use the logprob seems easier for human-inspection
            proc_pred_tags = chunks2tags(chunks, seq_length=len(pred_tags))

            sample = samples[i_sample]

            if modify_starting_i_tag:
                bio_sample = sample.to_bio()
                tags_llm = bio_sample.ner_tags
                out = modify_starting_i_tag_in_sequence(
                    tags_true=tags_llm, tags_pred=pred_tags, tag2id=tag2id, tags2chunks=tags2chunks, logits=logits_)
                if out.found:
                    # there can be multiple starting `I-` tags in a single sequence,
                    #   but the found ones, should be one-to-one correspondence w/ LLM-annotated tags by construction
                    #       will check for this below
                    d_mod_i_tag_ets[i_sample] = [mod.entity_type for mod in out.mods]
                    # override the predicted tags w/ the modified tags and update logprob for this sample
                    logits__ = out.logits
                    loss_ = cel(input=logits__, target=targets_).mean().item()
                    conf_ = -loss_
                    assert conf_ > conf  # sanity check the modified logprob is higher

                    d_log = asdict(sample)
                    d_log.update(llm_tags=tags_llm, bert_tags=pred_tags, original_logprob=fmt_e(conf), modified_logprob=fmt_e(conf_))
                    msg = (f'Edge Case: Swapped starting {pl.i("I-", c="y")} tags w/ {pl.i("B-", c="y")} tags '
                           f'for uncertainty score computation in sample w/ {pl.i(d_log, indent=1)}')
                    ec(msg=msg, kind='sample-swapped-starting-i-tag')
                    loss, conf = loss_, conf_
            assert len(pred_tags) == len(punc_tokenize(sample.sentence))  # sanity check
            confs.append(BertEvalNConfidence(
                loss=loss, conf=conf, pred_tags=pred_tags, pred_chunks=chunks, pred_tags_processed=proc_pred_tags))

        # for each sample & each entity, get the logits corresponding to the entity span
        #   then get the cross entropy score between the logits and the label for each token in the entity span
        #       average the score per token as the final score for the entity
        # first, get the indices for each index inside each entity span
        #   flatten them to index into the logits, for a single cross entropy computation
        #       also, get the corresponding indices (start & end) in the flattened logit indices corresponding to each entity span
        #           to later aggregate the entity-wise scores
        logit_idxs: List[Tuple[int, int]] = []  # a tuple of (sample index, entity index) for each logit
        tag_ids = []  # flattened tag ids corresponding to each logit
        # sample => entity => entity span indices start (inclusive) & enc index (exclusive) as nested list
        logit_index_spans: List[List[Tuple[int, int]]] = []
        global_i = 0
        for i_sample, entity_info_ in enumerate(entity_info):
            logit_index_spans_ = []
            for i_entity, entity_info__ in enumerate(entity_info_):
                encoded_idxs = entity_info__.encoded_indices
                logit_idxs_ = [i + 1 for i in encoded_idxs]  # need to shift the indices by 1 for the CLS token
                logit_idxs += [(i_sample, idx) for idx in logit_idxs_]
                tag_ids += entity_info__.tag_ids

                global_start, global_end = global_i, global_i + len(logit_idxs_)
                logit_index_spans_.append((global_start, global_end))
                global_i = global_end
            logit_index_spans.append(logit_index_spans_)
        # sanity check iterated through all entities in the samples in the batch
        assert global_i == len(logit_idxs) == len(tag_ids)
        assert n_entity == sum(len(logit_index_spans_) for logit_index_spans_ in logit_index_spans)

        sample_idxs, entity_idxs = zip(*logit_idxs)
        logits_span = logits[sample_idxs, entity_idxs, :]  # shape (#indicies for each of entity in all samples in batch, #tag)
        ce = cel(input=logits_span, target=torch.tensor(tag_ids).to(device))

        # assign the computed scores to each entity span
        for i_sample, (sample, logit_spans, entity_info_, conf) in enumerate(zip(samples, logit_index_spans, entity_info, confs)):
            enms, ets = sample.entity_names, sample.entity_types
            assert len(enms) == len(logit_spans)  # sanity check
            entity_confs = []  # confidence as probability of wrong annotation for each entity

            mod_i_tag_ets = [] if modify_starting_i_tag else None
            for enm, et, logit_span, entity_info__ in zip(enms, ets, logit_spans, entity_info_):
                start, end = logit_span
                tag_ids_ = tag_ids[start:end]    # the LLM-labeled tag
                tags = [id2tag[i] for i in tag_ids_]

                logits_span_ = logits_span[start:end, :]  # get the BERT-predicted tags for current entity span
                pred_ids = logits_span_.argmax(dim=-1).tolist()
                pred_tags = [id2tag[i] for i in pred_ids]

                # get the BERT-predicted tags for each entity span
                #   may introduce complications if the predicted tags not exactly 1 type...
                # first, drop potential the 1 additional start & end spans added
                tags_pos, pred_tags_pos = deepcopy(tags), deepcopy(pred_tags)
                if include_span_edge:
                    s, e = entity_info__.positive_span  # start inclusive, end exclusive
                    tags_pos = tags_pos[s:e]
                    pred_tags_pos = pred_tags[s:e]

                # chunks_pos = get_chunks(seq=[tag2id[t] for t in tags_pos], tag2id=tag2id)
                # pred_chunks_pos = get_chunks(seq=[tag2id[t] for t in pred_tags_pos], tag2id=tag2id)
                chunks_pos = tags2chunks(tags=tags_pos)
                pred_chunks_pos = tags2chunks(tags=pred_tags_pos)
                # sanity check LLM-annotation is the same entity type; should be the case by construction
                assert len(chunks_pos) == 1 and chunks_pos[0].entity_type == et

                d_log = dict(
                    sentence=sample.sentence, entity=enm, llm_annotated_entity_type=et, llm_tags=tags_pos, bert_tags=pred_tags_pos)
                pred_et = bert_pred_chunks2type(pred_chunks=pred_chunks_pos, ec=ec, **d_log)

                ce_ = ce[start:end]
                ce__ = ce_.mean().item()  # average the cross entropy score per token
                conf_ = -ce__  # just like OpenAI negative log likelihood
                # convert cross entropy to probability as confidence; didn't seem intuitive, too many values close to 1
                # conf_ = math.exp(-ce__)

                # probs = torch.exp(-ce_)  # try first to prob then average, larger values, but same trend, so ignore
                # conf2 = probs.mean().item()

                if modify_starting_i_tag:
                    # check for the case that the only difference between LLM and BERT annotations is one starting `I-` instead of `B-`

                    out = modify_starting_i_tag_in_sequence(
                        tags_true=tags, tags_pred=pred_tags, tag2id=tag2id, tags2chunks=tags2chunks, logits=logits_span_)
                    found = out.found
                    if found:
                        assert isinstance(pred_et, str)  # sanity check
                        mods = out.mods
                        assert len(mods) == 1  # sanity check only 1 starting `I-` tag is modified
                        mod = mods[0]
                        idx_diff = mod.index
                        mod_i_tag_ets.append(mod.entity_type)

                        # sanity check the difference is indeed the 1st tag
                        if include_span_edge:
                            assert idx_diff in [0, 1]  # the 1st tag must be index 0, or 1 if an additional starting token is included
                        else:
                            assert idx_diff == 0
                        logits_span_mod = out.logits

                        # re-compute the cross entropy score after the logits for these 2 tags are swapped
                        ce_mod = cel(input=logits_span_mod, target=torch.tensor(tag_ids_).to(device))
                        ce__mod = ce_mod.mean().item()
                        assert ce__mod < ce__  # sanity check the modified cross entropy is smaller
                        conf_mod = -ce__mod

                        d_log = dict(sentence=sample.sentence, entity=enm, entity_type=et, llm_tags=tags_pos, bert_tags=pred_tags_pos)
                        for k, tag, lp in [('start_tag_expected', mod.b_tag, conf_), ('start_tag_got', mod.i_tag, conf_mod)]:
                            d_log[k] = dict(tag=tag.tag, id=tag.id, logit=round_f(tag.logit), logprob=fmt_e(lp))
                        msg = (f'Edge Case: Swapped starting {pl.i("I-", c="y")} tags w/ {pl.i("B-", c="y")} tags '
                               f'for uncertainty score computation w/ {pl.i(d_log, indent=1)}')
                        ec(msg=msg, kind='swapped-starting-i-tag', args=dict(entity_type=et))

                        conf_ = conf_mod  # override original confidence
                entity_confs.append(BertConfidence(
                    conf=conf_, token_indices=entity_info__.sample_indices, pred_tags=pred_tags, pred_type=pred_et, llm_tags=tags))
            if modify_starting_i_tag:
                # sanity check different processing should result in the same order of found starting `I-` tags
                assert mod_i_tag_ets == d_mod_i_tag_ets[i_sample]
            conf.entity_confs = entity_confs
        conf_samples += [
            BertPredictedSample(sample=sample, bio_sample=sample.to_bio(ignore_case=lowercase), bert_pred=conf) for sample, conf in zip(samples, confs)
        ]
        # break  # TODO: debugging, just run 1 batch
    if ec.have_edge_case:
        logger.info(ec.summary())
    # store each sample w/ the BERT-predicted confidence
    samples_w_conf = [{**asdict(sample.sample), 'logprob': sample.bert_pred.conf} for sample in conf_samples]
    out_fnm = os_join(output_path, 'logprobs-sample.json')  # write to file; same filename as in logprob ranking from `process_data.py`
    with open(out_fnm, 'w') as f:
        json.dump(samples_w_conf, f, indent=4)

    # rank the samples by the BERT-predicted sample-wise confidence and log, w/ differences in BERT annotations
    samples_compare = []
    for i, sample in enumerate(conf_samples):
        sample, bert_pred = sample.sample, sample.bert_pred
        # for the version of original LLM-annotated sample, formatted for BERT pred comparison
        #   first, transform the original LLM-generated sentence & processed entities to split on whitespace,
        #       to be able to identify same annotations
        sample_llm = sample.to_split_on_puncs()
        sent_cp = sample_llm.sentence

        # also, get the BERT predicted tags => convert readable format
        pred_tags = bert_pred.pred_tags_processed  # follows the BIO format
        toks = punc_tokenize(sentence=sent_cp)
        assert len(pred_tags) == len(toks)  # sanity check
        sample_bert = NerBioExample(sentence=sent_cp, tokens=tuple(toks), ner_tags=tuple(pred_tags)).to_readable()

        conf = bert_pred.conf
        # use `logprob` to mirror key name used in LLM logprob ranking
        pair = dataset.CompareSamplePair(sample1=sample_llm, sample2=sample_bert, d_log=dict(logprob=f'{conf:.3e}'))
        samples_compare.append((pair, conf))
    samples_compare = sorted(samples_compare, key=lambda x: x[1])  # sort by BERT-predicted confidence
    msg = 'Samples compared w/ & sorted by BERT annotation confidence'
    dataset.compare_samples(
        samples=[sample for (sample, conf) in samples_compare], allow_no_diff=True, prefix1='LLM', prefix2='BERT',
        logger=logger, verbose=True, msg=msg)

    triples_w_conf: List[Dict[str, Any]] = []  # each (sentence, entity span, entity type) triple in the dataset
    triples_w_conf_log = []  # additionally include logging info
    n_diff_span, n_diff_type, n_diff = 0, 0, 0
    n_diff_span_omit_i = 0
    for sample in conf_samples:
        ner_sample = sample.sample
        sent, enms, ets = ner_sample.sentence, ner_sample.entity_names, ner_sample.entity_types

        entity_confs = sample.bert_pred.entity_confs
        for enm, et, conf in zip(enms, ets, entity_confs):
            conf_ = conf.conf
            # `average_logprob` for consistent key name w/ LLM logprob ranking
            d = dict(sentence=sent, span=enm, entity_type=et, average_logprob=conf_)
            triples_w_conf.append(d)

            d_log = deepcopy(d)
            d_log['confidence'] = f'{conf_:.3e}'
            diff_span = conf.pred_tags != conf.llm_tags
            diff_type = conf.pred_type != et
            if diff_span:
                # different predicted tag doesn't necessarily mean different span processed, so log the predicted tags
                d_log['LLM-span'] = pl.i(conf.llm_tags, sep=' ')
                d_log['BERT-span'] = pl.i(conf.pred_tags, sep=' ')
                n_diff_span += 1
                # also count #difference if difference is not just starting `I-` instead of `B-`
                #   this is effectively comparing the processed chunks
                llm_chunks = tags2chunks(tags=conf.llm_tags)
                pred_chunks = tags2chunks(tags=conf.pred_tags)
                if llm_chunks != pred_chunks:
                    n_diff_span_omit_i += 1
                    d_log['diff-span'] = True

            if diff_type:
                d_log['BERT-type'] = conf.pred_type
                n_diff_type += 1
            if diff_span or diff_type:
                n_diff += 1
            triples_w_conf_log.append((d_log, conf_))
    out_fnm = os_join(output_path, 'logprobs-triple.json')  # write to file
    with open(out_fnm, 'w') as f:
        json.dump(triples_w_conf, f, indent=4)
    triples_w_conf_log = sorted(triples_w_conf_log, key=lambda x: x[1])  # sort by entity-wise BERT-predicted confidence
    triples_w_conf_log = [d for (d, conf) in triples_w_conf_log]
    d_log = {
        '#samples-different': {
            '#different-span': n_diff_span, 'different-span-when-omit-starting-I-tag': n_diff_span_omit_i,
            '#different-type': n_diff_type, '#different': n_diff},
        'samples': triples_w_conf_log
    }
    # a detailed log which potentially shows the discrepancies between LLM and BERT predictions
    logger.info(f'LLM entity annotation triplets sorted by BERT-predicted confidence: {pl.i(d_log, indent=3)}')

    triples_w_conf_log_: List[Tuple[str, str, str, float]] = [tuple(d.values()) for d in triples_w_conf]
    # a more compact log on the top-ranking triples for each entity type
    dataset.log_n_save_triples_w_logprob(triples_w_logprob=triples_w_conf_log_, entity_types=entity_types, logger=logger, top_n_log='all')


if __name__ == '__main__':
    # dnm = 'conll2003-no-misc'
    # dnm = 'wiki-gold-no-misc'
    dnm = 'mit-movie'
    # dnm = 'mit-restaurant'
    # dnm = 'job-stack'

    lower = dnm in ['mit-movie', 'mit-restaurant']

    def check_bert_uncertain():
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'

        # best-performing model after LLM self-correction
        # model_dir_nm = '24-01-27_19-32-54_{md=deberta,#ep=16,lr=4e-5}_{fmt=n-p2,#cr=3,de=T}_{t=0}_choice-wording'
        # the model trained w/o LLM self-correction, a more fair comparison?
        # model_dir_nm = '24-01-27_20-10-52_{md=deberta,#ep=16,lr=4e-5}_{fmt=n-p2,#l=3,de=T,lc=T}_#et=2.5'
        # above except after NER sentence extraction edge case
        # dir_nm = '24-01-28_NER-Dataset_{fmt=n-p2,#l=3,de=T,lc=T}_fix-pref'
        # use the version w/o lower-casing, so that the uncertain triplets for LLM to correct still have the original case info
        dir_nm = '24-01-23_NER-Dataset_{fmt=n-p2,#l=3,de=T}_fix-logprob-triple-dup'

        model_dir_nm = '24-01-28_14-03-32_{md=deberta,#ep=16,lr=4e-5}_{fmt=n-p2,#l=3,de=T,lc=T}_fix-pref'

        # model_dir_nm = '23-11-12_13-12-35_{md=deberta,#ep=16,lr=5e-5}_ori-train-data-all'  # best-performing supervised model
        ner_dataset2uncertain_triples(dataset_name=dnm, dataset_dir_name=dir_nm, model_dir_name=model_dir_nm, lowercase=lower)
    check_bert_uncertain()
