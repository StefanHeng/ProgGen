import os
import json
import random
import argparse
from os.path import join as os_join
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from stefutil import *
from src.util import *
import src.util.ner_example as ner_utils


logger = get_logger(__name__)


__all__ = [
    'set_seed', 'get_output_dir',
    'InputExample', 'InputFeatures', 'read_examples_from_file', 'convert_examples_to_features', 'load_and_cache_examples', 'get_ner_tags', 'get_tag_to_id',
    'get_chunks', 'NerChunk',
    'round_metric',
    'EvalDummyArgs', 'Args'
]


def set_seed(args):
    logger.info(f'Setting seed to {pl.i(args.seed)}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def abbreviate_model_name(model_name: str) -> str:
    """
    :param model_name: BERT model name on hugging face
    """
    if model_name == 'bert-base-cased':
        return 'bert'
    elif model_name == 'bert-base-uncased':
        return 'bert-uncased'
    elif model_name == 'roberta-base':
        return 'roberta'
    elif model_name == 'microsoft/deberta-v3-base':
        return 'deberta'
    else:
        if '/' in model_name:
            nms = model_name.split('/')
            assert len(nms) == 2
            return nms[1]
        else:
            return model_name


def get_output_dir(args: argparse.Namespace, as_path: bool = True) -> str:
    now_ = now(fmt='short-full', for_path=True)
    n_ep = args.num_train_epochs
    if n_ep.is_integer():
        n_ep = int(n_ep)
    d = {'md': abbreviate_model_name(model_name=args.model_type), '#ep': n_ep, 'lr': args.learning_rate}
    ret = f'{now_}_{pl.pa(d)}'
    if args.prefix:
        ret = f'{ret}_{args.prefix}'

    if as_path:
        ret = os_join(pu.model_path, args.task, ret)
    return ret


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid: str, words: List[str], labels: List[str]):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels

    def __repr__(self):
        return f'{self.__class__.__name__}(id={self.guid}, words={self.words}, labels={self.labels})'


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, full_label_ids):
        
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.full_label_ids = full_label_ids


def read_examples_from_file(args: argparse.Namespace, data_dir: str, mode: str) -> List[InputExample]:
    def fnm2path(filename: str) -> str:
        options = [filename, os_join(data_dir, filename), os_join(data_dir, '..', filename)]

        for path in options:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f'Dataset File {pl.i(filename)} not found from {pl.i(options, indent=1)}')

    if mode == 'train':
        # file_path = os.path.join(data_dir, args.train_file)
        file_path = fnm2path(args.train_file)
    elif mode == 'valid':
        # file_path = os.path.join(data_dir, args.validation_file)
        file_path = fnm2path(args.validation_file)
    elif mode == 'test':
        # file_path = os.path.join(data_dir, args.test_file)
        file_path = fnm2path(args.test_file)
    elif mode == 'few':
        # file_path = os.path.join(data_dir, args.few_file or "train_few.jsonl")
        file_path = fnm2path(args.few_file or "train_few.jsonl")
    elif mode == 'unlabeled':
        assert args.unlabeled_file is not None
        file_path = fnm2path(args.unlabeled_file)
    else:
        # file_path = os.path.join(data_dir, "{}.json".format(mode))
        file_path = fnm2path(f'{mode}.json')
    guid_index = 1
    examples = []

    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # for item in data:
            # sic(file_path, item)
            words = item["tokens"]
            labels = item["labels"]
            
            examples.append(InputExample(guid=f'{mode}-{guid_index}', words=words, labels=labels))
            guid_index += 1
    
    return examples


def convert_examples_to_features(
    tags_to_ids, 
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    show_exnum=1,
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    extra_long_samples = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info(f"Writing example {pl.i(ex_index)} of {pl.i(len(examples))}")

        tokens = []
        label_ids = []
        full_label_ids = []
        # hp_label_ids = []
        # print(example.words, example.labels)
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)  # split into sub-word tokens
            label = tags_to_ids[label]
            if len(word_tokens) == 0:
                continue
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            # TODO: why is there both `label_ids` and `full_label_ids`?
            label_ids.extend([label] + [pad_token_label_id] * (len(word_tokens) - 1))
            # hp_label_ids.extend([hp_label if hp_label is not None else pad_token_label_id] + [pad_token_label_id] * (len(word_tokens) - 1))
            full_label_ids.extend([label] * len(word_tokens))
        
        # print(tokens, label_ids, full_label_ids)
        # assert 0
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 2  # TODO: so didn't handle RoBERTa?
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            # hp_label_ids = hp_label_ids[: (max_seq_length - special_tokens_count)]
            full_label_ids = full_label_ids[: (max_seq_length - special_tokens_count)]
            extra_long_samples += 1

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        # hp_label_ids += [pad_token_label_id]
        full_label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        # hp_label_ids = [pad_token_label_id] + hp_label_ids
        full_label_ids = [pad_token_label_id] + full_label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
       
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length
        # hp_label_ids += [pad_token_label_id] * padding_length
        full_label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        # assert len(hp_label_ids) == max_seq_length
        assert len(full_label_ids) == max_seq_length

        if ex_index < show_exnum:
            # logger.info("*** Example ***")
            # logger.info("guid: %s", example.guid)
            # logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            # logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            # logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            # logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            # logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            # logger.info("full_label_ids: %s", " ".join([str(x) for x in full_label_ids]))
            d_eg = dict(
                guid=example.guid,
                tokens=' '.join([str(x) for x in tokens]),
                input_ids=' '.join([str(x) for x in input_ids]),
                input_mask=' '.join([str(x) for x in input_mask]),
                segment_ids=' '.join([str(x) for x in segment_ids]),
                label_ids=' '.join([str(x) for x in label_ids]),
                full_label_ids=' '.join([str(x) for x in full_label_ids])
            )
            logger.info(f'*** Example ***\n{pl.fmt(d_eg)}')

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids, full_label_ids=full_label_ids)
        )
    logger.info(f"Found {pl.i(extra_long_samples)}/{pl.i(len(examples))} extra long examples")
    return features


@dataclass
class LoadDatasetOutput:
    tensor_dataset: TensorDataset = None
    texts: List[List[str]] = None


def load_and_cache_examples(
        args=None, tokenizer=None, labels: List[str] = None, pad_token_label_id: int = None, mode: str = None,
        remove_labels=False, tag2id: Dict[str, int] = None,
        return_texts: bool = False
) -> LoadDatasetOutput:
    
    # if args.local_rank not in [-1, 0] and not evaluate:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    
    # cached_features_file = os.path.join(
    #     args.data_dir,
    #     "cached_{}_{}_{}".format(
    #         mode, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)
    #     ),
    # )
    cached_features_file = None
    if args.cache_datasets:
        _md_nm = list(filter(None, args.model_name_or_path.split("/"))).pop()
        cached_features_file = os.path.join(
                args.cache_dir,
                # 'cached_{}_{}_{}_{}_{}.pt'.format(
                #     mode,
                #     args.task,
                #     list(filter(None, args.model_name_or_path.split("/"))).pop(),
                #     args.max_seq_length,
                #     args.prefix
                # )
                f'cached_{{dset={args.task}-{mode},md={_md_nm},l={args.max_seq_length},{args.prefix}}}.pt'
            )
    txts = None
    examples = read_examples_from_file(args, args.data_dir, mode)
    if args.cache_datasets and os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info(f"Loading features from cached file {pl.i(cached_features_file)}")
        features = torch.load(cached_features_file)
    else:
        logger.info(f"Creating features from dataset file at {pl.i(args.data_dir)}")
        features = convert_examples_to_features(
            tag2id,
            examples,
            labels,
            args.max_seq_length,
            tokenizer,
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id=pad_token_label_id,
        )
        if args.cache_datasets and args.local_rank in [-1, 0]:
            logger.info(f"Saving features into cached file {pl.i(cached_features_file)}")
            torch.save(features, cached_features_file)

    # if args.local_rank == 0 and not evaluate:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_full_label_ids = torch.tensor([f.full_label_ids for f in features], dtype=torch.long)
    # all_hp_label_ids = torch.tensor([f.hp_label_ids for f in features], dtype=torch.long)
    if remove_labels:  # TODO: what is this for? semi-supervised learning how?
        all_full_label_ids.fill_(pad_token_label_id)
        # all_hp_label_ids.fill_(pad_token_label_id)
    all_ids = torch.tensor([f for f in range(len(features))], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_full_label_ids, all_ids)

    if return_texts:
        txts = [ex.words for ex in examples]
    return LoadDatasetOutput(tensor_dataset=dataset, texts=txts)


def get_ner_tags(path: str = None, verbose: bool = False, dataset_name: str = 'conll2003') -> List[str]:
    """
    e.g. ["O", "B-LOC", "B-ORG", "B-PER", "B-MISC", "I-PER", "I-MISC", "I-ORG", "I-LOC"]
    """
    if path is not None:
        path = os_join(path, "tag_to_id.json")
    if path and os.path.exists(path):
        if verbose:
            logger.info(f'Loading NER tags from {pl.i(path)}')
        labels = []
        with open(path, "r") as f:
            data = json.load(f)
            for lb, _ in data.items():
                labels.append(lb)
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        if verbose:
            logger.info(f'Path {pl.i(path)} not found, using default NER tags for dataset {pl.i(dataset_name)}')
        # return
        dnm = dataset_name.replace('_', '-')
        types = sconfig(f'datasets.{dnm}.readable-entity-types')
        return ner_utils.ner_labels2tags(entity_types=types)


def get_tag_to_id(path: str = None, verbose: bool = False, dataset_name: str = 'conll2003') -> Dict[str, int]:
    """
    e.g. {"O": 0, "B-LOC": 1, "B-ORG": 2, "B-PER": 3, "B-MISC": 4, "I-PER": 5, "I-MISC": 6, "I-ORG": 7, "I-LOC": 8}
    """
    if path is not None:
        path = os_join(path, "tag_to_id.json")
    if path and os.path.exists(path):
        if verbose:
            logger.info(f'Loading NER tag map from {pl.i(path)}')
        with open(path, 'r') as f:
            data = json.load(f)
        ret = data
    else:
        if verbose:
            logger.info(f'Path {pl.i(path)} not found, using default NER tag map for dataset {pl.i(dataset_name)}')
        dnm = dataset_name.replace('_', '-')
        types = sconfig(f'datasets.{dnm}.readable-entity-types')
        ret = ner_utils.ner_labels2tag2index(entity_types=types)
    if verbose:
        logger.info(f'NER tag map: {pl.i(ret, indent=True)}')
    return ret


def get_chunk_type(tok: int, idx_to_tag: Dict[int, str]) -> Tuple[str, str]:
    """
    The function takes in a chunk ("B-PER") and then splits it into the tag (PER) and its class (B)
    as defined in BIOES

    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


@dataclass(eq=True, frozen=True)
class NerChunk:
    entity_type: str = None
    start: int = None
    end: int = None

    def to_tuple(self) -> Tuple[str, int, int]:
        return self.entity_type, self.start, self.end


def get_chunks(seq: List[int], tag2id: Dict[str, int], sort: bool = True) -> List[NerChunk]:
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tag2id: dict["O"] = 4
        sort: bool, if True, sort by occurrence

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tag2id["O"]
    idx_to_tag = {idx: tag for tag, idx in tag2id.items()}
    chunks = []

    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        if tok == default and chunk_type is not None:
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    ret = [NerChunk(entity_type=kd, start=start, end=end) for (kd, start, end) in chunks]
    if sort:
        ret = sorted(ret, key=lambda x: (x.start, x.end))
    return ret


def round_metric(m: float, n=4) -> float:
    return round(m * 100, n)


@dataclass
class EvalDummyArgs:
    """
    Intended for matching api for calls used in check_annotation_accuracy`, e.g. `evaluate`
    """
    per_gpu_eval_batch_size: int = 128
    local_rank: int = -1
    eval_batch_size: int = 128
    no_cuda: bool = False
    device = torch.device('cuda' if (torch.cuda.is_available() and not no_cuda) else 'cpu')
    n_gpu = torch.cuda.device_count()
    model_type: str = 'bert-base-cased'
    data_dir: str = None
    output_dir: str = None
    dataset_name: str = None
    train_file: str = None
    cache_dir: str = None
    overwrite_cache: bool = False
    cache_datasets: bool = None
    task: str = None
    max_seq_length: int = 128
    prefix: str = None


Args = Union[EvalDummyArgs, argparse.Namespace]
