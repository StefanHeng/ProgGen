# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import logging
import argparse
from os.path import join as os_join
from typing import List, Tuple, Dict, Set, Union, Any
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from stefutil import *
from src.util.sample_formats import get_default_entity_pair_map
from src.util.ner_example import *
from src.data_util import eval
from src.trainer.utils_ner import get_tag_to_id, get_chunks, NerChunk, EvalDummyArgs


__all__ = ['EvalResult', 'evaluate']


_logger = get_logger(__name__)


entity_map = get_default_entity_pair_map(sample_format='natural-pair-v2')


def _spans2entities(
        sent: List[str] = None, spans: List[NerChunk] = None, entity_format: str = 'natural-pair-v2', prefix: str = None,
        merge: bool = False
) -> Dict[str, List[str]]:
    """
    :param sent: List of tokens that make up the sentence
    :param spans: List of entity span annotations
    :param entity_format: sample format name for templating named entities & types
    :param prefix: prefix for keys in the returned dict
    :param merge: If true, convert annotation list into a single string
        Intended for compact indentation in json output
    """
    enms, ets = [], []
    for span in spans:
        toks = sent[span.start:span.end]
        enm = detokenize(toks)
        enms.append(enm)
        ets.append(span.entity_type)

    if entity_format:
        if entity_format != 'natural-pair-v2':
            raise NotImplementedError
        else:
            enms = [entity_map(enm, et) for enm, et in zip(enms, ets)]
            if merge:
                enms = pl.nc(enms)
        ret = {'entities': enms}
    else:
        ret = {'entity-names': enms, 'entity-types': ets}
    if prefix:
        ret = {f'{prefix}-{k}': v for k, v in ret.items()}
    return ret


@dataclass
class PredictedSample:
    sentence: List[str] = None
    tags_pred: List[str] = None
    tags_true: List[str] = None
    spans_pred: List[NerChunk] = None  # sorted by (start, end)
    spans_true: List[NerChunk] = None,
    loss: float = None

    def to_readable(self, pred_prefix: str = 'pred', true_prefix: str = 'true') -> str:
        # Pad token and tag strings for easy reading
        max_len = max([  # maximum #char for padding
            max(len(t) for t in self.sentence),
            max(len(t) for t in self.tags_pred),
            max(len(t) for t in self.tags_true)
        ])
        max_len = max(max_len + 2, 10)

        def _pad(lst: List[str]) -> List[str]:
            return [t.ljust(max_len) for t in lst]

        def _map(x) -> str:
            return pl.i(x, with_color=False)
        # drop 'O' tags for readability
        tags_pred = ['' if t == 'O' else t for t in self.tags_pred]
        tags_true = ['' if t == 'O' else t for t in self.tags_true]

        n_tok = len(self.sentence)

        def span_add_dummies(spans: List[NerChunk]) -> List[str]:
            # introduce dummy elements for spans, so that visually aligns w/ rows above, for readability
            spans = sorted(spans, key=lambda x: (x.start, x.end))
            # by construction of prediction, spans are mutually exclusive
            spans_starts = [sp.start for sp in spans]
            spans_pred = []
            for i in range(n_tok):
                if i in spans_starts:
                    span = spans[spans_starts.index(i)]
                    spans_pred.append(pl.i(span.to_tuple(), with_color=False))
                else:
                    spans_pred.append('')
            return spans_pred

        prefixes = ['sentence', f'{pred_prefix} tags', f'{true_prefix} tags', f'{pred_prefix} spans', f'{true_prefix} spans']
        contents = [self.sentence, tags_pred, tags_true, span_add_dummies(self.spans_pred), span_add_dummies(self.spans_true)]
        s = ''
        for prefix, content in zip(prefixes, contents):
            s += f'{prefix.ljust(20)}: {_map(_pad(content))}\n'
        return s

    def to_json(self, pred_prefix: str = 'pred', true_prefix: str = 'true', merge: bool = False) -> Dict[str, Any]:
        # get corresponding entity names and entity types
        sent = detokenize(self.sentence)
        ret = dict(sentence=sent)
        ret.update(_spans2entities(sent=self.sentence, spans=self.spans_pred, prefix=pred_prefix, merge=merge))
        ret.update(_spans2entities(sent=self.sentence, spans=self.spans_true, prefix=true_prefix, merge=merge))
        return ret


@dataclass
class EvalResult:
    epoch: float = None
    step: int = None
    loss: float = None
    predictions: List[PredictedSample] = None
    precision: float = None
    recall: float = None
    f1: float = None
    f1_partial: float = None
    df_token: pd.DataFrame = None  # classification report for token-level
    df_entity: pd.DataFrame = None  # classification report for entity-level
    df_entity_partial: pd.DataFrame = None  # classification report for entity-level that counts partial match as 0.5 true positive
    split: str = None
    save_path: str = None
    logger: logging.Logger = None

    def to_performance_dict(self, prefix: str = None, keys: List[str] = None) -> Dict[str, Any]:
        """
        For eval logging during training
        """
        keys = keys or ['loss', 'precision', 'recall', 'f1']
        ret = {k: v for k, v in asdict(self).items() if k in keys}
        if prefix:
            ret = {f'{prefix}_{k}': v for k, v in ret.items()}
        return ret

    def to_f1_log(self, zero_indexed: bool = True, digit: int = None, as_percentage: bool = None) -> Dict[str, float]:
        """
        for best performance logging after training
        """
        def metric_map(x: float) -> float:
            if as_percentage:
                x *= 100
            if digit is not None:
                x = round(x, digit)
            return x
        ret = dict()
        if self.epoch:
            ret['epoch'] = self.epoch + 1 if zero_indexed else self.epoch
        if self.step:
            ret['step'] = self.step
        ret['f1'] = metric_map(self.f1)
        if self.f1_partial:
            ret['partial-f1'] = metric_map(self.f1_partial)
        return ret

    def entity_wise_f1(self) -> Dict[str, float]:
        return eval.cls_report_df2class_wise_scores(df=self.df_entity)

    def write_reports(self, block_std: List[str] = None):
        logger = self.logger or _logger
        block = block_std or ['token']
        for kd, df in zip(['token', 'entity', 'entity-partial'], [self.df_token, self.df_entity, self.df_entity_partial]):
            eval.write_cls_report(df=df, kind=kd, split=self.split, output_path=self.save_path, block_std=block, logger=logger)

    def write_predictions(self, pred_prefix: str = 'pred', true_prefix: str = 'true', sort_by_loss: bool = True, filename: str = None):
        logger = self.logger or _logger
        fnm = filename or f'{self.split}-predictions'
        path_pred = os_join(self.save_path, fnm)
        with open(f'{path_pred}-raw.json', 'w') as f:  # a strict format with all info
            json.dump([asdict(p) for p in self.predictions], f, indent=2)

        # more readable formats for easy examination
        with open(f'{path_pred}.json', 'w') as f:  # a strict format with all info
            json.dump([p.to_json(pred_prefix=pred_prefix, true_prefix=true_prefix, merge=True) for p in self.predictions], f, indent=2)
        with open(f'{path_pred}.jsonl', 'w') as f:
            for p in self.predictions:
                json.dump(p.to_json(pred_prefix=pred_prefix, true_prefix=true_prefix), f)
                f.write('\n')
        with open(f'{path_pred}.log', 'w') as f:
            preds = self.predictions
            if sort_by_loss:  # sort in descending order of loss
                idxs = np.argsort([p.loss for p in preds])[::-1]
                # drop writing index to index in original dataset
                preds = [preds[i] for i in idxs]
                idxs_write2orig = {i: idx for i, idx in enumerate(idxs)}

                def idx2prefix(idx: int) -> str:
                    loss = round(preds[idx].loss, 4)
                    return f'{idx+1}, {pl.i(dict(original_index=idxs_write2orig[idx]+1, loss=loss), with_color=False)}'
            else:
                def idx2prefix(idx: int) -> str:
                    return f'{idx+1}'
            for i, p in enumerate(preds):
                f.write(f'{idx2prefix(i)}:\n{p.to_readable(pred_prefix=pred_prefix, true_prefix=true_prefix)}\n\n')
        logger.info(f'Test predictions saved to {pl.i(fnm)}')


def evaluate(
        args: Union[argparse.Namespace, EvalDummyArgs] = None,
        model: torch.nn.Module = None,
        labels: List[str] = None,
        entity_types: List[str] = None,
        pad_token_label_id: int = None,
        eval_dataset: TensorDataset = None,
        best: EvalResult = None,
        ignore_labels: List[str] = None,
        prefix: str = "",
        epoch: float = None,
        step: int = None,
        verbose: bool = True,
        split: str = None,
        logger: logging.Logger = None,
        texts: List[List[str]] = None,
        swap_pred_true: bool = False
) -> Tuple[EvalResult, EvalResult, bool]:
    """
    :param args: Command prompt arguments
    :param model: NER model
    :param labels: list of labels, indexing corresponds to model output
    :param entity_types: list of entity types
    :param ignore_labels: list of labels to ignore in evaluation
        Intended for zero-out new entity types
    :param pad_token_label_id: label id for padding
    :param best: EvalResult for best performance
    :param eval_dataset: evaluation dataset
    :param prefix: prefix for logging
    :param epoch: epoch number, intended for logging
    :param step: step number, intended for logging
    :param verbose: If true, eval dataset stats are logged
    :param split: Dataset split name
    :param logger: If given, use this logger for logging
    :param texts: Texts corresponding to eval_dataset, intended for logging
    :param swap_pred_true: If true, swap the values between pred and true
        Intended for `check_annotation_accuracy`, where
            predictions from supervised model are used as ground truth, and
            LLM annotations in the dataset file are considered as predictions
    """
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    #if args.n_gpu > 1:
    #    model = torch.nn.DataParallel(model)
    #model.to(args.device)
    logger = logger or _logger

    if prefix:
        prefix = f' {prefix}'  # add space for readability
    if verbose:
        d_log = {'#example': len(eval_dataset), 'batch size': args.eval_batch_size}
        logger.info(f"***** Running evaluation{prefix} *****\n{pl.fmt(d_log)}")
        # logger.info("  Num examples = %d", len(eval_dataset))
        # logger.info("  Batch size = %d", args.eval_batch_size)
        # logger.info(f'  {pl.i(d_log)}')
    eval_loss = 0.0
    nb_eval_steps = 0
    preds_dist = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc=f"Eval{prefix}", unit='ba'):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds_dist is None:
            preds_dist = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds_dist = np.append(preds_dist, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if ignore_labels is not None:
        idxs_zero = [labels.index(lb) for lb in ignore_labels]
        # pred is in shape (n_batch, max_seq_len, n_labels)
        preds_dist[:, :, idxs_zero] = -np.inf
    preds = np.argmax(preds_dist, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    out_id_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_id_list = [[] for _ in range(out_label_ids.shape[0])]
    lst_loss = []  # averaged loss for each sample

    cel = torch.nn.CrossEntropyLoss(ignore_index=pad_token_label_id)
    for i in range(out_label_ids.shape[0]):
        preds_dist_, lbs_id_ = [], []
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                pred_, lb_id = preds[i][j], out_label_ids[i][j]
                preds_list[i].append(label_map[pred_])
                out_id_list[i].append(lb_id)
                preds_id_list[i].append(pred_)

                preds_dist_.append(preds_dist[i][j])
                lbs_id_.append(lb_id)
        loss = cel(torch.tensor(preds_dist_), torch.tensor(lbs_id_))  # compute loss for current sample
        lst_loss.append(loss.item())
    correct_preds, total_correct, total_preds = 0., 0., 0.  # i variables
    tag2id = get_tag_to_id(path=args.data_dir, dataset_name=args.dataset_name)

    if swap_pred_true:
        logger.warning(f'Swapping pred and true for evaluation, intended for {pl.i("check_annotation_accuracy")} only')
    preds_s: List[PredictedSample] = []
    for i, (ground_truth_id, predicted_id, loss_) in enumerate(zip(out_id_list, preds_id_list, lst_loss)):
        # We use the get chunks function defined above to get the true chunks
        # and the predicted chunks from true labels and predicted labels respectively
        lab_chunks: Set[NerChunk] = set(get_chunks(ground_truth_id, tag2id=tag2id))
        lab_pred_chunks: Set[NerChunk] = set(get_chunks(predicted_id, tag2id=tag2id))

        true_ids: List[int] = out_id_list[i]
        preds_str, trues_str = preds_list[i], [labels[i_] for i_ in true_ids]
        spans_pred = sorted(lab_pred_chunks, key=lambda x: (x.start, x.end))
        spans_true = sorted(lab_chunks, key=lambda x: (x.start, x.end))
        if swap_pred_true:
            preds_str, trues_str = trues_str, preds_str
            spans_pred, spans_true = spans_true, spans_pred
        preds_s.append(PredictedSample(
            sentence=texts[i] if texts is not None else None,
            tags_pred=preds_str, tags_true=trues_str, spans_pred=spans_pred, spans_true=spans_true, loss=loss_
        ))

        # Updating the i variables
        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    lst_pred_int = [tag2id[lb] for lb in sum(preds_list, start=[])]
    lst_true_int = sum(out_id_list, start=[])
    trues_list: List[List[str]] = [[labels[i] for i in row] for row in out_id_list]
    if swap_pred_true:
        lst_pred_int, lst_true_int = lst_true_int, lst_pred_int
        preds_list, trues_list = trues_list, preds_list

    df_tok = eval.get_token_cls_report(trues=lst_true_int, preds=lst_pred_int, labels=labels, ignore_labels=ignore_labels)
    ent_out = eval.get_entity_cls_report(trues=trues_list, preds=preds_list, entity_types=entity_types)
    df_ent, df_ent_partial = ent_out.exact, ent_out.partial

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    new_f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    f1_ent = df_ent['f1-score']['micro avg']  # sanity check f1 computation (raw & from library) are the same
    assert new_f1 == f1_ent
    res = EvalResult(
        epoch=epoch, step=step, loss=eval_loss, predictions=preds_s,
        precision=p, recall=r, f1=new_f1, f1_partial=df_ent_partial['f1-score']['micro avg'],
        df_token=df_tok, df_entity=df_ent, df_entity_partial=df_ent_partial,
        split=split, save_path=args.output_dir, logger=logger
    )

    is_updated = False
    if best is None or new_f1 > best.f1:
        # best = [p, r, new_f1]
        best = res
        is_updated = True

    # results = {
    #    "loss": eval_loss,
    #    "precision": p,
    #    "recall": r,
    #    "f1": new_f1,
    #    "best_precision": best[0],
    #    "best_recall": best[1],
    #    "best_f1": best[-1]
    # }

    # logger.info("***** Eval results %s *****", prefix)
    # for key in sorted(results.keys()):
    #     logger.info("  %s = %s", key, str(results[key]))

    # return results, preds_list, best, is_updated
    return res, best, is_updated