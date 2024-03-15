import logging
from os.path import join as os_join
from typing import List, Dict, Any
from dataclasses import dataclass

import pandas as pd
from nervaluate import Evaluator

from stefutil import pl


__all__ = [
    'get_token_cls_report', 'get_entity_cls_report', 'EntityClsReport',
    'write_cls_report', 'cls_report_df2class_wise_scores'
]


def get_token_cls_report(
        trues: List[int], preds: List[int], labels: List[str], ignore_labels: List[str] = None
) -> pd.DataFrame:
    from stefutil import eval_array2report_df
    idx_lbs = list(range(len(labels)))
    args = dict(labels=idx_lbs, target_names=labels, zero_division=0, output_dict=True)
    df, _ = eval_array2report_df(labels=trues, preds=preds, report_args=args, pretty=False)
    if ignore_labels:  # drop the rows for newly added tags
        df = df[~df.index.isin(ignore_labels)]
    return df


PARTIAL_SCORING_TYPES = ['strict', 'ent_type']
SCORING_TYPE2OUTPUT_TYPE = dict(strict='exact', ent_type='partial')


def _get_entity_cls_report_ne(
        trues: List[List[str]] = None, preds: List[List[str]] = None, entity_types: List[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Converts the output of nervaluate.Evaluator.evaluate() to sklearn classification report format list of dicts

    Uses the `nerevaluate` library for entity-level evaluation
        The library implements scoring for the SemEval-2013 Task 9
    We use it to compute a partial match score, which is not implemented in `seqeval`

    The formula is given in https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
    This metric is also used in the UniversalNER paper

    :return: sklearn-style classification report dicts for strict and partial match
    """
    def _ne_eval_res2dict(res: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert evaluation result from the `nerevaluate` library to sklearn classification report format

        scoring type => count/metric type => count/metric value
            scoring type are for the SemEval-2013 Task 9: strict, exact boundary match, partial boundary match, type match
                In terms of the library, these types are called `strict`, `exact`, `partial`, and `ent_type` respectively.

        We are interested only in 2 scoring types:
            1. `strct`: sanity check the same score as in `seqeval` output, and
                We call this `strict`
            2. `ent_type`: for partial match: counts as true positive of 0.5 when the entity type matches but the boundaries overlap
                We call this `partial`

        We only keep `precision`, `recall`, and `f1` for the interested scoring types
        We also get `support`, which is the number of ground truth entities, given in the library by `
        """
        keys = list(res.keys())
        assert set(keys) == {'strict', 'exact', 'partial', 'ent_type'}  # sanity check
        metric_keys = ['precision', 'recall', 'f1', 'possible']
        mk2mk = {'precision': 'precision', 'recall': 'recall', 'f1': 'f1-score', 'possible': 'support'}
        return {s_t: {mk2mk[mk]: res[s_t][mk] for mk in metric_keys} for s_t in PARTIAL_SCORING_TYPES}
    res_, res_by_type = Evaluator(trues, preds, tags=entity_types, loader='list').evaluate()
    res_ = _ne_eval_res2dict(res_)
    res_by_type = {tp: _ne_eval_res2dict(res) for tp, res in res_by_type.items()}
    assert set(res_by_type.keys()) == set(entity_types)  # sanity check

    ret = dict()
    for st in PARTIAL_SCORING_TYPES:
        # d = {tp: res_by_type[tp][st] for tp in res_by_type.keys()}
        d = {tp: res_by_type[tp][st] for tp in entity_types}  # order by the entity types ordering
        d.update({'micro avg': res_[st]})

        # TODO: compute `macro avg` and `weighted avg`?
        ret[SCORING_TYPE2OUTPUT_TYPE[st]] = d
    return ret


@dataclass
class EntityClsReport:
    exact: pd.DataFrame = None
    partial: pd.DataFrame = None


def get_entity_cls_report(
        trues: List[List[str]] = None, preds: List[List[str]] = None, entity_types: List[str] = None
) -> EntityClsReport:
    import seqeval.metrics  # lazy import to save time
    # get exact match scores using `seqeval`
    df_se = seqeval.metrics.classification_report(trues, preds, digits=4, output_dict=True)

    # get exact match & partial match scores using `nerevaluate`
    out = _get_entity_cls_report_ne(trues=trues, preds=preds, entity_types=entity_types)
    df_ne_exact, df_ne_partial = out['exact'], out['partial']
    # sic(df_se, df_ne_exact, df_ne_partial)

    # sanity check that the exact match scores are the same
    # for k in entity_types + ['micro avg']:  # `macro avg` and `weighted avg` omitted since they are not computed in `nerevaluate`
    #     assert df_se[k] == df_ne_exact[k]
    # looks like `seqeval` and `nerevaluate` compute per-class metrics differently, but the final micro avg score should be the same
    #   also observed in the blog post https://skeptric.com/ner-evaluate/index.html
    #       > Note there’s a discrepancy here; the strict f1 for WORK_OF_ART is 79.1%, when seqeval gave 80.3%.
    #       This is because seqeval ignores the other types of tags when evaluating at a tag level, but nervaluate includes them.
    if df_se['micro avg'] != df_ne_exact['micro avg']:
        from stefutil import sic
        sic(df_se, df_ne_exact)
        sic(df_se['micro avg'], df_ne_exact['micro avg'])
    assert df_se['micro avg'] == df_ne_exact['micro avg']

    keys = entity_types + ['micro avg', 'macro avg', 'weighted avg']  # order the class-wise scores by the entity types ordering
    df_se = {k: df_se[k] for k in keys}

    # use the `seqeval` scores for exact match, and the `nerevaluate` scores for partial match
    df_exact = pd.DataFrame(df_se).transpose()
    df_partial = pd.DataFrame(df_ne_partial).transpose()
    # surprisingly, the `support` column is a float after construction, convert it to int
    df_exact['support'] = df_exact.support.astype(int)
    df_partial['support'] = df_partial.support.astype(int)
    return EntityClsReport(exact=df_exact, partial=df_partial)


def write_cls_report(
        df: pd.DataFrame = None, kind: str = 'entity', split: str = None, output_path: str = None,
        block_std: List[str] = None, logger: logging.Logger = None
):
    d = dict(split=split) if split else dict()
    d['kd'] = kind
    fnm = f'Cls-report_{pl.pa(d)}.csv'
    df.to_csv(os_join(output_path, fnm))
    msg = f'{pl.i(kind)} classification report'
    if split:
        msg = f'{split.capitalize()} {msg}'

    df.loc[:, df.columns != 'support'] *= 100  # convert all metrics to percentage
    with pd.option_context('display.float_format', '{:.1f}'.format):  # print the percentages to 1 decimal place
        msg = f'{msg}: \n{pl.i(df)}\n'
        msg = f'{msg}saved to {pl.i(fnm)}'
        # block output to stdout for token cls report
        logger.info(msg, extra=dict(block='stdout' if kind in (block_std or []) else None))


def cls_report_df2class_wise_scores(
        df: pd.DataFrame = None, metric: str = 'f1-score', keys_ignore: List[str] = None, to_percent: bool = True, decimal: int = 1
) -> Dict[str, float]:
    keys_ignore = keys_ignore or ['micro avg', 'macro avg', 'weighted avg']
    df = df[~df.index.isin(keys_ignore)]
    ret = df[metric].to_dict()

    if to_percent:
        if all(0 <= v <= 1 for v in ret.values()):  # if scores are in [0, 1], convert to percentage
            ret = {k: v * 100 for k, v in ret.items()}
        else:
            assert all(0 <= v <= 100 for v in ret.values())  # sanity check already in percentage
    return {k: round(v, decimal) for k, v in ret.items()}  # round to 1 decimal
