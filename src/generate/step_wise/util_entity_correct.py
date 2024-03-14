import os
import re
import json
import random
import logging
from os.path import join as os_join
from typing import Dict, Tuple, List, Union, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict

from stefutil import *
from src.util import *
from src.util import sample_check as check
from src.util.ner_example import *
from src.data_util import *


__all__ = [
    'CORRECTION_CLS_LABELS', 'CorrectionLabel', 'LABEL_CORRECT', 'LABEL_WRONG_BOUNDARY', 'LABEL_WRONG_TYPE', 'LABEL_NOT_NAMED_ENTITY',
    'STR_NOT_NAMED_ENTITY',
    'entity_type2entity_correction_options',
    'EntityCorrectionSample',

    'CORRECTION_DIR_NAME', 'ENTITY_TYPE_OTHER',
    'log_correction_samples', 'load_corrections', 'override_w_manual_corrections', 'log_n_save_corrections',
    'load_triples_w_logprob', 'UncertainTriplesOutput', 'log_n_select_uncertain_triples',
    'CorrectionSample', 'get_classification_report_row', 'ner_sample2corrected_sample'
]


_logger = get_logger('Correction Util')


_LABEL_CORRECT = '__correct__'
_LABEL_WRONG_BOUNDARY = '__wrong_boundary__'
_LABEL_WRONG_TYPE = '__wrong_type__'
_LABEL_NOT_NAMED_ENTITY = '__not_named_entity__'
_label2short_label = {
    _LABEL_CORRECT: 'CORRECT',
    _LABEL_WRONG_BOUNDARY: 'WRONG_SPAN',
    _LABEL_WRONG_TYPE: 'WRONG_TYPE',
    _LABEL_NOT_NAMED_ENTITY: 'NA'
}
_short_label2label = {v.lower(): k for k, v in _label2short_label.items()}  # reverse mapping
_CLS_LABELS = [_LABEL_CORRECT, _LABEL_WRONG_BOUNDARY, _LABEL_WRONG_TYPE, _LABEL_NOT_NAMED_ENTITY]


_label2template = {
    _LABEL_CORRECT: {
        'named entity':
            # 'is a named entity of type {entity_type}',
            'the span is a named entity of type {entity_type}',
        # 'term': 'Is a term of type {entity_type}',
    },
    _LABEL_WRONG_BOUNDARY: {
        'named entity':
            # 'contains a named {entity_type} entity but the span boundary is not precise',
            # 'the span contains a named {entity_type} entity but the boundary is not precise',
            'the span contains a {entity_type_full} but the span boundary is not precise',
        # 'term': 'Contains a term of type {entity_type} but the span boundary is not precise',
    },
    _LABEL_WRONG_TYPE: {
        'named entity':
            # 'is a named entity but category is not {entity_type}',
            # 'the span is a named entity but category is not {entity_type}',
            'the span is a named entity but the category is not {entity_type}',
        # 'term': 'Is a term but category is not {entity_type}',
    },
    _LABEL_NOT_NAMED_ENTITY: {
        'named entity':
            # 'not a named entity',
            'the span is not a named entity',
        # 'term': 'Not a term',
    }
}


_ordinals = 'ABCD'
_pattern_choice = re.compile(r'\(?(?P<ordinal>[A-D])\)?\.?')

_correction_label2str = {
    _LABEL_CORRECT: 'Correct Entity Annotation',
    _LABEL_WRONG_BOUNDARY: 'Wrong Boundary',
    _LABEL_WRONG_TYPE: 'Wrong Type',
    _LABEL_NOT_NAMED_ENTITY: 'Not a Named Entity'
}


@dataclass(eq=True, frozen=True)
class CorrectionLabel:
    label: str = None

    def to_ordinal(self) -> str:
        ret = _ordinals[_CLS_LABELS.index(self.label)]
        return f'({ret})'

    def to_short_label(self, lowercase: bool = False) -> str:
        ret = _label2short_label[self.label]
        if lowercase and self != LABEL_NOT_NAMED_ENTITY:
            ret = ret.lower()
        return ret

    def to_desc(self, entity_type: str = None, element_name: str = 'named entity') -> str:
        tpl = get(_label2template, f'{self.label}.{element_name}')
        k = 'entity_type_full' if self.label == _LABEL_WRONG_BOUNDARY else 'entity_type'
        return tpl.format(**{k: entity_type})

    @staticmethod
    def choice_to_label(choice: str = None) -> 'CorrectionLabel':
        m = _pattern_choice.match(choice)
        assert m is not None
        choice = m.group('ordinal')
        idx = _ordinals.index(choice)
        return CORRECTION_CLS_LABELS[idx]

    def to_str(self) -> str:
        return _correction_label2str[self.label]

    def __lt__(self, other):
        return _CLS_LABELS.index(self.label) < _CLS_LABELS.index(other.label)


LABEL_CORRECT = CorrectionLabel(label=_LABEL_CORRECT)
LABEL_WRONG_BOUNDARY = CorrectionLabel(label=_LABEL_WRONG_BOUNDARY)
LABEL_WRONG_TYPE = CorrectionLabel(label=_LABEL_WRONG_TYPE)
LABEL_NOT_NAMED_ENTITY = CorrectionLabel(label=_LABEL_NOT_NAMED_ENTITY)
CORRECTION_CLS_LABELS = [LABEL_CORRECT, LABEL_WRONG_BOUNDARY, LABEL_WRONG_TYPE, LABEL_NOT_NAMED_ENTITY]
STR_NOT_NAMED_ENTITY = '__Not-Entity__'


def entity_type2entity_correction_options(
        entity_type: str = None, entity_type_full: str = None, element_name: str = 'named entity', with_punc: bool = False,
) -> List[str]:
    ret = []
    n = len(CORRECTION_CLS_LABELS)
    for i, x in enumerate(CORRECTION_CLS_LABELS):
        et = entity_type_full if x == LABEL_WRONG_BOUNDARY else entity_type
        option = f'{x.to_ordinal()}. {x.to_desc(entity_type=et, element_name=element_name).capitalize()}'
        if with_punc:
            is_last = i == n - 1
            if is_last:
                option = f'{option}.'
            else:
                option = f'{option};'
        ret.append(option)
    return ret


@dataclass
class EntityCorrectionSample:
    sentence: str = None
    entity_span: str = None
    entity_type: str = None
    label: CorrectionLabel = None
    correct_span: str = None
    correct_type: str = None
    reason: str = None

    def to_readable_label(self, element_type: str = 'named entity', with_desc: bool = False) -> str:
        lb = self.label
        ret = f'{lb.to_ordinal()}.'
        if with_desc:
            ret = f'{ret} {lb.to_str()}.'
        if lb == LABEL_WRONG_BOUNDARY:
            ret = f'{ret} The correct span boundary is {edit.enclose_in_quote(self.correct_span)}.'
        elif lb == LABEL_WRONG_TYPE:
            if element_type == 'named entity':
                tp = 'entity type'
            else:
                assert element_type == 'term'
                tp = 'category'
            ret = f'{ret} The correct {tp} is {self.correct_type}.'
        return ret

    def to_dict(self):
        # the label is an objects, convert it to a string
        return dict(
            sentence=self.sentence, entity_span=self.entity_span, entity_type=self.entity_type,
            label=self.label.to_short_label(), correct_span=self.correct_span, correct_type=self.correct_type, reason=self.reason
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EntityCorrectionSample':
        lb = d.pop('label')
        if not isinstance(lb, CorrectionLabel):
            assert isinstance(lb, str)
            lb = lb.lower()
            assert lb in _short_label2label
            lb = CorrectionLabel(label=_short_label2label[lb])
            assert lb in CORRECTION_CLS_LABELS  # sanity check one of the 4 correction labels

        # sanity check the selected demos don't have multi-occurring entities so make it easy demos
        assert len(patterns.find_match(text=d['sentence'], keyword=d['entity_span'])) == 1
        return cls(**d, label=lb)


CORRECTION_DIR_NAME = 'correction'


def load_triples_w_logprob(dataset_name: str = None, dir_name: str = None) -> List[Dict[str, Any]]:
    path = dir_name
    if not os.path.exists(path):
        path = dataset_name2data_dir(dataset_name=dataset_name, input_dir=dir_name).path
    assert os.path.exists(path)
    with open(os_join(path, 'logprobs-triple.json'), 'r') as f:
        return json.load(f)


TripleSample = Dict[str, Any]  # dictionary containing (sentence, entity span, entity type) triple and logprob


@dataclass
class UncertainTriplesOutput:
    entity_type2triples: Dict[str, List[TripleSample]] = None
    d_log: Dict[str, Any] = None


def log_n_select_uncertain_triples(
        triples: List[Dict[str, Any]] = None, logprob_thresh: float = -2e-2, top_n: Union[int, float] = 0.2,
        group_by_type: bool = True, entity_types: List[str] = None, shuffle: Union[bool, int] = False,
        logger: logging.Logger = None, verbose: bool = True, ec: prettier.EdgeCases = None, score_key: str = 'average_logprob'
) -> UncertainTriplesOutput:
    """
    Load (sentence, entity span, entity type) triples and select samples w/ low log prob

    :param triples: samples w/ logprob file
    :param logprob_thresh: logprob upperbound for selecting uncertain samples
    :param top_n: If an int, select top n samples; if a float, select top n% samples
    :param group_by_type: If true, sort and select top samples for each entity type separately
    :param entity_types: list of entity types
    :param logger: logger for logging selected samples
    :param ec: Log edge case for highlighting entity span in sentence
    :param shuffle: If true, the samples for each entity type are shuffled
        If an integer, the samples are shuffled with the given random seed
    :param verbose: If true, log the selected samples
    :param score_key: The key in the sample dictionary for the uncertainty score to rank the samples
        Lower is more uncertain
    :return dict of (entity type => uncertain samples)
    """
    # select uncertain triples capped by count & logprob, try
    #   1> rank solely by log prob
    #   2> first stratify by entity type, then rank by log prob
    logger = logger or _logger
    n_triple = len(triples)

    count_cap = None
    if top_n is not None:
        if isinstance(top_n, int):
            count_cap = top_n
        else:
            assert isinstance(top_n, float)
            count_cap = round(n_triple * top_n)  # 20% of the samples

    triples = sorted(triples, key=lambda x: x[score_key])
    et2n_ch = Counter([x['entity_type'] for x in triples])
    if set(et2n_ch.keys()) != set(entity_types):
        sic(et2n_ch.keys(), entity_types)
        set_1, set_2 = set(et2n_ch.keys()), set(entity_types)
        sic(set_1 & set_2, set_1 - set_2, set_2 - set_1)
    assert set(et2n_ch.keys()) == set(entity_types)
    n_entity = sum(et2n_ch.values())
    entity_counts = {et: (et2n_ch[et], to_percent(et2n_ch[et] / n_entity, decimal=1)) for et in entity_types}
    triples_ch = [x for x in triples if x[score_key] < logprob_thresh]
    n_ch = len(triples_ch)
    d_log_count = {
        '#total-triples': n_triple, 'entity-type-counts': entity_counts,
        'logprob-thresh': logprob_thresh, 'top-n': top_n, 'count-thresh': count_cap,
        'group-by-type': group_by_type, '#challenging-triples': n_ch
    }
    if not group_by_type:
        if n_ch > count_cap:
            triples_ch = triples_ch[:count_cap]
            d_log_count['#challenging-triples-kept'] = count_cap
        else:
            d_log_count['#challenging-triples-kept'] = n_ch
        et2triples_ch = defaultdict(list)
        for x in triples_ch:
            et2triples_ch[x['entity_type']].append(x)
        et2triples_ch = {et: et2triples_ch[et] for et in entity_types}  # order by the entity type list given
    else:
        et2triples_ch = defaultdict(list)
        for x in triples_ch:
            et2triples_ch[x['entity_type']].append(x)
        # TODO: this will influence the shuffling order for earlier LLM completions
        et2triples_ch = {et: et2triples_ch[et] for et in entity_types}
        n_ch = sum([len(x) for x in et2triples_ch.values()])
        if n_ch > count_cap:  # keep the class-wise distribution, drop the same ratio of samples from each class
            ratio = count_cap / n_ch
            triples_ch = []
            for et, triplets in et2triples_ch.items():
                n = round(len(triplets) * ratio)
                et2triples_ch[et] = triplets_ = triplets[:n]
                triples_ch.extend(triplets_)
            d_log_count['#challenging-triples-kept'] = count_cap
        else:
            d_log_count['#challenging-triples-kept'] = n_ch
    n_ch = sum([len(x) for x in et2triples_ch.values()])
    d_log_count['%challenging-triples'] = round(n_ch / n_triple * 100, 2)

    def format_single(x: Dict[str, Any]) -> Tuple:
        k = sample_w_logprob2pretty_str(x, ec=ec, score_key=score_key)
        v = f'{x[score_key]:.3e}'
        src = x.get('source')
        if src:
            v = pl.i(dict(lp=v, src=src))
        return k, v
    triples_ch_log = {et: [format_single(x) for x in triples] for et, triples in et2triples_ch.items()}
    triples_ch_log = {et: {k: v for (k, v) in triples} for et, triples in triples_ch_log.items()}

    et2n_ch = {et: len(et2triples_ch[et]) for et in entity_types}  # order in the expected order of entity types
    n_entity = sum(et2n_ch.values())
    # also get the percentage of challenging samples for each entity type
    et2n_ch = {et: (et2n_ch[et], to_percent(et2n_ch[et] / n_entity, decimal=1)) for et in entity_types}
    d_log_count['#challenging-triples-kept-by-type'] = et2n_ch
    msg = f'Challenging triples selected via log prob w/ {pl.i(d_log_count, indent=2)}'

    srcs = [x.get('source') for x in triples_ch]
    srcs = [x for x in srcs if x]
    if len(srcs) > 0:
        count_src = dict(Counter(srcs).most_common())
        msg = f'{msg} and source counts: {pl.i(count_src, indent=1)}'
    if verbose:
        msg = f'{msg} and triples: {pl.i(triples_ch_log, indent=2)}'
    logger.info(msg)

    if shuffle:
        use_seed = isinstance(shuffle, int) and not isinstance(shuffle, bool)
        if use_seed:
            seed = shuffle
            random.seed(seed)
            logger.info(f'Shuffling samples w/ seed {pl.i(seed)}')

        shuf_idxs = dict()
        for et, triples in et2triples_ch.items():
            shuf_idxs[et] = idxs = list(range(len(triples)))
            random.shuffle(idxs)
            et2triples_ch[et] = [triples[i] for i in idxs]
        if use_seed:
            random.seed()

        d_log = dict(shuffle=shuffle, sent_idxs=shuf_idxs)
        logger.info(f'Samples shuffled w/ {pl.i(d_log, indent=2)}')
    return UncertainTriplesOutput(entity_type2triples=et2triples_ch, d_log=d_log_count)


def sample_w_logprob2pretty_str(sample: Dict[str, Any], ec: prettier.EdgeCases = None, score_key: str = 'average_logprob') -> str:
    sent, enm, et, lp = sample['sentence'], sample['span'], sample['entity_type'], sample[score_key]
    sent = prettier.highlight_span_in_sentence(
        sentence=sent, span=enm, ec=ec, format_span=lambda x: f'[{pl.i(x, c="y")}]',
        span_index=sample.get('index'), span_index_super=sample.get('index_super'))
    return f'{pl.i(enm, c="y")} ({pl.i(et, c="r")}) <= {sent}'


@dataclass
class NerSampleMatchOutput:
    is_match: bool = None
    matched_entity_index: int = None


@dataclass(eq=True, frozen=True)
class CorrectionSample:
    sentence: str = None
    span: str = None
    entity_type: str = None
    correction_label: CorrectionLabel = None
    correction: str = None

    # for resolving the same span in the sentence
    span_index: int = None  # the index of the multi-occurring span in the sentence
    span_index_super: int = None  # index if any entity in the sentence is a super-string

    def match_ner_sample(self, sample: NerReadableExample = None):
        if self.sentence != sample.sentence:  # optimize
            return NerSampleMatchOutput(is_match=False)

        enms, ets = sample.entity_names, sample.entity_types
        idxs_match = [i for i, (enm, et) in enumerate(zip(enms, ets)) if enm == self.span and et == self.entity_type]
        if len(idxs_match) > 1:  # must be due to multi-occurring entity
            assert self.span_index is not None
            idxs_match = [idxs_match[self.span_index]]
        assert len(idxs_match) in [0, 1]  # sanity check not multiple matches
        mch = len(idxs_match) == 1
        return NerSampleMatchOutput(is_match=mch, matched_entity_index=idxs_match[0] if mch else None)

    def match_correction(self, correction: 'CorrectionSample' = None):
        # are corrections on the same (sentence, entity span, entity type) triple
        return self.sentence == correction.sentence and self.span == correction.span and self.entity_type == correction.entity_type


ENTITY_TYPE_OTHER = 'other'  # used in prompt for LLM to classify the entity type


def log_correction_samples(
        entity_type2corrections: Dict[str, List[CorrectionSample]], logger: logging.Logger = None, ec: prettier.EdgeCases = None):
    et2ret = entity_type2corrections
    logger = logger or _logger
    # pretty-log each correction sample
    for et, samples in et2ret.items():
        # group corrections by correction type
        tp2samples = defaultdict(list)
        for s in samples:
            tp2samples[s.correction_label].append(s)

        log = []
        for crt_type in CORRECTION_CLS_LABELS:
            samples_ = tp2samples[crt_type]
            log_ = []
            for sample in samples_:
                sent, enm, et = sample.sentence, sample.span, sample.entity_type
                idx, idx_sup = sample.span_index, sample.span_index_super
                sent = prettier.highlight_span_in_sentence(
                    sentence=sent, span=enm, ec=ec, format_span=lambda x: f'[{pl.i(x, c="y")}]', span_index=idx, span_index_super=idx_sup)
                enm, et = pl.i(enm, c="y"), pl.i(et, c="r")

                s, e = pl.i('[', c='b'), pl.i(']', c='b')
                if crt_type == LABEL_CORRECT:
                    pair = f'{enm} ({et})'
                elif crt_type == LABEL_WRONG_BOUNDARY:
                    enm_crt = pl.i(sample.correction, c="g")
                    pair = f'{s}{enm} -> {enm_crt}{e} ({et})'
                else:
                    assert crt_type in [LABEL_WRONG_TYPE, LABEL_NOT_NAMED_ENTITY]
                    if crt_type == LABEL_WRONG_TYPE:
                        et_crt = sample.correction
                    else:
                        et_crt = STR_NOT_NAMED_ENTITY
                    et_crt = pl.i(et_crt, c="g")
                    pair = f'{enm} ({s}{et} -> {et_crt}{e})'
                log_.append((pair, sent, enm))
            # sort by original entity name then by sentence for the same correction type
            log_ = sorted(log_, key=lambda x: (x[2].lower(), x[1].lower()))
            log += log_
        log_ = defaultdict(list)  # the same entity span correction may happen multiple times, so group the corresponding sentences
        for pair, sent, _ in log:
            log_[pair].append(sent)
        # if just 1 sentence, index into the element to save indentation space
        log = {pair: sent[0] if len(sent) == 1 else sent for pair, sent in log_.items()}
        logger.info(f'Correction samples for entity type {pl.i(et)}: {pl.i(log, indent=2, value_no_color=True)}')


def load_corrections(path: str = None) -> Dict[str, List[CorrectionSample]]:
    with open(path) as f:
        corrections = json.load(f)['corrections']  # entity type -> list of corrections
    ret = defaultdict(list)
    for et, mcs in corrections.items():
        for mc in mcs:
            mc['correction_label'] = CorrectionLabel(**mc['correction_label'])
            ret[et].append(CorrectionSample(**mc))
    return ret


def override_w_manual_corrections(
        entity_type2corrections: Dict[str, List[CorrectionSample]], dataset_name: str = None, manual_edit_dir_name: str = None,
        logger: logging.Logger = None
) -> Dict[str, List[CorrectionSample]]:
    # override the processed corrections w/ the manual corrections
    import pandas as pd

    assert isinstance(manual_edit_dir_name, str)
    manual_path = dataset_name2data_dir(dataset_name=dataset_name, sub_dir=CORRECTION_DIR_NAME, input_dir=manual_edit_dir_name).path
    manual_corrections = load_corrections(path=os_join(manual_path, 'corrections_manual-edit.json'))
    logger = logger or _logger
    logger.info(f'Overriding LLM-generated corrections w/ manual corrections from {pl.i(manual_path)}')

    # compare w/ LLM self-corrections and log the LLM correction accuracy
    et2ret = entity_type2corrections
    entity_types = sconfig(f'datasets.{dataset_name}.readable-entity-types')
    et2lb2n_match = defaultdict(lambda: defaultdict(int))
    et2report = dict()
    for et, llm_crts in et2ret.items():
        manual_crts = manual_corrections[et]

        # sanity check manual corrections are well-formed
        for mc in manual_crts:
            # lb = mc.correction_label
            # if lb in [LABEL_CORRECT, LABEL_NOT_NAMED_ENTITY]:
            #     if mc.correction is not None:
            #         sic(mc)
            #     assert mc.correction is None
            # else:  # wrong type or wrong boundary
            #     assert mc.correction is not None
            #     if lb == LABEL_WRONG_TYPE:
            #         if mc.correction not in self.entity_types:
            #             sic(mc)
            #         assert mc.correction in self.entity_types
            #     else:
            #         assert lb == LABEL_WRONG_BOUNDARY
            #         assert have_word_overlap(span1=mc.span, span2=mc.correction)
            # above too complicated, ignore
            if mc.correction_label == LABEL_WRONG_BOUNDARY:
                if not check.have_word_overlap(span1=mc.span, span2=mc.correction):
                    sic(mc)
                assert check.have_word_overlap(span1=mc.span, span2=mc.correction)

        assert len(llm_crts) == len(manual_crts)  # sanity check the same #samples
        # first, get one-to-one map between LLM and manual corrections
        llm_idx2manual_idx = []
        for llm_crt in llm_crts:
            manual_idxs = [idx for idx, manual_crt in enumerate(manual_crts) if llm_crt.match_correction(correction=manual_crt)]
            assert len(manual_idxs) == 1  # sanity check one-to-one match
            llm_idx2manual_idx.append(manual_idxs[0])
        assert len(set(llm_idx2manual_idx)) == len(llm_crts)  # sanity check unique
        manual_crts = [manual_crts[i] for i in llm_idx2manual_idx]
        # sanity check the same position are on the same triple
        assert all(llm_crt.match_correction(correction=manual_crt) for llm_crt, manual_crt in zip(llm_crts, manual_crts))

        # check the accuracy of the LLM corrections, get #valid LLM corrections for each correction type
        lb2n_match = et2lb2n_match[et]
        for llm_crt, manual_crt in zip(llm_crts, manual_crts):
            lb_manual = manual_crt.correction_label
            if lb_manual == LABEL_CORRECT:
                lb_llm = llm_crt.correction_label
                if lb_llm == LABEL_CORRECT:
                    lb2n_match[lb_manual] += 1
            elif lb_manual == LABEL_WRONG_BOUNDARY:
                lb_llm = llm_crt.correction_label
                if lb_llm == LABEL_WRONG_BOUNDARY and llm_crt.correction == manual_crt.correction:
                    lb2n_match[lb_manual] += 1
            else:
                # 2 cases:
                #   1> correction to a relevant type to the dataset;
                #   2> correction to a irrelevant type, (e.g. Other) treat as not a named entity
                et_manual = None
                if lb_manual == LABEL_WRONG_TYPE and manual_crt.correction in entity_types:
                    lb_manual = LABEL_WRONG_TYPE
                    et_manual = manual_crt.correction
                else:
                    lb_manual = LABEL_NOT_NAMED_ENTITY

                if lb_manual == LABEL_WRONG_TYPE:
                    lb_llm = llm_crt.correction_label
                    if lb_llm == LABEL_WRONG_TYPE and llm_crt.correction == et_manual:  # LLM must also correct to the same type
                        lb2n_match[lb_manual] += 1
                else:
                    # if LLM corrects to an irrelevant type, also count as correct
                    lb_llm = llm_crt.correction_label
                    if lb_llm == LABEL_NOT_NAMED_ENTITY or (lb_llm == LABEL_WRONG_TYPE and llm_crt.correction not in entity_types):
                        lb2n_match[lb_manual] += 1
        et2lb2n_match[et] = lb2n_match

        # get precision, recall & f1 for each correction type
        lb2report = dict()
        for lb in CORRECTION_CLS_LABELS:
            n_match = lb2n_match[lb]
            n_manual = sum(x.correction_label == lb for x in manual_crts)
            lb2report[lb.label] = get_classification_report_row(true_positive=n_match, n_pred=n_match, n_true=n_manual, as_percent=True)
        n_match, n_crts = sum(lb2n_match.values()), len(llm_crts)
        et2report[et] = {'#correct': n_match, 'support': n_crts}
        df = pd.DataFrame(lb2report).T
        df['support'] = df.support.astype(int)
        logger.info(f'Correction accuracy for entity type {pl.i(et)}:\n{pl.i(df)}')

    # since same #prediction & #true, just get accuracy, for each entity type, across all correction types
    n_match, n_crts = (sum(et2report[et][k] for et in et2report) for k in ['#correct', 'support'])
    et2report['total'] = {'#correct': n_match, 'support': n_crts}
    # also for each correction type, across all entity types
    for lb in CORRECTION_CLS_LABELS:
        n_match = sum(et2lb2n_match[et][lb] for et in et2lb2n_match)
        n_crts = sum(sum(crt.correction_label == lb for crt in crts) for crts in manual_corrections.values())
        et2report[lb.label] = {'#correct': n_match, 'support': n_crts}
    # also check for all `incorrect` corrections
    n_match = sum(et2lb2n_match[et][lb] for et in et2report for lb in CORRECTION_CLS_LABELS if lb != LABEL_CORRECT)
    n_crts = sum(sum(crt.correction_label != LABEL_CORRECT for crt in crts) for crts in manual_corrections.values())
    et2report['incorrect'] = {'#correct': n_match, 'support': n_crts}
    df = pd.DataFrame(et2report).T
    df['accuracy'] = (df['#correct'] / df['support']).map(to_percent)
    df['support'] = df.support.astype(int)
    df = df[['accuracy', '#correct', 'support']]
    logger.info(f'Overall LLM correction accuracy:\n{pl.i(df)}')
    # raise NotImplementedError

    return manual_corrections  # actually override the LLM corrections


def log_n_save_corrections(
        entity_type2corrections: Dict[str, List[CorrectionSample]], output_path: str = None,
        logger: logging.Logger = None, d_log_count: Dict[str, Any] = None, timer: Union[str, Timer] = None, **write_kwargs
):
    # write all corrections to file, intended for manual correction
    et2ret = entity_type2corrections
    et2ret_write = {  # sort by the 4 correction types, then entity name, then sentence
        et: sorted(samples, key=lambda x: (x.correction_label, x.span.lower(), x.sentence.lower())) for et, samples in et2ret.items()}
    et2ret_write = {et: [asdict(x) for x in samples] for et, samples in et2ret_write.items()}  # convert to dict

    d_write = {**write_kwargs, 'corrections': et2ret_write}
    with open(os_join(output_path, 'corrections.json'), 'w') as f:
        json.dump(d_write, f, indent=4)

    n_extract = {et: len(crts) for et, crts in et2ret.items()}
    n_extract['total'] = sum(n_extract.values())
    d_log_count['#correction-extracted'] = n_extract

    n_extract_type = dict()
    for et, crts in et2ret.items():
        c = Counter([x.correction_label.label for x in crts])
        c = {ct.label: c[ct.label] for ct in CORRECTION_CLS_LABELS}  # modify the iteration order for logging
        # also add a count for total #samples that LLM corrected
        c['incorrect'] = sum(c_ for x, c_ in c.items() if x != LABEL_CORRECT.label)
        n_extract_type[et] = c
    n_extract_type_total = Counter()
    for et, n in n_extract_type.items():
        n_extract_type_total.update(n)
    n_extract_type['total'] = n_extract_type_total

    # also show ratio of correction types; note the sum of values for each key don't add to 1 cos there's the `incorrect` field
    n_extract_type_ratio = {k: {k_: to_percent(v_ / n_extract[k], decimal=1) for k_, v_ in d.items()} for k, d in n_extract_type.items()}
    d_log_count_by_type = {'#': n_extract_type, '%': n_extract_type_ratio}

    logger = (logger or _logger)
    if isinstance(timer, Timer):
        timer = timer.end()
    logger.info(f'Processed correction completions in {pl.i(timer)} w/ {pl.i(d_log_count, indent=1, align_keys=True)} '
                f'and {pl.i(d_log_count_by_type, indent=2, align_keys=2)}')


@dataclass
class CorrectedNerSampleOutput:
    entity_names: List[str] = None
    entity_types: List[str] = None
    correction_entity_indices: List[int] = None  # the index of entity that the correction is applied to
    correction_maps: Dict[CorrectionLabel, List[Tuple[str, str]]] = None  # correction kind => (original, corrected) pairs


def ner_sample2corrected_sample(
        sample: NerReadableExample = None, corrections: List[CorrectionSample] = None, allowed_entity_types: List[str] = None
) -> CorrectedNerSampleOutput:
    ner_sample, crts = sample, corrections
    enms, ets = list(ner_sample.entity_names).copy(), list(ner_sample.entity_types).copy()

    mch_e_idxs = []
    idxs_entity_drop = []
    for crt in crts:
        mch = crt.match_ner_sample(sample=ner_sample)
        assert mch.is_match  # sanity check
        mch_e_idx = mch.matched_entity_index
        mch_e_idxs.append(mch_e_idx)

        correction_type = crt.correction_label
        if correction_type == LABEL_WRONG_BOUNDARY:
            enms[mch_e_idx] = crt.correction
        elif correction_type == LABEL_WRONG_TYPE:
            is_relevant_type = crt.correction in allowed_entity_types
            if is_relevant_type:
                ets[mch_e_idx] = crt.correction
            else:
                idxs_entity_drop.append(mch_e_idx)  # need to drop them outside this loop to avoid match index shift
        else:
            assert correction_type == LABEL_NOT_NAMED_ENTITY
            idxs_entity_drop.append(mch_e_idx)
    if len(idxs_entity_drop) > 0:
        # drop the entities in reverse order to avoid index shift
        for i in sorted(idxs_entity_drop, reverse=True):
            enms.pop(i)
            ets.pop(i)
    crt_map = defaultdict(list)
    for crt in corrections:
        if crt.correction_label == LABEL_WRONG_BOUNDARY:
            crt_map[LABEL_WRONG_BOUNDARY].append((crt.span, crt.correction))
        elif crt.correction_label == LABEL_WRONG_TYPE:
            crt_map[LABEL_WRONG_TYPE].append((crt.entity_type, crt.correction))
        else:
            assert crt.correction_label == LABEL_NOT_NAMED_ENTITY
            crt_map[LABEL_NOT_NAMED_ENTITY].append((crt.span, STR_NOT_NAMED_ENTITY))
    return CorrectedNerSampleOutput(entity_names=enms, entity_types=ets, correction_entity_indices=mch_e_idxs, correction_maps=crt_map)


def get_classification_report_row(true_positive: int = None, n_pred: int = None, n_true: int = None, as_percent: bool = False) -> Dict[str, Any]:
    p = true_positive / n_pred if n_pred > 0 else 0
    r = true_positive / n_true if n_true > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0

    if as_percent:
        p, r, f1 = to_percent(p), to_percent(r), to_percent(f1)
    return dict(precision=p, recall=r, f1=f1, support=n_true)
