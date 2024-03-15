import re
import ast
import json
import logging
from os.path import join as os_join
from copy import deepcopy
from typing import Dict, Tuple, List, Union, Any
from dataclasses import dataclass
from collections import defaultdict

import numpy as np

from stefutil import get_logger, pl, ca, stem
from src.util import spans_overlap, api, patterns
from src.util.ner_example import NerReadableExample
from src.util.sample_formats import EntityPairTemplate, EntityPairEncloseType
from src.data_util import sample_edit as edit
from src.data_util.prettier import EdgeCases, sdpc


__all__ = [
    'TripleWLogProb',
    'Span2LogProb', 'LogProbOutput', 'Sample2LogProbsOutput', 'ner_sample2logprobs',
    'log_n_save_samples_w_logprob', 'log_n_save_triples_w_logprob'
]


_logger = get_logger('LogProb Util')


TripleWLogProb = Tuple[str, str, str, float]


@dataclass
class LogProbOutput:
    logprobs: List[Dict[str, Any]] = None  # logprobs for each token in the span
    avg_logprob: float = None  # average logprob normalized by #tokens
    min_logprob: float = None  # minimum logprob in greedy decoding, the smaller, the more uncertain
    span: str = None

    def top_n_avg_logprob(self, n: int = 3) -> float:
        """
        Average logprob of top n of the smallest tokens in the span
        """
        assert isinstance(n, int) and n > 0
        lps = sorted([lp['logprob'] for lp in self.logprobs])[:n]
        return np.mean(lps)


def completion_n_logprobs_match(completion: str = None, logprobs: List[Dict[str, Any]] = None) -> bool:
    # handle an artificial newline added to the end
    return api.tc35(text=completion) == len(logprobs) or \
        completion[-1] == '\n' and api.tc35(text=completion[:-1]) == len(logprobs)


def logprob_pair_decode_utf8(prob1: Dict[str, Any], prob2: Dict[str, Any]) -> str:
    """
    For weird edge case, LLM tokenization splits a single, weird UTF-8 character into 2 tokens, e.g.
        `ė` => `['\xc4', '\x97']`

    This function consumes 2 adjacent logprob objects and get the corresponding decoded string
    """
    prob, prob_next = prob1, prob2
    tok = prob['token']
    n_tok = len(tok)
    assert (n_tok, len(prob['bytes'])) in [(4, 1), (5, 2)]

    # sic(prob, prob_next)
    tok_next = prob_next['token']
    assert len(tok_next) == 4 and len(prob_next['bytes']) == 1
    if len(tok) == 4:
        assert tok.startswith('\\x')
        n_tok_pair = 1
    else:
        assert tok[0] == ' ' and tok[1:].startswith('\\x')
        n_tok_pair = 2
    assert tok_next.startswith('\\x')
    # these 2 bytes can be merged into a single character
    # first, get the decoded string form both of the 2 tokens
    str_dec = ast.literal_eval(f'b"{tok}{tok_next}"').decode('utf-8')
    assert len(str_dec) == n_tok_pair
    return str_dec


class Span2LogProb:
    """
    Given logprobs for a completion, extract logprobs for a span
    """
    sample_types = [
        '3-stage-entity-type',  # 3rd step in 3-stage gen: the label part
        '1-stage-sample',  # 1-stage gen: the entire sample
        '1-stage-annotations',  # 1-stage gen: the entity annotations part
        '1-stage-annotation',  # 1-stage gen: a single entity annotation
        'correction-label'  # LLM self-correction: the label part for a single correction
    ]

    def __init__(
            self, completion: str = None, logprobs: List[Dict[str, Any]] = None, sample_type: str = None,
            cot: bool = False, logger: logging.Logger = None, ec: EdgeCases = None
    ):
        # may not be the case sometimes, when LLM don't generate optimal tokenization, e.g.
        #   [`g`, `overment`] instead of [`government`]
        # if not completion_n_logprobs_match(completion=completion, logprobs=logprobs):
        #     tc35 = api_util.tc35
        #     sic(completion)
        #     sic(tc35(text=completion), len(logprobs))
        #     sic(tc35(text=completion[:-1]))
        # assert completion_n_logprobs_match(completion=completion, logprobs=logprobs)
        ca.assert_options(display_name='Sample Type for Span Extraction', val=sample_type, options=Span2LogProb.sample_types)
        self.completion = completion
        self.logprobs = logprobs
        self.sample_type = sample_type

        # build a map from completion string index to token-wise logprobs index
        self.str_idx2logprob_idx = dict()
        i_cpl = 0
        it = enumerate(logprobs)
        curr = next(it)
        while curr is not None:
            i_prob, prob = curr
            self.str_idx2logprob_idx[i_cpl] = i_prob
            # if len(prob['token']) != len(prob['bytes']):
            #     sic(prob['token'], prob['bytes'])
            # assert len(prob['token']) == len(prob['bytes'])
            assert len(prob['token']) > 0
            # if completion[i_cpl:i_cpl + len(prob['token'])] != prob['token']:
            #     sic(i_cpl, prob['token'], completion[i_cpl:i_cpl + len(prob['token'])])
            #     sic(len(prob['token']), len(prob['bytes']))
            n_tok = len(prob['token'])
            n_tok_cpl = len(completion[i_cpl:i_cpl + n_tok])
            # TODO: too complicated encoding stuff, just drop that completion
            tok = prob['token']
            n_tok = len(tok)
            span_in_cpl = completion[i_cpl:i_cpl + n_tok]
            # sic(tok, span_in_cpl)
            tok_exp = tok
            if span_in_cpl != tok:
                # a rare edge case where converting token into human-readable string is not intuitive, e.g. `\xc4\x97` => `ė`
                #   in such case, to ensure 1-to-1 map between completion range and logprobs, need special handling
                #   Seems to be explained here:
                #       https://community.openai.com/t/tokens-are-mangled-for-some-non-english-characters-resolved/74315/5?u=stefan.hg
                # sanity check is due to weird UTF-8 encoding
                _, prob_next = next(it, None)
                assert prob_next is not None
                str_dec = logprob_pair_decode_utf8(prob1=prob, prob2=prob_next)
                n_tok_pair = len(str_dec)

                # sanity check the decoded string is the same as the span in the completion
                assert completion[i_cpl:i_cpl + n_tok_pair] == str_dec
                tok_exp = str_dec
                n_tok = n_tok_pair

            assert completion[i_cpl:i_cpl + n_tok] == tok_exp
            i_cpl += n_tok

            curr = next(it, None)
        self.str_idx2logprob_idx[i_cpl] = len(logprobs)  # for the last token, note exclusive end
        # assert i_cpl + 1 == len(completion)  # sanity check coverage of completion by logprobs; note exclusive end

        self.cot = cot

        self.logger = logger or _logger
        self.ec = ec

    def __call__(self, span: str = None, index_start: int = None, index_end: int = None, sample_type: str = None) -> LogProbOutput:
        """
        :param span: A span in the completion to extract logprobs for
        :param index_start: If given, search for the span starting from this index
        :param index_end: If given, search for the span ending at this index
        """
        from stefutil import sic

        assert span is not None
        if index_start is not None and index_end is not None:
            assert index_start < index_end
        if index_start is None:
            index_start = 0
        if index_end is None:
            index_end = len(self.completion)
        cpl = self.completion[index_start:index_end]
        ms = patterns.find_match(text=cpl, keyword=span)

        if len(ms) != 1:
            sic(span, cpl, ms, len(ms), cpl == span)
        assert len(ms) == 1  # sanity check span only appears in the completion once
        m = ms[0]

        idx_cpl_start, idx_cpl_end = m.start() + index_start, m.end() + index_start
        # sic('initial idxs', idx_cpl_start, idx_cpl_end, span)

        sample_type = sample_type or self.sample_type
        # tok1_strt = None  # for sanity check modified token match special case
        if idx_cpl_start not in self.str_idx2logprob_idx:
            if sample_type == '1-stage-annotations':
                assert span.startswith('Named Entities')
                assert idx_cpl_start - 1 in self.str_idx2logprob_idx  # LLM may group a starting space to ` Named`
                idx_cpl_start -= 1
                span = f' {span}'
                tok = f' Named'
            else:
                assert sample_type in ['1-stage-annotation']  # LLM groups a starting space to the annotation

                if idx_cpl_start - 1 in self.str_idx2logprob_idx:
                    idx_cpl_start -= 1

                    lp_idx = self.str_idx2logprob_idx[idx_cpl_start]
                    tok = self.logprobs[lp_idx]['token']  # can't infer from just span, cos depends on LLM tokenization
                    c1 = tok[0]
                    assert tok[0] in (' ', '\t')  # sanity check

                    span = f'{c1}{span}'
                    if not span.startswith(tok):
                        sic(tok, span)
                    assert span.startswith(tok)
                else:
                    c1 = span[0]
                    if c1 not in ["'", '"', '(', '[', '$']:
                        str_i2lp_i = {k: v for k, v in self.str_idx2logprob_idx.items() if idx_cpl_start - 10 < k < idx_cpl_start + 10}
                        sic(span, idx_cpl_start, idx_cpl_end, self.completion[idx_cpl_start:idx_cpl_end], str_i2lp_i)
                    assert c1 in ["'", '"', '(', '[', '$']
                    # edge case: annotated span starts with a punctuation, due to e.g. is enclosed in quotes/brackets, e.g.
                    #   `'I Will Always Love You' (Song)`, and the initial token may become ` ['`
                    #   `(I Can’t Get No) Satisfaction (Song)`
                    #   `[Ristorante Roma] (Restaurant Name)`
                    #   `$15 (Price)`
                    #   ` "My Heart Will Go On" (Song)`, and the initial token may become ` ["`
                    assert idx_cpl_start - 2 in self.str_idx2logprob_idx
                    idx_cpl_start -= 2
                    tok = f" [{c1}"
                    span = f' [{span}'
                # tok1_strt = ' '
                # sic(tok)
                # sic(span, idx_cpl_start, idx_cpl_end, self.completion[idx_cpl_start:idx_cpl_end], self.str_idx2logprob_idx)
                # raise NotImplementedError
            # should not happen for other sample types
            d_log = dict(idx_cpl_start=idx_cpl_start, idx_cpl_end=idx_cpl_end, span=span, initial_token=re.escape(tok))
            msg = f'Edge case: LLM output initial token adjusted w/ {pl.i(d_log)}'
            self.ec(msg=msg, kind='logprob-idx-adjust', args=d_log, disable_std_log=True)

        # sic('before mod', span, idx_cpl_end)
        if idx_cpl_end not in self.str_idx2logprob_idx:
            if sample_type == '3-stage-entity-type':
                # for this function is intended for indexing span CLS of the form `Labels: ...`, see `generate_type.py`
                if self.cot:  # sanity check what label looks like
                    assert span[-2:] == '.\n'
                else:
                    assert span.endswith('entity\n') or span.endswith('of other type\n')
                if idx_cpl_end < len(self.completion) and self.completion[idx_cpl_end] == '\n':  # note end is exclusive
                    # LLM may group `.\n\n` as one token, for a valid indexing into logprobs, we need to adjust for this
                    if self.cot:
                        tok = '.\n\n'
                    else:  # no CoT generated
                        tok = '\n\n'
                    idx_cpl_end += 1
                    span += '\n'
                else:
                    assert idx_cpl_end == len(self.completion)  # sanity check
                    # added newline is artificial, completion didn't end with a newline
                    if self.cot:
                        tok = '.'
                    else:
                        tok = ' entity' if span.endswith('entity\n') else ' type'
                    idx_cpl_end -= 1
                    span = span[:-1]
            elif sample_type in ['1-stage-sample', '1-stage-annotations']:
                if span.endswith(')]\n'):
                    tok = ')]'
                else:
                    # due to wrong sample formatting, e.g.
                    #   `Named Entities: [James Bond (Character), new (Year), Trailer]\n`
                    #   `Named Entities: [romantic comedy (Genre), 'Can't Help Falling in Love' (Song) ]`
                    #   `Named Entities: [top-rated (Viewers' Rating), comedy (Genre), 2020 (Year)] \n`
                    #   `Named Entities: [BBC (organization), Asia-Pacific (location)].\n`
                    #   `Named Entities: [The Godfather (Title), directed (Director)] \n\t\n`
                    #   `Named Entities: [sci-fi (Genre), 2010s (Year), top-rated (Review)\n`
                    #   `Named Entities: [fancy (Rating), breakfast (Amenity), `
                    #   `Named Entities: [sushi (Cuisine), best (Rating), around here (Location) `
                    #   `Named Entities: None\n`
                    if span.endswith(' ]\n'):
                        tok = ' ]'
                    elif span.endswith('] \n'):
                        tok = ' '
                    elif span.endswith(']\t\n'):
                        tok = '\t'
                    elif span.endswith(']\n'):
                        tok = ']'
                    elif span.endswith(')\n'):
                        tok = ')'
                    elif span.endswith(')].\n'):
                        tok = ')].'
                    elif span.endswith(', \n'):
                        # s_idx2lp_idx = {k: v for k, v in self.str_idx2logprob_idx.items() if idx_cpl_end - 10 < k < idx_cpl_end + 10}
                        # sic(span, idx_cpl_start, idx_cpl_end, s_idx2lp_idx)
                        # sic(self.logprobs[self.str_idx2logprob_idx[idx_cpl_end + 1] - 1]['token'])
                        tok = ' '
                    elif span.endswith(') \n'):
                        tok = ' '
                    else:
                        if not span.endswith('None\n'):
                            s_idx2lp_idx = {k: v for k, v in self.str_idx2logprob_idx.items() if idx_cpl_end - 10 < k < idx_cpl_end + 10}
                            sic(span, idx_cpl_start, idx_cpl_end, s_idx2lp_idx)
                        assert span.endswith('None\n')
                        # if idx_cpl_end + 1 in self.str_idx2logprob_idx:
                        #     idx = self.str_idx2logprob_idx[idx_cpl_end + 1] - 1
                        # else:
                        #     idx = self.str_idx2logprob_idx[idx_cpl_end - 1] - 1
                        # sic(self.logprobs[idx]['token'])
                        # sic(self.logprobs[idx-1]['token'])
                        # raise NotImplementedError
                        tok = None  # see below
                # handle a rare case first, possibly due to LLM temperature
                if (span.endswith('] \n')) and idx_cpl_end + 1 not in self.str_idx2logprob_idx:
                    # s_idx2lp_idx = {k: v for k, v in self.str_idx2logprob_idx.items() if idx_cpl_end - 10 < k < idx_cpl_end + 10}
                    # sic(span, idx_cpl_start, idx_cpl_end, s_idx2lp_idx)
                    assert idx_cpl_end - 2 in self.str_idx2logprob_idx
                    # idx = self.str_idx2logprob_idx[idx_cpl_end - 2]
                    # sic(self.logprobs[idx]['token'])
                    # sic(self.logprobs[idx+1]['token'])
                    assert self.logprobs[self.str_idx2logprob_idx[idx_cpl_end - 2]]['token'] in [' \n\t\n', ' ']  # sanity check is rare case
                    tok = ')]'
                    idx_cpl_end -= 2
                    span = span[:-2]  # drop the trailing ` \n`
                elif span.endswith('None\n'):
                    # like a special case of the edge case below: the last 2 tokens are ` None`, and `\n\n`
                    if idx_cpl_end + 1 in self.str_idx2logprob_idx:  # due to LLM grouping newline as one token
                        tok = '\n\n'
                        idx_cpl_end += 1
                        span += '\n'
                    else:  # due to artificial newline added to completion
                        assert idx_cpl_end - 1 in self.str_idx2logprob_idx
                        tok = ' None'
                        idx_cpl_end -= 1
                        span = span[:-1]
                elif idx_cpl_end < len(self.completion) and self.completion[idx_cpl_end] == '\n':
                    # for edge case, final token is `\n\n\n` for an additional newline between NER samples...
                    if idx_cpl_end+1 not in self.str_idx2logprob_idx:
                        # if idx_cpl_end+1 not in self.str_idx2logprob_idx:
                        #     sic(tok, idx_cpl_end, self.str_idx2logprob_idx, span)
                        #     idx = self.str_idx2logprob_idx[idx_cpl_end+2]
                        #     sic(self.logprobs[idx-1]['token'])
                        assert idx_cpl_end+2 in self.str_idx2logprob_idx
                        tok_llm = self.logprobs[self.str_idx2logprob_idx[idx_cpl_end+2] - 1]['token']
                        assert tok_llm == '\n\n\n'
                        # so drop the trailing newline from the span
                        assert idx_cpl_end - 1 in self.str_idx2logprob_idx
                        tok = ')]'
                        idx_cpl_end -= 1
                        span = span[:-1]
                    else:
                        tok = f'{tok}\n\n'  # LLM grouped additional newline as one token

                        # another edge case: the final token is ` []\n\n`, instead of `]\n\n`
                        tok_llm = self.logprobs[self.str_idx2logprob_idx[idx_cpl_end+1] - 1]['token']
                        if tok_llm != tok:
                            # sic(tok_llm, tok)
                            assert (tok, tok_llm) == (']\n\n', ' []\n\n')
                            tok = tok_llm  # override to LLM token
                            # raise NotImplementedError
                        idx_cpl_end += 1
                        span += '\n'
                else:
                    # another edge case: the final token is ` []`, instead of `]`
                    # str_i2lp_i = {k: v for k, v in self.str_idx2logprob_idx.items() if idx_cpl_end - 10 < k < idx_cpl_end + 10}
                    # sic(idx_cpl_end, span, str_i2lp_i, tok, span)
                    tok_llm = self.logprobs[self.str_idx2logprob_idx[idx_cpl_end-1] - 1]['token']
                    if tok_llm != tok:
                        # sic(tok, tok_llm)
                        assert (tok, tok_llm) == (']', ' []')
                        tok = tok_llm  # override to LLM token
                    tok = tok  # due to artificial newline added to completion
                    idx_cpl_end -= 1
                    span = span[:-1]
                    # raise NotImplementedError
            else:
                assert sample_type == '1-stage-annotation'
                assert span.endswith(')')  # e.g. `4.8 stars (Viewers' Rating)`
                # sic(span, idx_cpl_start, idx_cpl_end, self.completion[idx_cpl_start:idx_cpl_end], self.str_idx2logprob_idx)

                # LLM grouped last char `)` w/ later chars in the span as one token, e.g.
                #   `,` for the next annotation
                #   `]`, `]\n`, `]\n\n` as end of sample annotations
                # sic(idx_cpl_end, idx_cpl_end + 1 in self.str_idx2logprob_idx)
                if idx_cpl_end + 1 in self.str_idx2logprob_idx:
                    # tok = '),'
                    idx_cpl_end += 1
                    idx_lp = self.str_idx2logprob_idx[idx_cpl_end] - 1  # note exclusive end
                    tok = self.logprobs[idx_lp]['token']
                    assert tok in ['),', ')]']  # either for next entity annotation, or end of entire completion
                    # if tok != '),':
                    #     sic(tok)
                    #     raise NotImplementedError
                    span += tok[1:]
                else:
                    # first add a `]`, the final token definitely starts from here
                    tok = ')]'  # keep adding a char until we find a match in logprobs
                    idx_cpl_end += 1
                    added = ']'
                    assert idx_cpl_end not in self.str_idx2logprob_idx  # sanity check need to add at least 1 newline char

                    for i, char in zip(range(1, 3), ['\n', '\n']):
                        idx_cpl_end += 1
                        added = f'{added}{char}'
                        tok = f'){added}'
                        # sic(i, added, tok)
                        if idx_cpl_end in self.str_idx2logprob_idx:
                            break
                    assert idx_cpl_end in self.str_idx2logprob_idx

                    # handle an edge case as above: final tokens is `)].`
                    tok_llm = self.logprobs[self.str_idx2logprob_idx[idx_cpl_end] - 1]['token']
                    # sic(tok_llm)
                    if tok_llm == ')].':
                        assert tok == ')]\n'
                        tok = tok_llm
                        span += '].'
                    elif tok_llm == ')\n\n':
                        # edge case: entity span didn't end w/ `]`, e.g.
                        #   `Named Entities: [budget-friendly (Price), diner (Amenity), all-day breakfast (Hours)`
                        assert tok == ')]\n'
                        tok = tok_llm
                        span += '\n\n'
                    else:
                        span += added
                # raise NotImplementedError

            # sanity check last token match
            if idx_cpl_end not in self.str_idx2logprob_idx:
                sic(span, idx_cpl_end, self.str_idx2logprob_idx)
            idx = self.str_idx2logprob_idx[idx_cpl_end] - 1  # note exclusive end
            if tok != self.logprobs[idx]['token']:
                s_idx2lp_idx = {k: v for k, v in self.str_idx2logprob_idx.items() if idx_cpl_end - 10 < k < idx_cpl_end + 10}
                sic(s_idx2lp_idx, idx_cpl_end, span, tok, self.logprobs[idx]['token'])
            assert tok == self.logprobs[idx]['token']

            d_log = dict(idx_cpl_start=idx_cpl_start, idx_cpl_end=idx_cpl_end, span=span, final_token=re.escape(tok))
            msg = f'Edge case: LLM output final token adjusted w/ {pl.i(d_log)}'
            # this consumes too much token space, log to file only
            self.ec(msg=msg, kind='logprob-idx-adjust', args=d_log, disable_std_log=True)

        if idx_cpl_start not in self.str_idx2logprob_idx or idx_cpl_end not in self.str_idx2logprob_idx:
            sic(self.str_idx2logprob_idx, idx_cpl_start, idx_cpl_end, span)  # sanity check
            sic(idx_cpl_start in self.str_idx2logprob_idx)
            sic(idx_cpl_end in self.str_idx2logprob_idx)
        assert idx_cpl_start in self.str_idx2logprob_idx and idx_cpl_end in self.str_idx2logprob_idx

        idx_prob_start, idx_prob_end = self.str_idx2logprob_idx[idx_cpl_start], self.str_idx2logprob_idx[idx_cpl_end]
        logprobs = self.logprobs[idx_prob_start:idx_prob_end]

        ret = [dict(token=lp['token'], logprob=lp['logprob']) for lp in logprobs]  # drop irrelevant fields
        lps = [lp['logprob'] for lp in logprobs]
        toks = [lp['token'] for lp in logprobs]
        joined = ''.join(toks)

        if joined != span:
            # given the UTF-8 encoding in tokens, need to decode it as in `__init__`
            # s = ''.join(lp['token'] for lp in logprobs)
            # check that must be due to weird UTF-8 encoding
            assert any('\\x' in tok for tok in toks)
            # str_dec = ast.literal_eval(f'b"{s}"').decode('utf-8')
            # decoding the whole thing seems to result in `EOF` errors, so decode each UTF-8 token pair if found
            joined = ''
            it = iter(logprobs)
            prob = next(it)
            while prob is not None:
                if '\\x' in prob['token']:
                    prob_next = next(it, None)
                    assert prob_next is not None and '\\x' in prob_next['token']  # sanity check must be a pair
                    str_dec = logprob_pair_decode_utf8(prob1=prob, prob2=prob_next)
                    joined += str_dec
                else:
                    joined += prob['token']
                prob = next(it, None)
        if joined != span:
            sic(idx_cpl_start, idx_cpl_end, span)
            sic(idx_prob_start, idx_prob_end)
            sic(joined, span)
        assert joined == span  # sanity check
        # average logprob normalized by #tokens
        return LogProbOutput(logprobs=ret, avg_logprob=sum(lps) / len(lps), min_logprob=min(lps), span=span)


@dataclass
class Sample2LogProbsOutput:
    logprob: float = None
    entity_logprobs: List[float] = None  # list of logprobs maps to each NER entity annotation by list position


def ner_sample2logprobs(
        sample: NerReadableExample = None, sample_str: str = None, sample_span: Tuple[int, int] = None, span2logprob: Span2LogProb = None,
        entities_str: str = None, pattern_entity: patterns.Patterns = None,
        entity_sep: str = None, entity_pair_map: EntityPairTemplate = None
) -> Sample2LogProbsOutput:
    """
    Extract sample-wise & entity-wise logprobs from a processed, i.e. valid NER sample

    :param sample: A processed NER sample
    :param sample_str: The entire sample string generated by LLM
    :param sample_span: The span (start, end indices) of the current sample in the LLM completion
    :param span2logprob: A Span2LogProb instance for extracting logprobs for a span
    :param entities_str: The entity annotations part of the sample string generated by LLM
    :param pattern_entity: regex pattern for extracting the entity annotation part
    :param entity_sep: expected separator between entity annotations
    :param entity_pair_map: A map between internal, structured (entity name and type) and natural language
    """
    from stefutil import sic

    ner_sample, s2lp = sample, span2logprob
    span, (sample_s, sample_e) = sample_str, sample_span
    label_start = 'Named Entities'
    assert span.count(label_start) == 1  # sanity check
    idx = span.index(label_start)  # get the substring starting from `label_start` till the end
    label_str = span[idx:]  # only the annotations part
    label_s = sample_s + idx
    label_e = label_s + len(label_str)
    lp_out = s2lp(span=label_str, index_start=label_s, index_end=label_e)
    lp_sample = lp_out.avg_logprob
    lp_ets = []

    # if lp_kd == '1-stage-annotations':
    # now get lp for each entity annotation, e.g. log prob of
    #   `joker (Character)` in `Named Entities: [joker (Character), dc universe (Title)]`
    enms, ets = ner_sample.entity_names, ner_sample.entity_types
    # need to split on the original entity list, cos the entity names may be already modified after processing,
    #   due to e.g. dropping enclosing quotes
    m_e = patterns.match_row(text=entities_str, pattern=pattern_entity, desc='entities')
    assert m_e is not None

    entities_raw = m_e.group('entities')
    if entity_sep not in entities_raw and '] [' in entities_raw:
        # for edge case: wrong annotation format, e.g.
        #   `Named Entities: [Honda Motor Co. (organization)] [Mexico (location)]`
        entities_raw = [e.strip() for e in entities_raw.split('] [')]
    else:
        entities_raw = [e.strip() for e in entities_raw.split(entity_sep)]  # a more lenient split

        assert isinstance(entity_pair_map, EntityPairEncloseType)
        assert entity_pair_map.open_char == '(' and entity_pair_map.close_char == ')'
        # edge case checking just like in entities extraction: an entity span contains comma itself
        if any('(' not in e or ')' not in e for e in entities_raw):
            entities_ = edit.merge_entities_on_separator(entities=entities_raw, entity_sep=entity_sep, entity_pair_map=entity_pair_map)
            # sic(entities_, list(enms))
            # assert len(entities_) == len(enms)  # sanity check
            entities_raw = entities_

    # sic(entities_raw, entities_str, pattern_entity)

    entity_spans = []
    for i, (annot_str, enm, et) in enumerate(zip(entities_raw, enms, ets)):
        n_found = label_str.count(annot_str)
        if n_found not in [1, 2, 3]:
            sic(annot_str, label_str, n_found)
        assert n_found in [1, 2, 3]  # sanity check

        if n_found == 1:
            idx = label_str.index(annot_str)
        else:
            # Edge case: both the correct match and a substring match
            #   `Named Entities: [comedy (Genre), romantic comedy (Genre)]\n`
            #       => Filter w/ re pattern match, the starting punc must be a comma or a bracket
            assert n_found in [2, 3]
            # sic(sample_str, label_str, annot_str, label_str.count(annot_str))
            import regex
            # pat = re.compile(rf'(\[|, )(?P<annot>{re.escape(annot_str)})(, |])')
            pat = regex.compile(rf'(\[|, )(?P<annot>{re.escape(annot_str)})(, |])')
            # sic(annot_str, pat, label_str)
            # need to activate `overlapped` cos we include the boundaries for exact span match
            ms = list(pat.finditer(label_str, overlapped=True))
            # sic(ms, len(ms))
            if n_found == 3:
                assert len(ms) == 1
                idx = ms[0].start('annot')
            else:
                assert n_found == 2
                if len(ms) == 1:  # sanity check just 1 valid match
                    idx = ms[0].start('annot')
                else:
                    # Edge case: same entity annotated twice, e.g.
                    #   `1. The Berlin Wall divided the city of Berlin for 28 years, creating two separate countries.
                    #   Named Entities: [Berlin (location), Berlin (location)]`
                    #   `3. "I'm looking for a place with great brunch options for Sunday brunch. Any recommendations?"
                    #   Named Entities: [great (Rating), brunch (Amenity), Sunday (Hours), brunch (Amenity)]`
                    assert len(ms) == 2
                    sic(ms, annot_str, label_str, entities_str, entities_raw, sample_str, sample_span, sample)
                    # assert len(entities_raw) == 2 and len(set(entities_raw)) == 1
                    assert len(entities_raw) - len(set(entities_raw)) == 1  # sanity check just 1 entity duplicated once
                    # so in 1st iteration, should get 1st match, and in 2nd iteration, should get 2nd match
                    idxs_enm = [i_ for i_, e in enumerate(entities_raw) if e == annot_str]
                    assert len(idxs_enm) == 2
                    assert i == idxs_enm[0] or i == idxs_enm[1]  # sanity check
                    is_1st_iter = i == idxs_enm[0]
                    # i_curr = [i_ for i_ in idxs_enm if i_ == i]
                    # assert len(i_curr) == 1  # sanity check
                    idx = ms[0 if is_1st_iter else 1].start('annot')
            # sic(idx, label_str[idx:idx+len(annot_str)])
            # raise NotImplementedError
        annot_s = label_s + idx
        annot_e = annot_s + len(annot_str)
        entity_spans.append((annot_s, annot_e))
        # sic(annot_str)
        lp_out = s2lp(span=annot_str, index_start=annot_s, index_end=annot_e, sample_type='1-stage-annotation')
        lp_ = lp_out.avg_logprob
        # lp_ = lp_out.top_n_avg_logprob(n=3)  # try another metric to select uncertain (potentially wrong) annotations

        # save just the logprob corresponding to each entity annotation,
        #   no **explicit** mapping to the NER sample for now, since they may be filtered for duplicates
        # et_lps.append((sent, enm, et, lp_))
        lp_ets.append(lp_)
    n_enm, n_lp = len(enms), len(lp_ets)
    if n_enm != n_lp:
        # A valid NER extracted sample, doesn't match the number of entity annotations generated,
        #   possibly due to, LLM didn't annotate multi-occurring entities K times, e.g.
        #       `Query: "does the movie I Want to Hold Your Hand feature the song I Want to Hold Your Hand"\n'
        #       Named Entities: [I Want to Hold Your Hand (Song)]`
        #       `3. "I want to watch a film with an amazing song, maybe one with a song by Spike Lee?"
        #       Named Entities: [song (Song), Spike Lee (Director)]`
        # Check this is the case, and if so, duplicate the same logprob, for each occurrence
        if (n_enm, n_lp) == (2, 1):
            if enms[0] != enms[1]:
                sic(enms, lp_ets, entities_str, entities_raw)
            assert enms[0] == enms[1]  # sanity check the 2 entities are the same
            lp_ets = lp_ets * 2  # duplicate the same logprob
        elif (n_enm, n_lp) in [(3, 2), (3, 1), (4, 3), (4, 2), (4, 1), (5, 4), (5, 3), (5, 2), (6, 5), (7, 4), (8, 7)]:
            # get the logprob corresponding to each raw entity annotation,
            #   send them to the potentially duplicated, actually processed entities
            # sic(entities_raw)
            enms_raw, ets_raw = zip(*[entity_pair_map.decode(pair=e) for e in entities_raw])
            enms_raw_ = []
            for enm in enms_raw:
                if (enm[0], enm[-1]) == ('[', ']'):  # similar edge case handling: drop enclosing brackets
                    enm = enm[1:-1]
                enms_raw_.append(enm)
            enms_raw = enms_raw_

            assert len(set(enms_raw)) == len(enms_raw)  # sanity check raw LLM annotations have no duplicates
            enm2lp = dict(zip(enms_raw, lp_ets))
            # sic(enm2lp, enms)
            lp_ets = [enm2lp[enm] for enm in enms]
            # raise NotImplementedError
        else:
            sic(sample, sample_str, entities_str)
            sic(lp_ets, enms, len(lp_ets), len(enms))
            raise NotImplementedError
    assert len(lp_ets) == len(enms)  # sanity check logprob maps to each entity annotation
    if spans_overlap(spans=entity_spans):
        sic(entity_spans, enms, lp_ets)
        sic(sample_str, entities_str, entities_raw)
        sic(sample)
    assert not spans_overlap(spans=entity_spans)  # sanity check entity spans don't overlap

    # lp_sample = lp_sample  # use the entire annotation span
    lp_sample = min(lp_ets) if len(lp_ets) > 0 else lp_sample  # use the most uncertain annotation
    return Sample2LogProbsOutput(logprob=lp_sample, entity_logprobs=lp_ets)


# def split_log_prob(completion: str, logprobs: List[Dict[str, Any]], n_sample: int = None) -> List[LogProbOutput]:
#     assert completion_n_logprobs_match(completion=completion, logprobs=logprobs)
#
#     sic(completion, logprobs, n_sample)
#     raise NotImplementedError


def log_n_save_samples_w_logprob(
        samples_w_logprob: List[Dict[str, Any]] = None, output_path: str = None, logger: logging.Logger = None, **log_kwargs
):
    samples_w_lp = samples_w_logprob
    with open(os_join(output_path, 'logprobs-sample.json'), 'w') as f:
        json.dump(samples_w_lp, f, indent=4)

    samples_sorted = sorted(samples_w_lp, key=lambda x: x['logprob'])  # sort samples by logprob and log
    lst_samples_log = []
    for sample in samples_sorted:
        sample: Dict[str, Any]
        sample = deepcopy(sample)
        lp = sample.pop('logprob')
        sample = sdpc(sample, as_str=False)
        sample['logprob'] = f'{lp:.3e}'
        lst_samples_log.append(sample)
    (logger or _logger).info(f'Samples sorted by log prob: {pl.i(lst_samples_log, indent=2)}', **log_kwargs)
    return samples_sorted


def log_n_save_triples_w_logprob(
        triples_w_logprob: [List[TripleWLogProb], List[Dict[str, Any]]],
        entity_types: List[str] = None, output_path: str = None, logger: logging.Logger = None, top_n_log: Union[int, str] = 100,
        **log_kwargs
):
    """
    Log NER entity annotations sorted by entity-wise log prob
        and save to file
    """
    if isinstance(triples_w_logprob[0], dict):
        assert all(isinstance(t, dict) for t in triples_w_logprob)  # sanity check in the dict format instead of tuples
        triples_tup = []
        for t in triples_w_logprob:
            sent, span, lb, lp = t['sentence'], t['span'], t['entity_type'], t['average_logprob']
            triples_tup.append((sent, span, lb, lp))
    else:
        assert all(isinstance(t, tuple) for t in triples_w_logprob)
        triples = []
        for t in triples_w_logprob:
            # sent, span, lb, lp = t['sentence'], t['span'], t['entity_type'], t['average_logprob']
            # triples.append((sent, span, lb, lp))
            sent, span, lb, lp = t
            triples.append(dict(sentence=sent, span=span, entity_type=lb, average_logprob=lp))
        triples_tup, triples_w_logprob = triples_w_logprob, triples

    # lst_log = sorted(triples_w_logprob, key=lambda t: t[-1])  # sort by logprob
    lst_log = sorted(triples_tup, key=lambda t: t[-1])
    # omit the sentence part, log as (span, type, logprob) triplets
    lst_log = [(span, lb, lp) for (sent, span, lb, lp) in lst_log]
    # if not set(lb for (span, lb, lp) in lst_log) == set(entity_types):
    #     sic(entity_types, set(lb for (span, lb, lp) in lst_log))
    assert set(lb for (span, lb, lp) in lst_log) == set(entity_types)  # sanity check

    # template a condensed version for easy manual inspection
    # top_n_log = 300
    # triplets_log = [f'{span} ({lb}): {logprob:.3e}' for (span, lb, logprob) in triplets_log[:top_n_log]][:top_n_log]
    # log as (type => span => logprobs) dict

    if top_n_log == 'all':
        top_n_log = None
    else:
        assert isinstance(top_n_log, int)
    dict_log = defaultdict(lambda: defaultdict(list))
    for (span, lb, logprob) in lst_log:
        dict_log[lb][span].append(logprob)
    # get top k spans for each type by average logprob
    dict_log = {lb: dict(sorted(d.items(), key=lambda kv: np.mean(kv[1]))[:top_n_log]) for lb, d in dict_log.items()}

    def template_span(s: str, lps: List[float]) -> Union[str, Dict[str, Any]]:
        if len(lps) == 1:
            return f'{lps[0]:.3e}'
        else:
            avg = np.mean(lps)
            d = dict(avg=f'{avg:.3e}', lps=[f'{lp_:.3e}' for lp_ in lps])
            # return f'{pl.i(d)}'
            return d

    dict_log = {lb: {s: template_span(s, lps) for s, lps in d.items()} for lb, d in dict_log.items()}
    # order the dict by type
    dict_log = {lb: dict_log[lb] for lb in entity_types if lb in dict_log}
    logger.info(f'Entity Classifications sorted by log prob: {pl.i(dict_log, indent=3)}', **log_kwargs)

    if output_path is not None:
        with open(os_join(output_path, 'logprobs-triple.json'), 'w') as f:
            json.dump(triples_w_logprob, f, indent=2)
        logger.info(f'Classification log probs written to {pl.i(stem(output_path))}', **log_kwargs)
