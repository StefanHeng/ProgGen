import re
import logging
from typing import Dict, Tuple, List, Union, Any
from dataclasses import dataclass

from stefutil import *
from src.util import patterns
from src.data_util import completions
from src.data_util.prettier import EdgeCases


__all__ = [
    'CheckIndexOutput', 'check_match_group_indices', 'pattern_enumerated_index',
    'NaturalSample', 'BioSample', 'ExtractedSample', 'GroupedStrSample',
    'SplitSamplesOutput', 'split_samples'
]


_logger = get_logger('Sample-Split')


@dataclass
class NaturalSample:
    sentence: str = None
    entities: List[str] = None


@dataclass
class BioSample:
    tokens: List[str] = None
    tags: List[str] = None


ExtractedSample = Union[NaturalSample, BioSample]


@dataclass
class CheckIndexOutput:
    # indices_got=idxs_got, indices_correct=idxs_correct, indices_are_correct=idxs_are_correct, d_log=d_log
    match_success: bool = None
    indices_got: List[int] = None
    indices_expect: List[int] = None
    d_log: Dict[str, Any] = None


def check_match_group_indices(
        ms: List[re.Match], group_name: str = 'idx', logger: logging.Logger = None, ec: EdgeCases = None, **kwargs
) -> CheckIndexOutput:
    idxs_got = [int(m.group(group_name)) for m in ms]
    idxs_exp = list(range(1, len(ms) + 1))  # sanity check idx is continuous
    idx_match = idxs_got == idxs_exp
    d_log = {**(kwargs or dict()), 'idxs-got': idxs_got, 'idxs-correct': idxs_exp}

    def log_edge(msg: str = None, kind: str = None):
        ec(msg=msg, kind=kind) if ec else (logger or _logger).warning(msg)
    if not idx_match:
        # edge case after GPT3.5 update: sample enum may start from e.g. `8.`
        idx_1st = idxs_got[0]
        idxs_exp = list(range(idx_1st, idx_1st + len(ms)))
        idx_match = idxs_got == idxs_exp
        if idx_match:
            log_edge(msg=f'Edge case: sample enum starts from non-1st index w/ {pl.i(d_log, indent=1)}', kind='wrong-start-enum')
    if not idx_match:
        log_edge(msg=f'Edge case: sample enum mismatch w/ {pl.i(d_log, indent=1, align_keys=1)}', kind='wrong-enum')
    return CheckIndexOutput(match_success=idx_match, indices_got=idxs_got, indices_expect=idxs_exp, d_log=d_log if not idx_match else None)


@dataclass
class GroupedStrSample:
    sentence: str = None
    entities: str = None
    entities_raw: str = None


@dataclass
class SplitSamplesOutput:
    success: bool = None
    matches: List[re.Match] = None
    pattern_map: Dict[re.Pattern, List[str]] = None
    groups: List[str] = None
    samples: List[str] = None
    spans: List[Tuple[int, int]] = None  # spans corresponding to each match
    grouped: List[GroupedStrSample] = None
    has_enum_prefix: bool = None
    indices_check: CheckIndexOutput = None
    d_log: Dict[str, Any] = None


pattern_enumerated_index = [
    re.compile(r'(?P<idx>\d+)\. '),
    re.compile(r'(?P<idx>\d+)\.')  # just add a space, TODO: not sure why `( )?` doesn't work
]


def split_samples(
        completion: str = None, pattern: Union[re.Pattern, List[re.Pattern]] = None, has_enum_prefix: bool = None, has_cot: bool = False,
        ec: EdgeCases = None, filename: str = None, silent: bool = False, return_match_map: bool = None
) -> SplitSamplesOutput:
    if completion[-1] != '\n':  # append newline if not present
        completion += '\n'
    if has_enum_prefix is None:
        has_enum_prefix = completions.completion_has_enum_prefix(completion=completion)
    pattern_map = None
    if has_cot:
        if return_match_map:
            raise NotImplementedError
        assert has_enum_prefix
        # a single pattern match for optional and arbitrary number of reasoning bullets is challenging
        #   => split by enumeration index first
        ms_idxs: List[re.Match] = patterns.find_non_overlap_matches(pattern=pattern_enumerated_index, text=completion, return_matches=True)

        start_idxs = [m.start() for m in ms_idxs]  # # split the completions by starting indices; will be non-overlapping by construction
        # not necessarily the case, LLM may say, e.g.
        #   `For each sentence, let's identify the entity names and their corresponding entity types:`
        # assert start_idxs[0] == 0
        end_idxs = start_idxs[1:] + [len(completion)]
        samples = [completion[s:e] for s, e in zip(start_idxs, end_idxs)]

        ms = []
        for sample in samples:
            ms_ = patterns.find_match(text=sample, pattern=pattern)
            n_match = len(ms_)
            if n_match > 1:
                sic(completion, sample, ms_)
            assert n_match <= 1
            if n_match == 1:  # edge case: last sample may not be complete, e.g. just reasoning, no annotations
                ms.append(ms_[0])
            else:
                d_log = dict(filename=filename, sample=sample, pattern=pattern)
                msg = f'Edge case: no match found in enumerated sample w/ {pl.i(d_log)}'
                ec(msg=msg, kind='sample-mismatch')
        if len(ms) == 0:
            d_log = dict(filename=filename, completion=completion, pattern=pattern)
            if silent:
                return SplitSamplesOutput(success=False, has_enum_prefix=has_enum_prefix, d_log=d_log)
            else:
                raise ValueError(f'No match found in any enumerated sample w/ {pl.i(d_log)}')
    else:
        try:
            args = dict(pattern=pattern, text=completion, return_matches=True)
            ms: List[re.Match]
            if return_match_map:
                out = patterns.find_non_overlap_matches(**args, return_match_map=return_match_map)
                ms, pattern_map = out.matches, out.pattern_map
            else:
                ms = patterns.find_non_overlap_matches(**args)
        except ValueError:
            d_log = dict(filename=filename, completion=completion, pattern=pattern)
            if silent:
                return SplitSamplesOutput(success=False, has_enum_prefix=has_enum_prefix, d_log=d_log)
            else:
                raise ValueError(f'No sample match found in completion w/ {pl.i(d_log)}')

    samples = [m.group() for m in ms]

    check_idx = None
    if has_enum_prefix:
        check_idx = check_match_group_indices(ms=ms, filename=filename, completion=completion, samples=samples, ec=ec)
    sents = [m.group('sentence') if 'sentence' in m.groupdict() else None for m in ms]
    ents = [m.group('entities') if 'entities' in m.groupdict() else None for m in ms]
    grouped = [GroupedStrSample(sentence=sent, entities=ent) for sent, ent in zip(sents, ents)]
    groups = [m.group() for m in ms]
    spans = [m.span() for m in ms]
    return SplitSamplesOutput(
        success=True, matches=ms, groups=groups, samples=samples, grouped=grouped, spans=spans,
        has_enum_prefix=has_enum_prefix, indices_check=check_idx, pattern_map=pattern_map
    )
