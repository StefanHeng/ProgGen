import re
import logging
from copy import deepcopy
from typing import Dict, Tuple, List, Union, Any, Callable, Iterable
from collections import defaultdict, Counter
from dataclasses import asdict

from stefutil import *
from src.util.ner_example import *
from src.util.sample_formats import *
from src.util import patterns


__all__ = [
    'color_code_prompt', 'print_prompts', 'EdgeCases',
    'AnnotationsTemplator', 'at', 'atc',
    'highlight_span_in_sentence',
    'SampleDictPrettier', 'sdp', 'sdpc',
]


_logger = get_logger('Prettier-Sample')


def color_code_prompt(prompt: str = None) -> str:
    """
    Color-code a prompt for semantic segmentation by a simple heuristic
    """
    # first, split up the prompt into sections
    ret = ''
    sep = '\n\n'
    segs = prompt.split(sep)
    it = iter(segs)
    prev = None
    curr = next(it)
    assert curr is not None
    cs = ['b', 'm', 'r', 'y', 'g']
    i_c = None
    while curr is not None:
        # iteratively check, does the difference between current and next segment indicate a new section?
        # if prev is not None:
        # declare different is current segment is pretty long, via either char count or line count
        long_enough = False
        n_lines = curr.count('\n')
        n_chars = len(curr)
        if n_chars > 250:
            long_enough = True
        elif n_chars > 150 and n_lines > 0:
            long_enough = True
        elif n_lines > 0 and all(len(c) > 60 for c in curr.split('\n')):
            long_enough = True
        elif '\n' not in curr and n_chars > 120:
            long_enough = True
        elif n_lines > 3:
            long_enough = True
        elif '---' in curr or 'Examples:' in curr:
            long_enough = True
        if prev is None:
            i_c = 0
        elif long_enough:
            i_c += 1
            i_c = i_c % len(cs)
        ret += f'{pl.i(curr, c=cs[i_c])}{sep}'

        prev = curr
        curr = next(it, None)
    return ret


def print_prompts(prompt: Union[Callable[[], str], List[str], Iterable[str]], n: int = None) -> List[str]:
    """
    Calls `prompt_constructor` `n` times and print to console as sanity check
    """
    if isinstance(prompt, list):
        assert n is None
        prompts = prompt
        n = len(prompts)
    elif hasattr(prompt, '__iter__'):
        prompts = list(prompt)
        if n is not None:
            n = min(n, len(prompts))
    else:
        n = n or 5
        prompts = [prompt() for _ in range(n)]

        if any(not p for p in prompts):
            prompts = [p for p in prompts if p]  # filter out empty/None prompts
            n = len(prompts)
            assert n > 0
    assert all(isinstance(p, str) for p in prompts)  # sanity check

    # prompts = [f'Prompt {i}:\n{pl.i(p)}' for i, p in enumerate(prompts, start=1)]
    prompts = [f'Prompt {i}:\n{color_code_prompt(p)}' for i, p in enumerate(prompts, start=1)]
    if n == 1:
        print(prompts[0])
    else:
        for i in range(n):
            sep = '\n\n\n' if i != n - 1 else ''
            print(f'{prompts[i]}{sep}')
    return prompts


class EdgeCases:
    """
    Stores edge cases encountered during processing
    """

    type2counter_prefix = {
        # for NER extraction
        'not-allowed-entity-type': 'Invalid entity types',
        'missing-entity-type': 'Entities w/ type missing',
        'wrong-sample-count': 'Completions w/ wrong sample count',
        'no-sample-decoded': 'Completions with no samples extracted',
        'swap-non-ascii-quote': 'Non-ASCII quotes swapped',
        'entity-not-found': 'Entities spans generated but not found in sentence',
        'entity-overlap-drop': 'Entity spans dropped to resolve overlap',
        'wrong-entity-annotation-format': 'LLM-generated entity annotation in wrong format',
        'A-as-entity-&-a-in-sentence': 'LLM annotation contains `A` and `a` is in sentence',
        'sentence-too-short': 'Sentence too short for dropping enclosing quotes',

        # for mapping back entities
        'multi-occur-entity-add-index': 'Index added for entities that appears multiple times in sentence',
        'super-string-entity-add-index': 'Index added for entities that are sub-strings of other entities in sentence',

        # for sentence extraction
        'filtered': 'Filtered sentences',
        'drop-inline-type': 'Sentences dropped for inline type annotations',
        'drop-emph': 'Emphasized terms in sentence - enclosing brackets dropped',
        'drop-emph-word': 'Potentially broken words due to dropping brackets',

        # for span extraction
        'entity-span-drop-puncs': 'Entity spans w/ enclosing quotes/brackets dropped',
        'entity-span-quote-on-side': 'Entity spans w/ quotes on at least one side',

        # for span CLS label extraction
        'negative-span-drop': 'Filtered spans for not a relevant entity',
        'drop-span-emph': 'Sentences w/ emphasized spans dropped',
        'wrong-label-format': 'Unexpected generated labels',
        'generated-span-mismatch': 'Generated span different from that in the prompt',
        'multi-occur-entity': 'Multi-occurring entities',
        'overlapping-entity-span-filtered': 'Groups of overlapping entity spans',

        'no-cot': 'Completions w/o CoT reasoning',
        'generated-incomplete-spans': "Completions that didn't finish generating spans",
        'generated-unrequested-spans': 'Completions that generated unrequested spans',
        'incorrect-sample-format': 'Completions w/ incorrect sample format',
        'incorrect-sample-format-failed': 'Completions w/ incorrect sample format resulting in extraction failure',

        # for span logprob extraction
        'logprob-idx-adjust': 'Actual LLM-generated tokens for label span',

        # for highlighting span in sentence
        'multiple-entity-span-match-highlights': 'Multiple entity spans found in sentence',

        # for (sentence, span, entity type) triple correction
        'wrong-span-gen-same': 'LLM labels as wrong span but provided the same span',
        'wrong-span-end-period': 'LLM labels as wrong span but provided the same span w/ period',
        'wrong-type-gen-same': 'LLM labels as wrong type but provided the same type',
        'type-correct-unknown': 'LLM generated unseen corrected type',
        'corrected-input-span': 'LLM further corrected input span in prompt',
        'modified-input-span-&-corrected-to-same': 'LLM modified the input span and corrected it back to the input span',
        'modified-input-span-&-corrected-as-substring': 'LLM modified the input span and corrected it as a further substring',
        'declare-correct-but-generated-correction': 'LLM declared annotation is correct but generated a correction',
        'declare-yes-but-corrected-input-span': 'LLM declared annotation is correct but provided a corrected input span',
        'declare-wrong-type-but-corrected-input-span': 'LLM declared entity type is wrong but provided a corrected input span',
        'declare-none-entity-but-corrected-input-span': 'LLM declared none-entity but provided a corrected input span',
        'declare-none-entity-on-different-span': 'LLM declared none-entity on a completely different span',
        'declare-wrong-type-but-on-different-span': 'LLM declared entity type is wrong but annotated a completely different span',
        'declare-wrong-span-but-on-different-span': 'LLM declared span is wrong but annotated a completely different span',
        'declare-wrong-type-and-corrected-span': 'LLM declared entity type is wrong and also corrected the span',
        'corrected-span-too-different': 'LLM-corrected span has no overlap w/ input span',
        'corrected-span-not-in-sentence': 'LLM-corrected span not in sentence',
        'corrected-span-multi-occur': 'LLM-corrected span appears multiple times in sentence',
        'correction-type-missing': "LLM provided correction but didn't specify correction type",
        'wrong-span-no-correction': "LLM declared wrong span boundary but didn't provide a correction",
        'wrong-type-no-correction': "LLM declared wrong entity type but didn't provide a correction",
        'generated-span-not-in-sentence': 'LLM generated span not in sentence',

        'corrected-entity-span-overlap': 'LLM-corrected-span causes sample entity spans to overlap',
        'confused-choice-letter': "LLM generated choice description don't match choice letter",
        'span-not-re-generated': "LLM didn't re-generate the span in non-last-group response",

        # for BERT-prediction on LLM-generated samples
        'multiple-type-in-llm-annotated-entity': 'BERT predicted multiple types for a single LLM-annotated entity',
        'swapped-starting-i-tag': 'BERT-predicted starting I-tags swapped to B-tags',
    }

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or _logger
        # failure edge case? => Edge case type => List of logging messages
        self.edge_cases: Dict[bool, Dict[str, List[str]]] = {b: defaultdict(list) for b in [True, False]}
        self.type2counter = {b: defaultdict(Counter) for b in [True, False]}  # failure edge case? => Edge case type => Counter

        self.entity_pair_map = get_default_entity_pair_map(sample_format='natural-pair-v2')

    def clear(self):
        """
        Clear records of earlier edge cases
        """
        self.edge_cases = {b: defaultdict(list) for b in [True, False]}
        self.type2counter = {b: defaultdict(Counter) for b in [True, False]}

    def __call__(
            self, msg: Union[str, Dict[str, Any]] = None, kind: str = None, args: Dict[str, Any] = None, failed: bool = False,
            disable_std_log: bool = False
    ):
        """
        Logs and saves edge cases

        :param msg: Logging message
        :param kind: Edge case type
        :param args: Additional arguments to keep track of
        :param failed: Whether the edge case results in sample extraction failure
        :param disable_std_log: If True, don't log to stdout
            Intended to save terminal output space, there must be a file write handler to keep a record
        """
        assert kind is not None
        if isinstance(msg, dict):
            # cache some frequent edge cases, intended for natural-pair-v2
            assert kind in ['none-entity', 'multi-entity-annot-missing']
            if kind == 'none-entity':
                msg = f'Edge case: entity list contains [{pl.i("None")}] in sample {pl.fmt(msg)}'
            else:
                assert kind == 'multi-entity-annot-missing'
                msg = f'Entity name appears multiple times but only one annotation w/ {pl.fmt(msg)}'
            return self.__call__(msg=msg, kind=kind)

        log_args = dict()
        if disable_std_log:
            handlers = self.logger.handlers  # sanity check a file-write handler is available
            assert len(handlers) > 0 and any(isinstance(h, logging.FileHandler) for h in handlers)
            log_args['extra'] = dict(block='stdout')
        self.logger.warning(msg, **log_args)
        self.add(kind=kind, msg=msg, args=args, failed=failed)

    def add(self, kind: str = None, msg: str = None, args: Dict[str, Any] = None, failed: bool = False):
        if args is not None:
            t2c = self.type2counter[failed]

            # for NER extraction
            if kind == 'not-allowed-entity-type':
                enms_na, ets_na = args['entity_names_not_allowed'], args['entity_types_not_allowed']
                not_allowed = list(zip(enms_na, ets_na))
                not_allowed = [self.entity_pair_map(pl.i(e, c='y'), pl.i(t, c='r')) for e, t in not_allowed]
                t2c[kind].update(not_allowed)
            elif kind == 'missing-entity-type':
                t2c[kind].update(args['entities'])
            elif kind in ['wrong-sample-count', 'no-sample-decoded']:
                fnm, n_exp, n_got = args['filename'], args['n_sample_expect'], args['n_sample_found']
                fnm, n_exp, n_got = pl.i(fnm, c='r'), pl.i(n_exp, c='y'), pl.i(n_got, c='g')
                t2c[kind].update([f'{fnm}: {n_got} vs {n_exp}'])
            elif kind == 'swap-non-ascii-quote':
                t2c[kind].update(args['counts'])
            elif kind == 'entity-not-found':
                t2c[kind].update(args['missing_entity_names'])
            elif kind == 'entity-overlap-drop':
                ovl, rsv = args['overlapping_entity_spans'], args['resolved_entity_spans']
                ovl, rsv = pl.i(ovl, c='r'), pl.i(rsv, c='g')
                t2c[kind].update([f'{ovl} => {rsv}'])
            elif kind == 'wrong-entity-annotation-format':
                ori_str, extracted = args['entities'], args['entities_extracted']
                ori_str, extracted = pl.i(ori_str, c='r'), pl.i(extracted, c='g')
                t2c[kind].update([f'{ori_str} => {extracted}'])
            elif kind in ['A-as-entity-&-a-in-sentence', 'sentence-too-short']:
                t2c[kind].update([args['sentence']])

            elif kind in ['multi-occur-entity-add-index', 'super-string-entity-add-index']:
                t2c[kind].update([args['span']])

            # for sentence extraction
            elif kind in ['filtered', 'drop-inline-type']:
                t2c[kind].update([args['sentence']])
            elif kind == 'drop-emph':
                t2c[kind].update(args['emphasized'])
            elif kind == 'drop-emph-word':
                t2c[kind].update(args['words'])

            # for span extraction
            elif kind in ['entity-span-quote-on-side', 'entity-span-drop-puncs']:
                t2c[kind].update(args['spans'])

            # for span CLS label extraction
            elif kind == 'negative-span-drop':
                t2c[kind].update([args['span']])
            elif kind == 'drop-span-emph':
                t2c[kind].update([args['original_sentence']])
            elif kind == 'generated-span-mismatch':
                span, span_expected = args['span_generated'], args['span_expected']
                t2c[kind].update([f'{span_expected} => {span}'])
            elif kind == 'declare-correct-but-generated-correction':
                span_gen, correction = args['span_generated'], args['correction']
                span_gen, correction = pl.i(span_gen, c='y'), pl.i(correction, c='g')
                t2c[kind].update([f'{span_gen} vs {correction}'])
            elif kind == 'wrong-label-format':
                span, entity_type = args['span'], args['entity_type']
                t2c[kind].update([f'{span} => {entity_type}'])
            elif kind == 'multi-occur-entity':
                t2c[kind].update(args['entity_names'])
            elif kind == 'overlapping-entity-span-filtered':
                t2c[kind].update(args['span_groups'])
            elif kind in [  # `no-cot` and `incorrect-sample-format` also for span extraction
                'no-cot', 'generated-incomplete-spans', 'generated-unrequested-spans',
                'incorrect-sample-format', 'incorrect-sample-format-failed'
            ]:
                t2c[kind].update([args['filename']])

            # for span logprob extraction
            elif kind == 'logprob-idx-adjust':
                start, end = args.get('initial_token'), args.get('final_token')
                s, e = start is not None, end is not None
                assert (s or e) and not (s and e)  # sanity check only one provided
                tok = f'Start: [{pl.i(start, c="b")}]' if s else f'End: [{pl.i(end, c="b")}]'
                t2c[kind].update([tok])

            # for highlighting span in sentence
            elif kind == 'multiple-entity-span-match-highlights':
                sent, span = args['sentence-highlighted'], args['entity_name']
                t2c[kind].update([f'{span} <= {sent}'])

            # for (sentence, span, entity type) triple correction
            elif kind in ['wrong-span-gen-same', 'wrong-span-end-period']:
                span, corrected_span = args['span'], args['corrected_span']
                if span == corrected_span or f'{span}.' == corrected_span:  # exactly the same or difference is a trailing period
                    t2c[kind].update([span])
                else:
                    assert span.lower() == corrected_span.lower()  # only difference is case
                    span, corrected_span = pl.i(span, c='y'), pl.i(corrected_span, c='g')
                    t2c[kind].update([f'{span} vs {corrected_span}'])
            elif kind == 'wrong-type-gen-same':
                t2c[kind].update([args['entity_type']])
            elif kind == 'type-correct-unknown':
                t2c[kind].update([args['corrected_entity_type']])
            elif kind in [
                'corrected-input-span', 'declare-yes-but-corrected-input-span', 'declare-wrong-type-but-corrected-input-span',
                'declare-none-entity-but-corrected-input-span', 'declare-none-entity-on-different-span',
                'modified-input-span-&-corrected-to-same'
            ]:
                span_exp, span_got = args['span_expected'], args['span_generated']
                span_exp, span_got = pl.i(span_exp, c='y'), pl.i(span_got, c='g')
                t2c[kind].update([f'{span_exp} => {span_got}'])
            elif kind in ['modified-input-span-&-corrected-as-substring']:
                span_exp, span_got, span_corrected = args['span_expected'], args['span_generated'], args['correction']
                span_exp, span_got, span_corrected = pl.i(span_exp, c='r'), pl.i(span_got, c='y'), pl.i(span_corrected, c='g')
                t2c[kind].update([f'{span_exp} => {span_got} => {span_corrected}'])
            elif kind in ['corrected-entity-span-overlap']:
                original_spans_n_corrections = args['original_spans_n_corrections']
                elms = [f'{pl.i(s, c="y")} => {pl.i(c, c="g")}' for s, c in original_spans_n_corrections]
                t2c[kind].update(elms)

                # span_ori, span_crt = args['original_span'], args['corrected_span']
                # span_ori, span_crt = pl.i(span_ori, c='y'), pl.i(span_crt, c='r')
                # sic(span_ori, span_crt)
                # raise NotImplementedError
                # t2c[kind].update([f'{span_ori} => {span_crt}'])
            elif kind in [
                'declare-wrong-type-but-on-different-span',
                'declare-wrong-span-but-on-different-span',
                'declare-wrong-type-and-corrected-span'
            ]:
                span_exp, span_got, et = args['span_expected'], args['span_generated'], args['correction']
                span_exp, span_got, et = pl.i(span_exp, c='y'), pl.i(span_got, c='r'), pl.i(et, c='g')
                t2c[kind].update([f'{span_exp} vs {span_got} ({et})'])
            elif kind in ['corrected-span-too-different', 'corrected-span-not-in-sentence', 'corrected-span-multi-occur']:
                span, span_crt = args['span'], args['corrected_span']
                span, span_crt = pl.i(span, c='y'), pl.i(span_crt, c='g')
                t2c[kind].update([f'{span} => {span_crt}'])
            elif kind in ['correction-type-missing', 'wrong-span-no-correction', 'wrong-type-no-correction']:
                t2c[kind].update([args['filename']])
            elif kind in ['confused-choice-letter']:
                label, correct_letter = args['label'], args['correct_letter']
                label, correct_letter = pl.i(label, c='r'), pl.i(correct_letter, c='g')
                t2c[kind].update([f'{correct_letter} vs {label}'])
            elif kind in ['span-not-re-generated']:
                t2c[kind].update([args['filename']])
            elif kind == 'generated-span-not-in-sentence':
                span_exp, span_gen = args['span_expected'], args['span_generated']
                span_exp, span_gen = pl.i(span_exp, c='g'), pl.i(span_gen, c='r')
                t2c[kind].update([f'{span_exp} v.s. {span_gen}'])

            elif kind == 'multiple-type-in-llm-annotated-entity':
                et_dist = args['type_dist']
                t2c[kind].update([pl.i(et_dist)])
            elif kind == 'swapped-starting-i-tag':
                t2c[kind].update([args['entity_type']])
            else:
                raise NotImplementedError(kind)
        self.edge_cases[failed][kind].append(msg)

    @property
    def have_edge_case(self) -> bool:
        return any(len(v) > 0 for v in self.edge_cases.values())

    def summary(self) -> str:
        ret = []
        for failed in (True, False):
            kd2kd_c = {k: len(v) for k, v in self.edge_cases[failed].items()}
            if len(kd2kd_c) > 0:
                prefix = 'Failure Edge cases encountered' if failed else 'Edge cases encountered'
                ret.append(f'{prefix}: {pl.i(kd2kd_c, indent=1)}')

            for kind, arg2c in self.type2counter[failed].items():
                if len(arg2c) > 0:
                    if all(v == 1 for v in arg2c.values()):  # convert to list if every key appears only once
                        c = list(arg2c.keys())
                    else:  # enforce ordering by count in output
                        c = dict(arg2c.most_common())
                    ret.append(f'{self.type2counter_prefix[kind]} ({pl.i(kind, c="y")}): {pl.i(c, indent=1)}')
        return '\n'.join(ret)


class AnnotationsTemplator:
    """
    Template entity annotations for a sequence
        Intended for easier & compact manual examination in logging
    """
    def __init__(
            self, sample_format: str = 'natural-pair-v2', entity_sep: str = None, entity_pair_map: EntityPairTemplate = None,
            color: bool = None
    ):
        if sample_format != 'natural-pair-v2':
            raise NotImplementedError
        self.entity_sep = entity_sep or get_default_entity_sep(sample_format=sample_format)
        self.entity_pair_map = entity_pair_map or get_default_entity_pair_map(sample_format=sample_format)
        self.color = color or False

    def __call__(
            self, sample: Union[NerExample, Dict[str, Any]] = None, entity_names: List[str] = None, entity_types: List[str] = None,
            color: bool = None
    ) -> str:
        enms, ets = None, None
        if sample is not None:
            if isinstance(sample, dict):
                enms, ets = sample.get('entity_names'), sample.get('entity_types')
                if enms is None or ets is None:
                    sic(sample, enms, ets)
                    raise NotImplementedError
                assert enms is not None and ets is not None
            else:
                if not isinstance(sample, NerReadableExample):
                    raise NotImplementedError(sample)
                enms, ets = sample.entity_names, sample.entity_types
        if entity_names is not None:
            enms = entity_names
        if entity_types is not None:
            ets = entity_types

        sep = self.entity_sep
        c = self.color if color is None else color
        if c:
            enms = [pl.i(enm, c='y') for enm in enms]
            ets = [pl.i(et, c='r') for et in ets]
            sep = pl.i(self.entity_sep, c='m')
        assert enms is not None
        if ets is None:
            ret = f'{sep} '.join(enms)
        else:
            assert len(enms) == len(ets)  # sanity check
            ret = f'{sep} '.join(self.entity_pair_map(enm, et) for enm, et in zip(enms, ets))
        pref, post = '[', ']'
        if c:
            pref, post = pl.i(pref, c='m'), pl.i(post, c='m')
        return f'{pref}{ret}{post}'


at = AnnotationsTemplator(sample_format='natural-pair-v2')
atc = AnnotationsTemplator(sample_format='natural-pair-v2', color=True)


def highlight_span_in_sentence(
        sentence: str = None, span: str = None, ec: EdgeCases = None, format_span: Callable[[str], str] = None,
        pref: str = None, post: str = None,
        allow_multi_occurrence: bool = False, span_index: int = None, span_index_super: int = None, color: bool = False,
        debug_for_test: bool = False
) -> str:
    # find all occurrences of entity name in sentence & highlight by enclosing in double braces
    # complicated logic needed to ensure only 1 match, e.g. 2 occurrences, TODO: ignore this problem for now
    #   Sentence: `Who directed the film The Matrix and its sequel, The Matrix Reloaded, featuring Morpheus?`
    #   Span: `The Matrix`
    sent, enm = sentence, span
    # if not (span in sentence or span.lower() in sentence.lower()):
    #     sic(sent, enm)
    assert span in sentence or span.lower() in sentence.lower()  # sanity check

    ms = patterns.find_match(text=sent, keyword=enm)
    n_ms = len(ms)
    if n_ms > 1:
        if allow_multi_occurrence:
            if ec:
                d_log = {'sentence': sent, 'entity_name': enm, '#matches': n_ms}

                pat = re.compile(re.escape(enm))  # highlight the entity name in the sentence
                sent_high = pat.sub(f'[{pl.i(enm, c="y")}]', sent)
                d_log['sentence-highlighted'] = sent_high
                msg = f'More than 1 match found w/ {pl.i(d_log)}'
                # sic(sentence, span, span_index)

                ec(msg=f'Edge Case: {msg}', kind='multiple-entity-span-match-highlights', args=d_log)
            if not debug_for_test:  # TODO: this is to filter out cases I will allow for multiple occurrences
                raise NotImplementedError  # not allow this from now on, TODO: still testing pipeline
        else:  # resolve multiple occurrences, potentially due to entity super-string
            if not (span_index is not None or span_index_super is not None):
                sic(sent, enm, ms)
            assert span_index is not None or span_index_super is not None  # sanity check at least 1 provided to resolve ambiguity
            # get that index; prioritize super-string index cos it covers all occurrences
            idx = span_index_super if span_index_super is not None else span_index
            ms = [ms[idx]]

    if n_ms == 0:  # must be due to casing
        ms = patterns.find_match(text=sent, keyword=enm, ignore_case=True)
        if len(ms) != 1:
            sic(sent, enm, ms)
        assert len(ms) == 1

    if not allow_multi_occurrence:
        if len(ms) != 1:
            sic(sent, enm, ms)
        assert len(ms) == 1  # sanity check

    fmt = None
    if format_span is not None:
        fmt = format_span
    elif pref is not None or post is not None:
        def fmt(x):
            x = pl.i(x, c='y') if color else x
            return f'{pref}{x}{post}'
    fmt = fmt or (lambda x: f'{{{{{x}}}}}')
    s, e = ms[0].span('keyword')  # get the span for the match group
    return sent[:s] + fmt(enm) + sent[e:]  # replace that span found


# match e.g. `entity_names-1`
_pattern_dup_y = re.compile(r'^(?P<key>entity_names)-(?P<idx>\d+)$')


class SampleDictPrettier:
    """
    Make sample logging dict more compact & readable by combining low-level fields and joining list elements
    """
    def __init__(self, sample_format: str = 'natural-pair-v2', color: bool = False):
        self.sample_format = sample_format
        if sample_format != 'natural-pair-v2':
            raise NotImplementedError
        self.at = AnnotationsTemplator(sample_format=sample_format, color=color)
        self.color = self.at.color

        # self.key_order = [
        #     'X', 'X_quote-dropped', 'entities', 'stripped-entities', 'Y',
        #     'missing_entity_names', 'entity_spans_quote_drop', 'Y_reordered', 'Y_not_allowed'
        # ]

        self.pattern_annotation_key = re.compile(r'^(?P<prefix>.*)(?P<kind>entity_names|entity_types)(?P<postfix>.*)$')

    def _get_key_pairs(self, d_sample: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """
        Find pairs of (entity names, entity types) keys w/ the same key description,
            e.g. `entity_names_corrected` and `entity_types_corrected`
        Group them together for `AnnotationsTemplator`
        """
        desc2types = defaultdict(list)
        for k in d_sample.keys():
            mch = self.pattern_annotation_key.match(k)
            if mch is not None:
                prefix, key, postfix = mch.group('prefix', 'kind', 'postfix')
                desc2types[(prefix, postfix)].append(key)
        # find the paired keys
        ret = []
        for (pref, post), types in desc2types.items():
            if len(types) == 2:
                assert len(set(types)) == 2  # sanity check both entity names and entity types are present
                # reconstruct the key pair, also get the new key name for the combined annotation
                k_enm = f'{pref}entity_names{post}'
                k_ets = f'{pref}entity_types{post}'
                out_key = f'{pref}Y{post}'
                ret.append((k_enm, k_ets, out_key))
        return ret

    def __call__(self, d_sample: Dict[str, Any] = None, as_str: bool = True, color: bool = None) -> Union[str, Dict[str, Any]]:
        if isinstance(d_sample, NerExample):
            if not isinstance(d_sample, NerReadableExample):
                raise NotImplementedError(d_sample)
            d_sample = asdict(d_sample)

        # def strify_list(key: str = None) -> Dict[str, Any]:
        #     lst = d_sample.get(key)
        #     if lst is not None and isinstance(lst, list):
        #         d_sample[key] = pl.nc(lst)  # merge to a string representation
        #     return d_sample

        def _merge_y(
                entity_names_key: str = 'entity_names', entity_types_key: str = 'entity_types', output_key: str = 'y'
        ) -> Dict[str, Any]:
            enms, ets = d_sample.get(entity_names_key), d_sample.get(entity_types_key)
            if enms is not None and ets is not None:
                del d_sample[entity_names_key], d_sample[entity_types_key]  # drop the keys
                assert len(enms) == len(ets)  # sanity check
                if isinstance(enms, tuple):
                    enms = list(enms)
                if isinstance(ets, tuple):
                    ets = list(ets)
                d_sample[output_key] = self.at(entity_names=enms, entity_types=ets, color=color)
            return d_sample

        def format_logprob(logprob_key: str = 'logprob') -> Dict[str, Any]:
            if logprob_key in d_sample:
                assert isinstance(d_sample[logprob_key], float)
                d_sample[logprob_key] = f'{d_sample[logprob_key]:.3e}'
            return d_sample

        d_sample = deepcopy(d_sample)
        d_sample['X'] = d_sample.pop('sentence')
        keys_processed = ['X']  # for ordering the keys
        if 'sentence_quote_dropped' in d_sample:
            d_sample['X_quote-dropped'] = d_sample.pop('sentence_quote_dropped')
            keys_processed.append('X_quote-dropped')

        # # d_sample = strify_list(key='entities')
        # # d_sample = strify_list(key='stripped-entities')
        # d_sample = _merge_y(entity_names_key='entity_names', entity_types_key='entity_types', output_key='Y')
        # # d_sample = strify_list(key='missing_entity_names')
        # # d_sample = strify_list(key='entity_spans_quote_drop')
        # d_sample = _merge_y(
        #     entity_names_key='entity_names_in_sentence', entity_types_key='entity_types_in_sentence', output_key='Y_in_sentence')
        # d_sample = _merge_y(entity_names_key='reordered_entity_names', entity_types_key='reordered_entity_types', output_key='Y_reordered')
        # d_sample = _merge_y(
        #     entity_names_key='entity_names_not_allowed', entity_types_key='entity_types_not_allowed', output_key='Y_not_allowed')
        # d_sample = _merge_y(
        #     entity_names_key='entity_names_after_type_filter', entity_types_key='entity_types_after_type_filter', output_key='Y_after_type_filter')
        #
        # # for en_k, et_k, k in [
        # #     ('entity_names_ori', 'entity_types_ori', 'Y_ori'),
        # #     ('entity_names_corrected', 'entity_types_corrected', 'Y_corrected'),
        # #     ('entity_names_corrected_initial', 'entity_types_corrected_initial', 'Y_corrected_initial'),
        # #     ('entity_names_corrected_final', 'entity_types_corrected_final', 'Y_corrected_final'),
        # # ]:
        # for post in ['ori', 'corrected', 'corrected_initial', 'corrected_final', 'LLM', 'BERT']:
        #     en_k, et_k, k = f'entity_names_{post}', f'entity_types_{post}', f'Y_{post}'
        #     d_sample = _merge_y(entity_names_key=en_k, entity_types_key=et_k, output_key=k)
        #
        # keys_not_processed = [k for k in d_sample.keys() if k not in self.key_order]
        #
        # keys_dup = []
        # # merge annotations for duplicated sample: same X, different Y
        # k2mch = {k: _pattern_dup_y.match(k) for k in keys_not_processed}
        # k2mch = {k: mch for k, mch in k2mch.items() if mch is not None}
        # for k_enm, mch in k2mch.items():
        #     idx = mch.group('idx')
        #     k_ets = f'entity_types-{idx}'
        #     assert k_ets in keys_not_processed
        #     k_out = f'Y-{idx}'
        #     d_sample = _merge_y(entity_names_key=k_enm, entity_types_key=k_ets, output_key=k_out)
        #     keys_not_processed.remove(k_enm)
        #     keys_not_processed.remove(k_ets)
        #     keys_dup.append(k_out)
        for k_enm, k_ets, k_out in self._get_key_pairs(d_sample=d_sample):
            assert k_enm in d_sample and k_ets in d_sample
            d_sample = _merge_y(entity_names_key=k_enm, entity_types_key=k_ets, output_key=k_out)
            keys_processed.append(k_out)

        for k in ['logprob', 'confidence']:
            if k in d_sample:
                d_sample = format_logprob(logprob_key=k)
                keys_processed.append(k)

        # if len(keys_not_processed) > 0 and set(keys_not_processed) > {'filename'}:
        #     d_log = dict(keys_not_processed=keys_not_processed, d_sample=d_sample)
        #     raise NotImplementedError(pl.fmt(d_log))
        # keys = self.key_order.copy() + keys_dup + keys_not_processed
        # d_sample = {k: d_sample[k] for k in keys if k in d_sample}  # order the keys

        # order the keys: move keys modified to the top
        keys_not_processed = [k for k in d_sample.keys() if k not in keys_processed]
        d_sample = {k: d_sample[k] for k in keys_processed + keys_not_processed}
        return pl.fmt(d_sample) if as_str else d_sample


# syntactic sugar
sdp = SampleDictPrettier()


def sdpc(d_sample: Dict[str, Any] = None, as_str: bool = True):
    # my own coloring and indentation instead of the default from `pl.fmt`
    d = sdp(d_sample=d_sample, as_str=False, color=True)
    return pl.i(d, indent=1) if as_str else d
