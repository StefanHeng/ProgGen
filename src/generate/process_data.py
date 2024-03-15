"""
Convert LLM completion in NL to structured NER data
"""

import os
import re
import json
import random
import logging
from os.path import join as os_join
from copy import deepcopy
from typing import List, Tuple, Dict, Union, Any
from dataclasses import dataclass
from collections import Counter, defaultdict

from stefutil import get_logger, pl, ca, stem, ordinal, Timer
from src.util import sconfig, span_pair_overlap, dataset_name2data_dir, sample_fmt2data_fmt, patterns, sample_check as check
from src.util.ner_example import NerExample, NerReadableExample, NerBioExample, detokenize
from src.util.sample_formats import (
    EntityPairTemplate, EntityPairEnclosePair, EntityPairEncloseBoth, TokenMapEnclose,
    get_default_entity_sep, get_default_entity_pair_map, get_default_token_map
)
from src.data_util import *
from src.data_util.prettier import EdgeCases, sdpc


__all__ = [
    'ExtractSingleSampleOutput', 'Np2SampleOutP2put', 'Np2Transform',
    'ReadableSamples', 'Completion2Samples', 'NerDatasetWriter'
]


_logger = get_logger('Completion2Ner-Data')


@dataclass
class ExtractSingleSampleOutput:
    success: bool = None
    sample: split.ExtractedSample = None
    sentence: str = None
    entities_raw: str = None
    no_entity: bool = None
    no_entity_dropped: bool = None


@dataclass
class ExtractSampleOutput:
    success: bool = None
    sample: split.NaturalSample = None
    sentence: str = None
    entities_raw: str = None


@dataclass
class Np2Triple:
    sentence: str = None
    entity_names: List[str] = None
    entity_types: List[str] = None


@dataclass
class Np2TripleOutput:
    success: bool = None
    sample: Np2Triple = None
    d_log: Dict[str, Any] = None  # in case of failure
    fail_reason: str = None


@dataclass
class Np2SampleOutP2put:
    success: bool = None
    sample: NerReadableExample = None
    # whether extraction failed or not, the sentence extraction part should always succeed; use it to be re-annotated via 2-stage Gen
    sentence: str = None
    d_log: Dict[str, Any] = None  # in case of failure
    fail_reason: str = None


class Np2Transform:
    """
    `natural-pair-v2` format is widely used, group all processing & edge case handling into one class.
        for both one-time and step-wise NER extraction
    """
    def __init__(
            self, dataset_name: str = 'conll2003',
            pattern_sentence: patterns.Patterns = None, pattern_entity: patterns.Patterns = None,
            entity_sep: str = None, entity_pair_map: EntityPairTemplate = None, allowed_entity_types: List[str] = None,
            lowercase_entity_type: bool = None, drop_puncs: Union[bool, str] = True, pattern_emph: patterns.Patterns = None,
            batched: bool = None, ec: EdgeCases = None, generate_type: str = None
    ):
        self.dataset_name = dataset_name
        self.pattern_extract_sent = pattern_sentence
        self.pattern_extract_entity = pattern_entity

        # For edge case: a single entity annotation enclosed in brackets, see `str2x_n_y`, e.g.
        #   `[dirty dancing (Title)]` in `[dirty dancing (Title)] [year (Year)]`
        self.pattern_entity_edge = re.compile(r'\[(?P<entity>[^\[\]]+?)]')

        self.entity_sep = entity_sep
        self.entity_pair_map = entity_pair_map

        self.allowed_entity_types = allowed_entity_types
        self.lowercase_entity_type = lowercase_entity_type
        self.drop_puncs = drop_puncs
        self.pattern_emph = pattern_emph

        self.batched = batched
        self.generate_type = generate_type

        self.ec = ec or EdgeCases()

    def str2x_n_y(self, sample: Union[str, split.GroupedStrSample], **kwargs) -> ExtractSampleOutput:
        """
        Initial splitting into (sentence, list of entity annotations) pairs
        """
        if isinstance(sample, str):
            sample_split = sample.strip().split('\n')
            if len(sample_split) != 2:
                sample_split = sample.strip().split('\n\n')  # for edge case, additional newline in between
            # if len(sample_split) != 2:  # for edge case, additional newline after sentence prefix, kinda ad-hoc for now, TODO
            #     assert '. Sentence: \n' in sample
            #     sample_split = sample.strip().split('\n')
            #     assert len(sample_split) == 3
            #     sample_split = [sample_split[0] + sample_split[1], sample_split[2]]
            #     # sic(sample_split)
            #     # raise NotImplementedError
            if not self.batched and len(sample_split) != 2:
                # for edge case: newline after Y prefix
                #   `Query: XXXXX\n
                #    Named Entities: \n XXXXX`
                lines = sample.strip().split('\n')
                n_ln = len(lines)
                if self.dataset_name == 'mit-movie' and n_ln == 3 and lines[-1] == 'Director':  # drop the last weird line
                    sample_split = [lines[0], lines[1]]
                else:
                    ln_y_pref = 'Named Entities: '
                    # if not (n_ln in [3, 4, 5, 6, 7] and lines[1] == ln_y_pref):
                    #     sic(sample)
                    # assert n_ln in [3, 4, 5, 6, 7] and lines[1] == ln_y_pref
                    assert lines[1] == ln_y_pref
                    x, y_pref, ys = lines[0], lines[1], lines[2:]

                    # LLM may even generate in other unexpected styles, e.g.
                    #   Markdown: `Named Entities:
                    #   - Frozen (Title)
                    #   - actors (Actor)
                    #   - trailer (Trailer)`
                    #   No separating comma: `Named Entities:
                    #   Tom Hanks (Actor)
                    #   stranded on a desert island (Plot)`
                    if all(y.startswith('- ') for y in ys):  # drop the bullet prefix
                        ys = [y[2:] for y in ys]
                    if len(ys) > 1 and all(not y.strip().endswith(self.entity_sep) for y in ys):
                        n_ys = len(ys)  # add the separating comma unless the last one
                        ys = [f'{y}{self.entity_sep}' if i + 1 < n_ys else y for i, y in enumerate(ys)]

                    y = ''.join(ys)  # effectively drop the newline in between
                    # sic(sample, x, y)
                    # raise NotImplementedError
                    sample_split = [x, f'{y_pref}{y}']
            if len(sample_split) != 2:
                d_log = dict(sample=sample)
                self.ec(msg=f'Edge case: failed to split sample w/ {pl.i(d_log)}', kind='sample-split-fail', failed=True)
                raise NotImplementedError
                # return ExtractSampleOutput(success=False)
            sent, entities_raw = sample_split
            sent, entities_raw = sent.strip(), entities_raw.strip()
            m_s = patterns.match_row(text=sent, pattern=self.pattern_extract_sent, desc='sentence', **kwargs)
            assert m_s is not None
            m_e = patterns.match_row(text=entities_raw, pattern=self.pattern_extract_entity, desc='entities', **kwargs)
            if m_e is None:
                d_log = dict(sentence=sent, entities=entities_raw)
                self.ec(msg=f'Edge case: failed to match entities w/ {sdpc(d_log)}', kind='entities-mismatch', failed=True)
                raise NotImplementedError
                # return ExtractSampleOutput(success=False)

            sent, entities = m_s.group('sent').strip(), m_e.group('entities').strip()
        else:
            assert isinstance(sample, split.GroupedStrSample)
            sent, entities, entities_raw = sample.sentence.strip(), sample.entities.strip(), sample.entities_raw

        # edge case: LLM didn't follow entity annotation formatting, e.g.
        #   Annotations enclosed in brackets instead of separated by comma: `Named Entities: [dirty dancing (Title)] [year (Year)]`
        #   An even weirder format, `Named Entities: [Preview clip (Trailer)], [new Batman (Title)]`, ignore for now
        # if '] [' in entities_raw or '], [' in entities_raw:
        if '] [' in entities_raw:
            # get both bracket ends
            idx_s = entities_raw.index(entities) - 1
            idx_e = idx_s + len(entities) + 1

            # sic(entities_raw)
            assert (entities_raw[idx_s], entities_raw[idx_e]) == ('[', ']')  # sanity check these brackets should be present
            entities_str = f'[{entities}]'
            assert entities_str == entities_raw[idx_s:idx_e + 1]

            # extract each entity annotation by pattern finding
            ms = list(self.pattern_entity_edge.finditer(entities_str))
            # sic(entities_str, ms)
            # raise NotImplementedError
            entities = [m.group('entity') for m in ms]
            d_log = dict(sentence=sent, entities=entities_raw, entities_extracted=entities)
            msg = f"Edge Case: LLM-generated entity annotation didn't follow formatting w/ {pl.i(d_log, indent=1)}"
            self.ec(msg=msg, kind='wrong-entity-annotation-format', args=d_log)
        else:
            # entities = entities.split(f'{self.entity_sep} ')
            # sic(entities)
            if entities == '':
                entities = []
            else:
                entities = [e.strip() for e in entities.split(self.entity_sep)]  # a more lenient split
            # if sent == 'What are the top-rated horror films currently showing and where can I buy tickets?':
            #     sic(entities)
            #     raise NotImplementedError

        merge = True
        # merge = False
        if merge and len(entities) > 1:
            # if a single span don't contain the entity type, assume the named entity itself contains `,`, i.e. `self.entity_sep`
            #   => try to merge the entity spans into one

            if any('(' not in e or ')' not in e for e in entities):
                entities_ = edit.merge_entities_on_separator(
                    entities=entities, entity_sep=self.entity_sep, entity_pair_map=self.entity_pair_map)

                if len(entities) != len(entities_):
                    d_log = {'entity-sep': self.entity_sep, 'original-entities': entities, 'merged-entities': entities_, **kwargs}
                    self.ec(msg=f'Edge case: merged entities on separator w/ {pl.i(d_log, indent=1)}', kind='merged-entities')
                entities = entities_

        sent = edit.sanitize_quotes(text=sent, ec=self.ec, sample_kind='sentence')
        entities = [edit.sanitize_quotes(text=enm, ec=self.ec, sample_kind='entity') for enm in entities]

        if entities_raw.endswith(','):  # edge case, see `self.pattern_extract_entity`
            d_log = dict(sentence=sent, entities=entities_raw)
            msg = f'Edge case: trailing comma in in-complete entity list w/ {sdpc(d_log)}'
            self.ec(msg=msg, kind='incomplete-entity-list')
        return ExtractSampleOutput(
            success=True, sample=split.NaturalSample(sentence=sent, entities=entities), entities_raw=entities_raw, sentence=sent)

    def y2entity_n_type(
            self, sample: split.NaturalSample, has_entity_type: bool = True, resolve_failure: bool = False, filename: str = None
    ) -> Np2TripleOutput:
        """
        Split each entity annotation into the entity span and the entity type
        """
        sent, entities = sample.sentence, sample.entities
        d_log = dict(sentence=sent, entities=entities)
        fail_ret = Np2TripleOutput(success=False, d_log=d_log)
        if patterns.is_none(sent):
            msg = f'Edge case: sentence is [{pl.i("None")}] for sample {sdpc(d_log)}'
            self.ec(msg=msg, kind='none-sentence', failed=True)
            return fail_ret

        entities_ = [e.strip() for e in entities]
        if entities != entities_:
            d_log['stripped-entities'] = entities_
            msg = f'Edge case: entity names contain leading/trailing spaces in sample {sdpc(d_log)}'
            entities = entities_
            self.ec(msg=msg, kind='entity-space')
        if any(patterns.is_none(e) for e in entities):
            self.ec(msg=d_log, kind='none-entity')
            entities = [e for e in entities if not patterns.is_none(e)]

        if has_entity_type:
            # for edge case, e.g.
            #   "Kenya's Automotive Industry Sees Growth Amidst Economic Challenges"
            #   Entity Names: [Kenya (location), Automotive Industry]
            if len(entities) >= 1 and any(self.entity_pair_map.entity_type_missing(e) for e in entities):
                kd = 'missing-entity-type'
                msg = f'Edge case: entity type missing in sample '
                idxs_miss = [i for i, e in enumerate(entities) if self.entity_pair_map.entity_type_missing(e)]
                entities_miss = [entities[i] for i in idxs_miss]

                if resolve_failure:  # drop such entity annotations
                    entities = [e for i, e in enumerate(entities) if i not in idxs_miss]
                    d_log['entities_after_drop'] = entities_
                    msg = f'{msg}, entities w/o entity type dropped'
                self.ec(msg=f'{msg} w/ {sdpc(d_log)}', kind=kd, args=dict(entities=tuple(entities_miss)), failed=True)

                if not resolve_failure:
                    fail_ret.fail_reason = kd
                    return fail_ret

            en, et = [], []
            for e in entities:
                try:
                    en_, et_ = self.entity_pair_map.decode(e)
                except ValueError:
                    if filename:
                        d_log['filename'] = filename
                    kd = 'entity-pair-decode'
                    self.ec(msg=f'Failed to decode entity pair [{pl.i(e)}] w/ {sdpc(d_log)}', kind=kd, failed=True)
                    fail_ret.fail_reason = kd
                    return fail_ret
                en.append(en_)
                et.append(et_)
            return Np2TripleOutput(success=True, sample=Np2Triple(sentence=sent, entity_names=en, entity_types=et))
        else:
            return Np2TripleOutput(success=True, sample=Np2Triple(sentence=sent, entity_names=entities))

    def sanitize(
            self, sentence: str = None, entity_names: List[str] = None, entity_types: List[str] = None,
            error_prefix: str = None, filename: str = None, resolve_failure: bool = False
    ) -> Np2SampleOutP2put:
        """
        Check for edge cases to make sure samples are well-formed

        :param sentence: the sentence in the sample
        :param entity_names: the named entities in the sample
        :param entity_types: the entity types corresponding to `entity_names`
        :param error_prefix: prefix to be added to edge case/error messages
        :param filename: the filename of the sample
        :param resolve_failure: If true, will try to remedy mal-formed samples, e.g.
            1> If any entity span is not found in the sentence, the corresponding entity annotation will be dropped
            2> If any entity spans overlap, entity annotations involved will be resolved by dropping based on heuristics
            3> If any entity type is not allowed, the corresponding entity annotation will be dropped

            intended for few-shot ICL baseline,
        """
        error_prefix = '' if error_prefix is None else f'{error_prefix} '

        # for edge case, e.g.
        #   "SAN FRANCISCO 2022-05-10"
        #   Entity Names: [SAN FRANCISCO (location), 2022-05-10 (None)]
        # drop the entity with type None; TODO, when LLM fails, does it mean uncertainty, and that we should discard this sample?
        sent, enms, ets = sentence, entity_names, entity_types
        d = dict(sentence=sent)
        d_log = dict(sentence=sent, entity_names=enms)  # by construction, len(en) == len(et)
        if filename is not None:
            d_log['filename'] = filename
        if ets is not None:
            d_log['entity_types'] = ets
        fail_ret = Np2SampleOutP2put(success=False, d_log=d_log, sentence=sent)

        has_none_entity = any(patterns.is_none(e) for e in enms)
        has_none_entity = has_none_entity or (ets is not None and any(patterns.is_none(e) for e in ets))
        if has_none_entity:
            self.ec(msg=d_log, kind='none-entity')
            et_remove, en_remove = [], []
            for en_, et_ in zip(enms, ets):
                if patterns.is_none(en_) or patterns.is_none(et_):
                    et_remove.append(et_)
                    en_remove.append(en_)
            for et_, en_ in zip(et_remove, en_remove):
                ets.remove(et_)
                enms.remove(en_)
        for i, en_ in enumerate(enms):
            # edge case: entity name annotation is enclosed in quotes, but the sentence doesn't have the quotes, e.g.
            #   Sentence: Can you tell me the year in which the animated movie Zootopia was released?
            #   Named Entities: ['Zootopia' (movie)]
            if ((en_[0] == "'" and en_[-1] == "'") or (en_[0] == '"' and en_[-1] == '"')) and \
                    en_ not in sent and en_[1:-1] in sent:
                msg = f'Edge case: entity name enclosed in quotes not found in sentence w/ {sdpc(d_log)}'
                self.ec(msg=msg, kind='entity-enclosed-quote')
                enms[i] = edit.drop_enclosing_quotes(en_)

        eis = check.entities_in_sentence(sentence=sent, entity_names=enms, ignore_case=True)
        if not eis.all_found:
            found, not_found = eis.entities_found, eis.entities_not_found
            d_log['missing_entity_names'] = not_found
            kd = 'entity-not-found'
            msg = f'{error_prefix}Entity names not found as exact match in sentence'

            if resolve_failure:  # drop these entities not found in the sentence
                if len(found) > 0:
                    enms, ets = zip(*[(en, et) for en, et in zip(enms, ets) if en in found])
                    enms, ets = list(enms), list(ets)
                else:
                    enms, ets = [], []
                d.update(entity_names=enms, entity_types=ets)
                d_log.update(entity_names_in_sentence=enms, entity_types_in_sentence=ets)
                msg = f'{msg}, entities not found dropped'
            self.ec(msg=f'{msg} w/ {sdpc(d_log)}', kind=kd, args=d_log, failed=True)

            if not resolve_failure:
                fail_ret.fail_reason = kd
                return fail_ret

        # drop enclosing quotes from entity names
        out = edit.drop_entities_enclosing_puncs(
            entity_names=enms, dataset_name=self.dataset_name, drop=self.drop_puncs, ec=self.ec, d_log=d_log)
        enms = out.entity_names
        if len(out.entity_names_modified) > 0:
            d_log['entity_spans_quote_drop'] = out.entity_names_modified

        if self.generate_type != 'baseline-both':
            sent_ = edit.drop_puncs_in_sentence(sentence=sent, pattern_emph=self.pattern_emph, ec=self.ec, d_log=d_log, filename=filename)
            if sent != sent_:  # sentence was changed
                if self.dataset_name not in ['conll2003-no-misc', 'wiki-gold-no-misc', 'mit-movie', 'mit-restaurant']:
                    from stefutil import sic
                    # TODO: check that no enclosing quote in original dataset sentences
                    sic(sent, sent_)
                    raise NotImplementedError(filename)
                d['sentence'] = fail_ret.sentence = sent = sent_
        # sanity check entities still in sentence
        assert check.entities_in_sentence(sentence=sent, entity_names=enms, ignore_case=True).all_found

        def log_overlap_fail(args_: Dict[str, Any] = None):
            kd_ = 'entity-overlap'
            msg_ = f'{error_prefix}Entity names overlapping in sentence w/ {sdpc(d_log)}'
            if resolve_failure:
                kd_ = f'{kd_}-drop'
                msg_ = f'{msg_}, the longer-overlapping entity span dropped'
            self.ec(msg=msg_, kind=kd_, args=args_, failed=True)

            if not resolve_failure:
                fail_ret.fail_reason = kd_
                return fail_ret

        # a simple overlap case: each entity name appears only once in sentence
        ovl_out = check.entities_overlapping(sentence=sent, entity_names=enms, ignore_case=True, search_in_order=False)
        overlap, ms = ovl_out.overlap, ovl_out.matches
        # if overlap and len(ms) == len(enms):  # each match found exactly once
        if overlap:
            args = None
            if resolve_failure:
                # need to drop entity spans to make sure no overlap; a good heuristic is dropping ranked by logprobs, but complicated, TODO
                #   for now, just drop the longer span
                # sanity check all entities are unique & only 1 span in sentence 
                assert len(set(enms)) == len(enms)
                entity_spans = ovl_out.entity_spans
                if not all(len(spans) == 1 for spans in entity_spans):
                    from stefutil import sic
                    sic(sent, enms, ets)
                    sic(entity_spans)
                assert all(len(spans) == 1 for spans in entity_spans)
                entity_spans = [spans[0] for spans in entity_spans]
                # get all possible unordered pairs & whether they overlap
                n_enms = len(enms)
                overlap_pairs = [
                    (i1, i2) for i1 in range(n_enms-1) for i2 in range(i1+1, n_enms)
                    if span_pair_overlap(span1=entity_spans[i1], span2=entity_spans[i2])
                ]
                n_ovl = len(overlap_pairs)
                assert n_ovl > 0  # sanity check

                if n_ovl == 1:  # a simple case: only 2 entities overlap w/ each other
                    i1, i2 = overlap_pairs[0]
                    enm1, enm2 = enms[i1], enms[i2]
                    n1, n2 = len(enm1), len(enm2)
                    assert n1 != n2  # sanity check
                    idx_enm_longer = i1 if n1 > n2 else i2

                    args = dict(overlapping_entity_spans=enms)
                    # drop the longer entity span
                    d_log.update(dropped_entity_name=enms[idx_enm_longer])
                    enms, ets = deepcopy(enms), deepcopy(ets)
                    enms.pop(idx_enm_longer)
                    ets.pop(idx_enm_longer)
                    args['resolved_entity_spans'] = enms
                    # d_log['dropped_entity_name'] = enms[idx_enm_longer]
                    d_log.update(entity_name_after_overlap_drop=enms, entity_type_after_overlap_drop=ets)
                    d.update(entity_names=enms, entity_types=ets)
                else:
                    from stefutil import sic
                    sic(sent, enms, ets)
                    raise NotImplementedError
            log_overlap_fail(args_=args)

        # edge case: the same entity name appears twice, but in entity list, only one of that name is present
        # e.g. Note `NASA`
        #   "NASA administrator Jim Bridenstine praised the mission as a significant achievement for both SpaceX and NASA,
        #       showcasing American excellence in space exploration."
        #   Entity Names: [NASA (organization), SpaceX (organization), Jim Bridenstine (person)]
        ic = True
        # edge case: 2 entity names differ in case only,
        #   e.g.  SINGAPORE 2022-09-15: Air pollution reaches hazardous levels in Singapore as forest fires continue to
        #       ravage neighboring countries.
        if edit.entities_differ_in_case_only(entity_names=enms, sentence=sent, ec=self.ec):
            ic = False
            msg = f'Edge case: entity names differ in case only w/ {sdpc(d_log)}'
            self.ec(msg=msg, kind='entity-case-diff')
        c = check.get_non_overlapping_keyword_counts(sentence=sent, keywords=enms, ignore_case=ic)
        if any(c[enm] == 0 for enm in enms):  # some entity names not found in sentence
            miss_ens = [enm for enm, count in c.items() if count == 0]
            d_log['missing_entity_names'] = miss_ens
            msg = f'{error_prefix}Entity names not found in sentence for overlapping w/ {sdpc(d_log)}'
            kd = 'entity-not-found-overlap'
            self.ec(msg=msg, kind=kd, failed=True)
            fail_ret.fail_reason = kd
            return fail_ret

        if any(c[enm] > 1 for enm in enms) and len(enms) == len(set(enms)):  # entity name list are all distinct, but some appear multiple times
            # sic(enms, ets)
            lim = 2**10 if self.generate_type == 'baseline-both' else 4  # effectively no limit
            # too_much_entities = (self.dataset_name == 'conll2003-no-misc' and self.generate_type == 'baseline-both')
            # sic(sent)
            out = edit.duplicate_multi_occurring_entities(
                entity_names=enms, entity_types=ets, entity_name2count=c, d_log=d_log, allow_limit=lim)
            enms, ets = out.entity_names, out.entity_types
            # conll-2003 test set has a sentence w/ 29 entities, re-ordering will take huge time

            msg = f'{error_prefix}Entity name appears multiple times but only one annotation w/ {sdpc(d_log)}'
            self.ec(msg=msg, kind='multi-entity-annot-missing')
        c_ = Counter(enms)  # counts that may be overlapping
        enms_dup = [enm for enm, cnt in c_.items() if cnt > 1]
        if len(enms_dup) > 0:
            # it must be the case that there are at least 2 occurrences in the sentence
            if not all(len(patterns.find_match(text=sent, keyword=enm, ignore_case=ic, strict=False)) > 1 for enm in enms_dup):
                msg = f'Edge case: the same entity name annotated multiple times, but appears only once in sentence w/ {sdpc(d_log)}'
                kd = 'too-many-entity-annot'
                self.ec(msg=msg, kind=kd, failed=True)
                fail_ret.fail_reason = kd
                return fail_ret

        # make sure the entity ordering is correct, *then* check for more sophisticated overlaps
        ic = True
        if edit.entities_differ_in_case_only(entity_names=enms, sentence=sent, ec=self.ec):
            # for edge case, e.g.
            #   sentence: "Who directed the thriller movie with the song 'Thriller' by Michael Jackson in its soundtrack?"
            #   entity_names: ['thriller', 'Thriller', 'Michael Jackson', 'directed']
            ic = False
        # intended for huge #entities in a sentence for CoNLL-2003 test set, use a smaller limit to speed up
        ordering_insert_limit = 7 if self.generate_type == 'baseline-both' else None
        out = edit.reorder_entities(
            sentence=sent, entity_names=enms, entity_types=ets, ignore_case=ic, insert_if_more=ordering_insert_limit)
        if out.reordered:
            # d_log.update(reordered_entity_names=out.entity_names, reordered_entity_types=out.entity_types)
            d_log['reordered_entity_names'] = out.entity_names
            if ets is not None:
                d_log['reordered_entity_types'] = out.entity_types
            self.ec(msg=f'{error_prefix}Reordered entities w/ {sdpc(d_log)}', kind='entity-reorder')
            d['entity_names'], d['entity_types'] = enms, ets = out.entity_names, out.entity_types

        if check.entities_overlapping(sentence=sent, entity_names=enms, ignore_case=True).overlap:
            assert out.reordered  # otherwise, code shouldn't reach here
            if resolve_failure:
                raise NotImplementedError
            log_overlap_fail()

        allowed = self.allowed_entity_types
        if self.lowercase_entity_type:  # revert to the original casing
            lower_aet = [aet.lower() for aet in allowed]
            laet2aet = dict(zip(lower_aet, allowed))
            for i, et_ in enumerate(ets):
                if et_ in lower_aet:
                    ets[i] = laet2aet[et_]
        if ets is not None and allowed is not None and not all(et_ in allowed for et_ in ets):
            ens_na, ets_na = [], []
            for en_, et_ in zip(enms, ets):
                if et_ not in allowed:
                    ens_na.append(en_)
                    ets_na.append(et_)
            d_log.update(entity_names_not_allowed=ens_na, entity_types_not_allowed=ets_na)
            kd = 'not-allowed-entity-type'
            msg = f'{error_prefix}Entity types not in allowed types'

            if resolve_failure:  # drop these unseen entity annotations
                if len(enms) == len(ens_na):  # all entity annotations of unseen entity types
                    enms, ets = [], []
                else:  # at least 1 annotation can be kept
                    enms, ets = zip(*[(en, et) for en, et in zip(enms, ets) if et in allowed])
                    enms, ets = list(enms), list(ets)
                d.update(entity_names=enms, entity_types=ets)
                d_log.update(entity_names_after_type_filter=enms, entity_types_after_type_filter=ets)
                msg = f'{msg}, entities w/ unseen entity types dropped'
            self.ec(msg=f'{msg} w/ {sdpc(d_log)}', kind=kd, args=d_log, failed=True)

            if not resolve_failure:
                fail_ret.fail_reason = kd
                return fail_ret
        d.update(entity_names=tuple(enms), entity_types=tuple(ets) if ets is not None else None)
        return Np2SampleOutP2put(success=True, sample=NerReadableExample(**d), sentence=sent)


@dataclass
class SplitSampleOutput:
    """
    For an individual sample in a LLM completion
    """
    sample: str = None
    sample_raw: str = None
    span: Tuple[int, int] = None


@dataclass
class ReadableSamples:
    n_sample_found: int = None
    samples: List[NerExample] = None
    sentences: List[str] = None  # NER sample extraction may fail, but all sentences generated should all be successfully extracted
    logprobs: List[float] = None  # logprob corresponding to each sample, by average logprob on entire sample/entity annotations
    # logprob corresponding to each entity annotation
    entity_logprobs: List[List[float]] = None
    edge_cases: List[dict] = None
    failed_sentences: Dict[str, List[str]] = None
    num_no_entity: int = None
    num_dropped: int = None


class Completion2Samples:
    def __init__(
            self, dataset_name: str = 'conll2003',
            token_sep: str = ',', entity_sep: str = None, entity_pair_map: EntityPairTemplate = None,
            sample_format: str = 'natural-pair', has_enum_prefix: bool = False,
            allowed_entity_types: List[str] = None, lowercase_entity_type: bool = False,
            drop_samples_wo_entities: Union[bool, float] = False, logger: logging.Logger = None, ec: EdgeCases = None
    ):
        """
        :param dataset_name: name of dataset, e.g. `conll2003`
        :param token_sep: See `demo2prompt::PromptConstructor`
        :param entity_sep: See `demo2prompt::PromptConstructor`
        :param entity_pair_map: See `demo2prompt::PromptConstructor`
        :param sample_format: See `demo2prompt::PromptConstructor::prompt_format`
        :param has_enum_prefix: for extracting from output of web-based ChatGPT,
            relevant only if `data_format` in [`natural-pair`, `natural-inline`]
        :param allowed_entity_types: If given, only samples with entity types in this list are kept
        :param lowercase_entity_type: If true, entity types were lowercased during generation
        :param drop_samples_wo_entities: If a sample doesn't have any entities, drop it
            If a float is given, drop samples w/ probability `drop_samples_wo_entities`
        :param logger: logger for writing to file
        :param ec: Stores edge cases encountered during processing
        """
        self.dataset_name = dataset_name
        self.has_enum_prefix = has_enum_prefix

        self.token_sep = token_sep
        self.entity_sep = entity_sep or get_default_entity_sep(sample_format=sample_format)
        self.entity_pair_map = entity_pair_map or get_default_entity_pair_map(sample_format=sample_format)
        self.token_map = get_default_token_map(sample_format=sample_format)

        self.sample_format = sample_format

        ca(sample_format=sample_format)
        d_dset = sconfig(f'datasets.{dataset_name}')
        pref_x, pref_y = d_dset['x-decoded-name'], d_dset['y-decoded-name']
        pref_x, pref_y = patterns.options2re_options(options=pref_x), patterns.options2re_options(options=pref_y)
        if sample_format == 'natural-pair':
            if has_enum_prefix:  # tend to be web-based ChatGPT outputs
                # check if LLM intended to generate a sample in this line
                # e.g. `15. whatever`
                self.pattern_check = re.compile(r'^(?P<idx>\d+)\. (?P<sent>.*)$')

                # make sure LLM generated according to prompt & extract entities
                # e.g. `15. Bla bla bla. Entities: ent1: type1, ent2: type2, ent3: type3.`
                self.pattern_extract = re.compile(r'^(\d+)\. (?P<sent>.*) Entities: (?P<entities>.*)\.$')
            else:  # tend to be completion from API calls, this follows formatting of prompt more strictly
                self.pattern_check = re.compile(r'^sentence:(.*)\nentities:(.*)$')

                # sentences and entities are both on new lines
                self.pattern_extract_sent = re.compile(r'^sentence: (?P<sent>.*)$')

                # edge case: missing period, e.g.
                # sentence: Feeding infected meat and bone meal to sheep could not be ruled out, it said.
                # entities: -
                self.pattern_extract_entity = [re.compile(r'^entities: (?P<entities>.*)\.$'), re.compile(r'^entities: (?P<entities>.*)$')]
        elif sample_format == 'natural-pair-v2':
            pat_hspace = r'([ \t]*)'
            if has_enum_prefix:
                # make sure if sample has index, double quotes and template keyword `Entity Names` appears
                self.pattern_search = [
                    # edge case:
                    #   1. double quote missing from sentence
                    #   2. certain number of spaces before `Entity Names`
                    # re.compile(r'(?P<idx>\d+)\. (")?(?P<sent>.*)(")?\n(\s)*?(Entity Names|Entities|Named Entities): (?P<entities>.*)\n', re.IGNORECASE)
                    # after GPT3.5 change, add Sentence Prefix
                    re.compile(rf'(?P<idx>\d+)\.{pat_hspace}({pref_x}:)?( )?(")?(?P<sent>.*)(")?\n(\n)?{pat_hspace}({pref_y}):{pat_hspace}(?P<entities>.+)\n', re.IGNORECASE)
                ]
                self.pattern_extract_sent = [
                    re.compile(rf'^(?P<idx>\d+)\.{pat_hspace}({pref_x}:)?{pat_hspace}"(?P<sent>.*?)"$', re.IGNORECASE),
                    re.compile(rf'^(?P<idx>\d+)\.{pat_hspace}({pref_x}:)?{pat_hspace}(?P<sent>.*)$', re.IGNORECASE)
                ]

                # edge case: entire sample is empty, and shows `None`, e.g.
                # 2. Elon Musk announces plans for a new Tesla factory in Germany.
                # Entity Names: [Elon Musk (person), Tesla (organization), Germany (location)]
                # 3. None
                # ...
                # 4. "None."
                self.pattern_empty_edge = re.compile(r'(?P<idx>\d+)\. "?None\.?"?\n', re.IGNORECASE)
            else:
                # edge case:
                #   1. trailing whitespace after sentence double quotes
                #   2. double quote missing from sentence
                #   3. additional newline in between, or no newline at all, for the latter case, ignore for now
                self.pattern_search = [
                    re.compile(rf'({pref_x}( (?P<idx>\d+))?:)?{pat_hspace}"?(?P<sent>.*?)"?( )?\n(\n)?{pat_hspace}({pref_y}):{pat_hspace}(?P<entities>.+)\n', re.IGNORECASE),
                    # a rare edge case where enum prefix is after sentence prefix, e.g.
                    #   `Query 2: "What year did Jessica Chastain star in a mystery movie"`
                    #       ignore checking the enum index for this case
                    # edge case: 1. `Entities` instead of `Entity Names`; 2. missing space after entity prefix
                    re.compile(rf'"(?P<sent>.*)"\n({pref_y}): (?P<entities>.*)\n', re.IGNORECASE)
                ]
                self.pattern_extract_sent = [
                    re.compile(rf'^({pref_x}( (?P<idx>\d+))?:)?( )?"(?P<sent>.*)"$', re.IGNORECASE),
                    re.compile(rf'^({pref_x}( (?P<idx>\d+))?:)?( )?(?P<sent>.*)$', re.IGNORECASE)
                ]

            self.pattern_extract_entity = [
                # edge case, trailing period; white spaces inside brackets
                re.compile(rf'^({pref_y}):( )?\[(?P<entities>.*)](\.)?$', re.IGNORECASE),
                re.compile(rf'^({pref_y}): \[( )?(?P<entities>.*)( )?](\.)?$', re.IGNORECASE),
                # edge case: incomplete entity list, e.g.
                #   `Entity Names: [Europa Clipper (spacecraft), `
                #   `Named Entities: [sci-fi (Genre), 2010s (Year), top-rated (Review)`
                re.compile(rf'^({pref_y}): \[(?P<entities>.*),?$', re.IGNORECASE),
                re.compile(rf'^({pref_y}): \[(?P<entities>.*)](\.)?$', re.IGNORECASE),
                re.compile(rf'^({pref_y}): \[(?P<entities>.*)](\.)?$', re.IGNORECASE),
                # edge case, missing brackets
                re.compile(rf'^({pref_y}): (?P<entities>.*)$', re.IGNORECASE)
            ]
        elif sample_format == 'natural-inline':
            if has_enum_prefix:
                # e.g. `15. sentence: whatever`
                self.pattern_check = self.pattern_extract = re.compile(r'^(?P<idx>\d+)\. (?P<sent>.*)$')
            else:
                self.pattern_check = self.pattern_extract = re.compile(r'^sentence: (?P<sent>.*)$')

            if not isinstance(self.entity_pair_map, EntityPairEnclosePair):
                raise NotImplementedError
        elif sample_format == 'natural-inline-v2':
            if has_enum_prefix:
                self.pattern_search = [
                    re.compile(r'(?P<idx>\d+)\. "(?P<sent>.*)"\nAnnotated Sentence: "(?P<entities>.*)"\n', re.IGNORECASE),
                    # edge case: double quote missing from sentence
                    re.compile(r'(?P<idx>\d+)\. (?P<sent>.*)\nAnnotated Sentence: "(?P<entities>.*)"\n', re.IGNORECASE)
                ]
                self.pattern_extract_sent = [
                    re.compile(r'^(?P<idx>\d+)\. "(?P<sent>.*)"$', re.IGNORECASE),
                    re.compile(r'^(?P<idx>\d+)\. (?P<sent>.*)$', re.IGNORECASE)
                ]
            else:
                # sentence in double quotes, followed by `Annotated Sentence` prefix
                self.pattern_search = re.compile(r'"(?P<sent>.*)"\nAnnotated Sentence: "(?P<sent_annot>.*)"\n', re.IGNORECASE)
                self.pattern_extract_sent = re.compile(r'^"(?P<sent>.*)"$', re.IGNORECASE)
            self.pattern_extract_annot = re.compile(r'^Annotated Sentence: "(?P<sent_annot>.*)"$', re.IGNORECASE)
        elif sample_format == 'bio-list':
            self.pattern_check = re.compile(r'^tokens:(.*)\nentity tags:(.*)$', re.IGNORECASE)
            self.pattern_extract_toks = re.compile(r'^tokens: (?P<tokens>.*)$', re.IGNORECASE)
            self.pattern_extract_tags = [
                re.compile(r'^entity tags: (?P<tags>.*)$', re.IGNORECASE),
                re.compile(r'^NER tags: (?P<tags>.*)$', re.IGNORECASE)
            ]

            # edge case: 1) tokens and tags are capitalized, 2) missing newline character
            self.patterns_check_edge = [
                # re.compile(r'^Tokens:(.*)\nEntity tags:(.*)$'),
                re.compile(r'^tokens:(.*)\nNER tags:(.*)$'),
                re.compile(r'^tokens:(?P<tokens>.*)entity tags:(?P<tags>.*)$')
            ]

            self.pattern_token = re.compile(r'^\[(?P<token>.+)]$')
        elif sample_format == 'bio-list-v2':
            if has_enum_prefix:
                # make sure if sample has index, template keywords of `Sentence:`, `Tokens`, `NER Tags` in that order
                # being lenient here, match any number of spaces & newlines
                self.pattern_search = [
                    re.compile(r'(?P<idx>\d+)\. Sentence: (.*?)(\n)(\s+)?Tokens: (.*?)(\s+)?NER Tags: (.*?)\n', re.IGNORECASE),
                    # # edge case: `sentence` prefix missing
                    re.compile(r'(?P<idx>\d+)\. (.*?)(\n)(\s+)?Tokens: (.*?)(\s+)?NER Tags: (.*?)\n', re.IGNORECASE)
                ]

                self.pattern_extract_sent = [
                    re.compile(r'^(?P<idx>\d+)\. Sentence: "(?P<sent>.*)"$', re.IGNORECASE),
                    re.compile(r'^(?P<idx>\d+)\. "(?P<sent>.*)"$', re.IGNORECASE)
                ]
                # edge case, starting whitespaces
                # e.g. 1. Sentence: "The United Nations has declared a humanitarian crisis in the war-torn country."
                #       Tokens: [The, United, Nations, has, declared, a, humanitarian, crisis, in, the, war-torn, country, .]
                #       NER tags: [B-organization, I-organization, I-organization, O, O, O, O, O, O, O, O, O, O]
                self.pattern_extract_toks = re.compile(r'^(\s+)?Tokens: \[(?P<tokens>.*)]$', re.IGNORECASE)
                self.pattern_extract_tags = re.compile(r'^(\s+)?NER Tags: \[(?P<tags>.*)]$', re.IGNORECASE)
            else:
                self.pattern_search = re.compile(r'Sentence: (.*?)(\n)(\s+)?Tokens: (.*?)(\n)(\s+)?NER Tags: (.*?)\n', re.IGNORECASE)
                self.pattern_extract_sent = re.compile(r'^Sentence: "(?P<sent>.*)"$', re.IGNORECASE)
                self.pattern_extract_toks = re.compile(r'^Tokens: \[(?P<tokens>.*)]$', re.IGNORECASE)
                self.pattern_extract_tags = re.compile(r'^NER Tags: \[(?P<tags>.*)]$', re.IGNORECASE)
        else:
            assert sample_format == 'bio-line'  # TODO: WIP

            # e.g. `<token>, <tag in BIO format>`
            self.pattern_extract_pair = re.compile(r'^(?P<token>.+), (?P<tag>O|B-.+|I-.+)$')
        self.pattern_emph = [re.compile(rf'\[(?P<et>[^]]{{1,45}})]', re.IGNORECASE)]

        self.allowed_entity_types = allowed_entity_types
        if allowed_entity_types and self.sample_format != 'natural-pair-v2':
            raise NotImplementedError
        self.lowercase_entity_type = lowercase_entity_type or False
        if drop_samples_wo_entities:
            if self.sample_format not in ['natural-pair-v2']:
                raise NotImplementedError
        self.drop_samples_wo_entities = drop_samples_wo_entities
        self.drop_with_prob, self.drop_prob = False, None
        if isinstance(drop_samples_wo_entities, float):
            self.drop_with_prob = True
            self.drop_prob = drop_samples_wo_entities

        self.logger = logger or _logger  # allow passing in custom logger for writing to file
        self.ec = ec or EdgeCases(logger=self.logger)
        if self.sample_format == 'natural-pair-v2':
            self.nt = Np2Transform(
                dataset_name=self.dataset_name, pattern_sentence=self.pattern_extract_sent, pattern_entity=self.pattern_extract_entity,
                entity_sep=self.entity_sep, entity_pair_map=self.entity_pair_map,
                allowed_entity_types=self.allowed_entity_types, lowercase_entity_type=self.lowercase_entity_type, drop_puncs='both',
                pattern_emph=self.pattern_emph, ec=self.ec, batched=True
            )

    def split_completion(self, completion: str = None, filename: str = None) -> List[SplitSampleOutput]:
        if self.has_enum_prefix or self.sample_format in ['natural-pair-v2', 'natural-inline-v2', 'bio-list-v2']:
            # sanity check
            assert self.sample_format in ['natural-pair', 'natural-pair-v2', 'natural-inline', 'natural-inline-v2', 'bio-list-v2']

            if self.sample_format in ['natural-pair', 'natural-inline']:
                idx_prev = 0
                for ln in completion.split('\n'):
                    m = self.pattern_check.match(ln)
                    if m is not None:  # this row is a sample
                        idx = int(m.group('idx'))
                        # assert idx_prev is None or idx == idx_prev + 1  # sanity check idx is continuous
                        assert idx == idx_prev + 1
                        idx_prev = idx
                        yield ln
            else:  # `natural-pair-v2`, `natural-inline-v2`, `bio-list-v2`
                completion = f'{completion}\n'  # add trailing newline to match tag entries for last sample
                out = split.split_samples(
                    completion=completion, pattern=self.pattern_search, has_enum_prefix=self.has_enum_prefix, ec=self.ec, silent=True)
                if not out.success:
                    d_log = dict(completion=completion, filename=filename)
                    self.ec(msg=f'Failed to find samples in completion:\n{pl.i(d_log)}', kind='invalid-sample-format', failed=True)
                    return []

                for sample, mch in zip(out.samples, out.matches):
                    sample_raw = sample
                    sample = '\n'.join([ln for ln in sample.split('\n') if ln.strip() != ''])  # drop additional newline chars
                    s, e = span = mch.span()
                    assert sample_raw == completion[s:e]  # sanity check
                    yield SplitSampleOutput(sample=sample, sample_raw=sample_raw, span=span)
        else:
            samples = [s for s in completion.split('\n\n') if s.strip() != '']
            # edge case, samples are separated by a single `\n`
            if self.sample_format == 'natural-inline' and len(samples) == 1:
                samples = [s for s in completion.split('\n') if s.strip() != '']

            if self.sample_format == 'bio-list' and len(samples) >= 2:
                # edge case: 2 newline characters between tokens and tags
                i = 0
                n_sample = len(samples)
                samples_ = []
                while i+1 < n_sample:
                    ln = samples[i]
                    if self.pattern_extract_toks.match(ln) is not None and self.pattern_extract_tags.match(samples[i + 1]) is not None:
                        self.ec(msg='Edge case: 2 newline characters between tokens line and tags line', kind='newline')
                        samples_.append(ln + '\n' + samples[i+1])
                        i += 2
                    else:
                        samples_.append(ln)
                        i += 1
                samples = samples_

            for i, ln in enumerate(samples):
                # for edge case where samples are separated by 3 `\n`s, in such case, samples starts w/ `\n`
                if ln[0] == '\n':
                    ln = ln.lstrip()
                    self.ec(msg=f'Edge case: sample starts w/ newline:\n[{pl.i(ln)}]', kind='newline')

                if self.sample_format != 'bio-line':
                    m = self.pattern_check.match(ln)
                    if self.sample_format == 'bio-list':
                        i = 0
                        patterns = self.patterns_check_edge
                        while m is None and i < len(patterns):
                            m = patterns[i].match(ln)
                            i += 1

                    # last generated sample may be broken, e.g. doesn't have `entities`
                    if m is None and i + 1 == len(samples):
                        self.ec(msg=f'Last generated sample may be broken:\n{pl.i(ln)}', kind='last-sample-broken')
                        continue
                    assert m is not None
                # for `line-bio`, there's not really a format to check...
                yield ln

    @staticmethod
    def fix_wrong_bio(tags: List[str] = None, tokens: List[str] = None) -> Tuple[List[str], List[Dict[str, str]]]:
        # 2 edge cases encountered, tag doesn't follow BIO format
        # e.g. 1>
        #   tokens: [The, Federal, Bureau, of, Investigation, arrested, several, individuals, for, their, involvement, in, an,
        #   international, drug, ring, .]
        #   tags: [O, B-organization, I-organization, O, I-organization, O, O, O, O, O, O, O, O, O, O, O, O]
        # e.g. 2>
        #   tokens: [President, Biden, signed, an, executive, order, to, promote, voting, rights, and, increase, access, to, the, ballot, .]
        #   tags: [B-title, I-person, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O]
        curr_type = None
        tags_, fixed = [], []

        for i, t in enumerate(tags):
            if t == 'O':
                curr_type = None
            else:
                prefix, lb = t.split('-')
                if prefix == 'B':
                    curr_type = lb
                else:
                    assert prefix == 'I'
                    if lb != curr_type:
                        if curr_type is None:
                            # t_ = t  # TODO: this doesn't fix the edge case sample exactly...
                            assert tags[i-1] == 'O' and tokens[i-1] == 'of'
                            tags_[i-1], _t = f'I-{lb}', tags[i-1]
                            fixed.append({'index': i, 'wrong': _t, 'fixed': tags_[i-1]})
                        else:  # TODO: hard to figure out the correct label...
                            t_ = f'B-{lb}'
                            fixed.append({'index': i, 'wrong': t, 'fixed': t_})
                            t = t_
                        curr_type = lb
            tags_.append(t)
        return tags_, fixed

    def extract_single_sample(self, sample: str) -> ExtractSingleSampleOutput:
        _match = patterns.match_row
        if self.sample_format in ['natural-pair', 'natural-pair-v2']:
            entities_raw = None
            if self.sample_format == 'natural-pair' and self.has_enum_prefix:
                m = _match(text=sample, pattern=self.pattern_extract, desc='sample', has_enum_prefix=self.has_enum_prefix)
                sent, entities = m.group('sent').strip(), m.group('entities').split(f'{self.entity_sep} ')
                sent: str
            else:  # `natural-pair` without enum, `natural-pair-v2`
                out = self.nt.str2x_n_y(sample=sample, has_enum_prefix=self.has_enum_prefix)
                if not out.success:
                    sent = out.sentence
                    assert sent is not None
                    return ExtractSingleSampleOutput(success=False, sentence=sent)

                sent, entities = out.sample.sentence, out.sample.entities
                entities_raw = out.entities_raw
            # edge case: sample had no entities at all
            # e.g.
            #   sentence: The stock market reached an all-time high, surpassing 30,000 points.
            #   entities: None.
            if len(entities) == 0 or (len(entities) == 1 and patterns.is_none(entities[0])):
                d_log = dict(sentence=sent, entities=entities)
                entities = []

                drop = False
                if self.drop_samples_wo_entities:
                    if self.drop_with_prob:
                        drop = random.random() < self.drop_prob
                    else:
                        drop = True

                msg = f'Sample w/ empty entity list: {sdpc(d_log)}'
                if drop:
                    msg = f'{msg} and dropped'
                self.ec(msg=msg, kind='empty-entity-list')
                if drop:
                    raise NotImplementedError
                    # return ExtractSingleSampleOutput(success=False, no_entity=True, no_entity_dropped=True)
            return ExtractSingleSampleOutput(
                success=True, sample=split.NaturalSample(sentence=sent, entities=entities), no_entity=entities == [],
                sentence=sent, entities_raw=entities_raw)
        elif self.sample_format == 'natural-inline':
            # same code regardless of `has_enum_prefix`
            m = _match(text=sample, pattern=self.pattern_extract, desc='annotated sentence')
            sent: str = m.group('sent').strip()
            self.entity_pair_map: EntityPairEnclosePair
            pattern_pair = self.entity_pair_map.pattern_decode_search

            entities = patterns.find_non_overlap_matches(pattern=pattern_pair, text=sent)

            # find all non-overlapping matches of the entity pair template, replace with the `name` group in the template
            # return NaturalSample(sentence=re.sub(pattern_pair, r'\g<name>', sent), entities=entities)
            raise NotImplementedError
        elif self.sample_format == 'natural-inline-v2':
            sent, annot = sample.split('\n')
            m_s = _match(text=sent, pattern=self.pattern_extract_sent, desc='sentence')
            m_a = _match(text=annot, pattern=self.pattern_extract_annot, desc='annotated sentence')

            sent, annot = m_s.group('sent').strip(), m_a.group('sent_annot').strip()
            self.entity_pair_map: EntityPairEncloseBoth
            pattern_pair = self.entity_pair_map.pattern_decode_search

            ms: List[re.Match] = patterns.find_non_overlap_matches(
                pattern=pattern_pair, text=annot, return_matches=True, accept_no_match=True)
            entities = [m.group() for m in ms]
            # Reconstruct the normal sentence without formatting and annotations
            sent_recon = re.sub(pattern_pair, r'\g<name>', annot)
            if sent != sent_recon:
                d_log = {'sentence': sent, 'annotated sentence': annot, 'reconstructed sentence': sent_recon, 'entities': entities}
                msg = f'Edge case: sentence and reconstructed sentence don\'t match'

                enms = [m_.group('name') for m_ in ms]
                only_diff_comma = False  # check if missing comma *outside* entity names
                ms = [re.search(re.escape(nm), sent) for nm in enms]  # enforce exact match on entity names
                if all(m_ is not None for m_ in ms):
                    spans = [m_.span() for m_ in ms]
                    if len(spans) > 1:
                        strt, end = spans[0]  # sanity check all matches are in-order and non-overlapping
                        for strt_, end_ in spans[1:]:
                            assert strt_ >= end
                            strt, end = strt_, end_

                    def drop_comma_outside() -> str:
                        """
                        Drop all commas that's not in any of the entity spans
                        """
                        ret = ''
                        for i, c in enumerate(sent):
                            if c == ',' and not any(s[0] <= i < s[1] for s in spans):
                                continue
                            ret += c
                        return ret
                    only_diff_comma = drop_comma_outside() == sent_recon
                only_diff_period = f'{sent_recon}.' == sent  # missing trailing period (.) for annotated sentence
                only_diff_punc = only_diff_comma or only_diff_period
                if only_diff_comma:
                    msg = f'{msg} on missing comma(s) only'
                elif only_diff_period:
                    msg = f'{msg} on missing trailing period only'
                self.ec(msg=f'{msg} w/ {pl.fmt(d_log)}', kind='missing-punc-in-reconstruction')
                if not only_diff_punc:  # have to drop such samples
                    # return None
                    raise NotImplementedError
            # return NaturalSample(sentence=sent, entities=entities)
            raise NotImplementedError
        elif self.sample_format == 'bio-list':
            if '\n' in sample:
                toks, tags = sample.split('\n')

                toks, tags = toks.strip(), tags.strip()
                m_e = _match(text=toks, pattern=self.pattern_extract_toks, desc='tokens')
                m_g = _match(text=tags, pattern=self.pattern_extract_tags, desc='tags')

                toks, tags = m_e.group('tokens'), m_g.group('tags').split(f'{self.entity_sep} ')
            else:  # for edge case
                m = self.patterns_check_edge[1].match(sample)  # TODO: check
                assert m is not None
                toks, tags = m.group('tokens'), m.group('tags')
                toks, tags = toks.strip(), tags.strip().split(f'{self.entity_sep} ')

            prior_implementation = False
            d_log = dict(sample=sample, tokens=toks, tags=tags)
            if prior_implementation:
                toks = toks.split(f'{self.token_sep} ')
                n_tok = len(toks)
                assert isinstance(self.token_map, TokenMapEnclose)
                toks = [self.token_map.decode(t, strict=(i+1 != n_tok)) for i, t in enumerate(toks)]  # see `TokenMap`
            else:
                def log_trailing_punc():
                    # handles edge case, trailing punctuations outside of token enclosing, same as in `decode` w/ `strict` above
                    __msg = f'Edge case: trailing punctuations outside of token enclosing in sample w/ {pl.fmt(d_log)}'
                    self.ec(msg=__msg, kind='tok-trailing-punc')
                if toks[-1] in ['.', ',']:
                    toks = toks[:-1]
                    log_trailing_punc()
                elif toks[-5:] == '"."."':
                    toks = toks[:-2]
                    log_trailing_punc()
                assert isinstance(self.token_map, TokenMapEnclose)
                toks = self.token_map.find_tokens_by_enclosing(toks)

            def log_trailing_punc():
                _msg = f'Error: trailing punctuation after a tag in sample w/ {pl.fmt(d_log)}'
                self.ec(msg=_msg, kind='tag-trailing-punc')

            if tags[-1] == 'O.':  # an edge case/bad completion encountered
                tags[-1] = 'O'
                log_trailing_punc()
            if any(t == 'O;' for t in tags):
                tags = ['O' if t == 'O;' else t for t in tags]
                log_trailing_punc()
            # edge case: NER tag like `O-location`
            if any(s.startswith('O-') for s in tags):
                msg_ = f'Error: NER tag with type begins w/ [{pl.i("O")}] in sample w/ {pl.fmt(d_log)}'
                self.ec(msg=msg_, kind='broken-tag')
                tags = ['O' if s.startswith('O-') else s for s in tags]

            # return BioSample(tokens=toks, tags=tags)
            raise NotImplementedError
        elif self.sample_format == 'bio-list-v2':
            sent, toks, tags = sample.split('\n')
            m_s = _match(text=sent, pattern=self.pattern_extract_sent, desc='sentence')
            m_t = _match(text=toks, pattern=self.pattern_extract_toks, desc='tokens')
            m_g = _match(text=tags, pattern=self.pattern_extract_tags, desc='tags')
            sent = m_s.group('sent').strip()
            toks = m_t.group('tokens').strip().split(f'{self.token_sep} ')
            tags = m_g.group('tags').strip().split(f'{self.entity_sep} ')

            # for edge case:
            #   sentence: The fashion designer's latest collection received rave reviews from critics and fashion enthusiasts.
            #   tokens: [The, fashion, designer, 's, latest, collection, received, rave, reviews,
            #           from, critics, And, fashion, enthusiasts, .]
            #   note the `And`
            if detokenize(toks).lower() != sent.lower():
                d_log = dict(sentence=sent, tokens=toks)
                self.ec(msg=f'Edge case: sentence and tokens don\'t match w/ {pl.fmt(d_log)}', kind='sent-tok-mismatch')

            new_tags, fixed = Completion2Samples.fix_wrong_bio(tags=tags, tokens=toks)
            if len(fixed) > 0:
                assert new_tags != tags
                d_log = dict(sentence=sent, tokens=toks, original_tags=tags, fixed=fixed)
                self.ec(msg=f'Edge case: wrong BIO tags w/{pl.fmt(d_log)}', kind='broken-tags')
                tags = new_tags
            # return BioSample(tokens=toks, tags=tags)
            raise NotImplementedError
        else:
            from stefutil import sic

            assert self.sample_format == 'bio-line'

            pairs = sample.split('\n')
            sic(pairs)
            toks, tags = [], []
            for pr in pairs:
                m = self.pattern_extract_pair.match(pr)
                if m is None:
                    sic(pr)
                assert m is not None
                t, tag = m.group('token'), m.group('tag')
                toks.append(t)
                tags.append(tag)
            # return BioSample(tokens=toks, tags=tags)
            raise NotImplementedError

    def __call__(
            self, completion: str, n_samples: int = None, filename: str = None, logprobs: List[Dict[str, Any]] = None,
            resolve_extraction_failure: bool = False
    ) -> ReadableSamples:
        data = []
        edge_cases: List[Dict[str, Any]] = []

        if completion[-1] != '\n':  # for pattern match
            completion += '\n'
        samples = list(self.split_completion(completion=completion, filename=filename))
        samples_str = [s.sample for s in samples]
        n_sample_found = len(samples_str)
        e_sent = 'Entity Names: None'
        if any(s == e_sent for s in samples_str):
            msg = f'Edge case: entire sample is [{pl.i(e_sent)}]'
            self.ec(msg=msg, kind='none-sample')
            edge_cases.append(dict(sample=e_sent))

            idxs_drop = [i for i, s in enumerate(samples_str) if s == e_sent]
            # samples_str = [s for s in samples_str if s != e_sent]
            samples = [s for i, s in enumerate(samples) if i not in idxs_drop]
            # samples_str = [s for i, s in enumerate(samples_str) if i not in idxs_drop]

        if n_samples is not None and len(samples) != n_samples:
            # based on the number of samples in the completion, not the successfully decoded ones
            d_log = dict(filename=filename, n_sample_expect=n_samples, n_sample_found=len(samples))
            msg = f'Expected {pl.i(n_samples)} samples, but decoded {pl.i(len(samples))} w/ {pl.i(d_log)}'
            self.ec(msg=msg, kind='wrong-sample-count', args=d_log, failed=True)

        n_drop, n_non_entity = 0, 0
        sents = []  # all sentences in completion
        failed_sentences = defaultdict(list)  # edge case => sentence
        lps, s2lp, et_lps = None, None, None
        # lp_kd = '1-stage-sample'  # use logprob for the entire sample
        lp_kd = '1-stage-annotations'  # use logprob for the label part

        if logprobs:
            s2lp = logprob.Span2LogProb(logprobs=logprobs, completion=completion, sample_type=lp_kd, ec=self.ec, logger=self.logger)
            lps, et_lps = [], []
        drop_punc_args = dict(pattern_emph=self.pattern_emph, ec=self.ec, filename=filename)
        for i, sample in enumerate(samples, start=1):
            sample_str = sample.sample
            sample_ord = f'{pl.i(ordinal(i))}/{pl.i(len(samples))}'
            err_pref = f'Edge case in extracting {sample_ord} sample:'

            out = self.extract_single_sample(sample_str)
            if not isinstance(out, ExtractSingleSampleOutput):
                raise NotImplementedError
            if out.no_entity_dropped:
                n_drop += 1
                continue
            if out.no_entity:
                n_non_entity += 1
            if not out.success:
                raise NotImplementedError
                # continue
            entities_raw = out.entities_raw
            sent = out.sentence
            out = out.sample

            def handle_broken(d_log_: Dict[str, Any] = None, sentence: str = None):
                # Add to edge case, also flag the sentence as challenging
                edge_cases.append(d_log_)
                assert sentence is not None
                sentence = edit.drop_puncs_in_sentence(sentence=sentence, d_log=d_log_, **drop_punc_args)
                assert out.fail_reason is not None
                failed_sentences[out.fail_reason].append(sentence)
                sents.append(sentence)

            if self.sample_format in ['natural-pair', 'natural-pair-v2', 'natural-inline', 'natural-inline-v2']:
                out = self.nt.y2entity_n_type(sample=out, filename=filename, resolve_failure=resolve_extraction_failure)
                if not out.success:
                    # `pair2triple` will not modify the sentence, so can use the one from before
                    # but since this is end of pipeline, drop enclosing quotes & inner brackets here
                    handle_broken(d_log_=out.d_log, sentence=sent)
                    continue
                out = out.sample
                sent, en, et = out.sentence, out.entity_names, out.entity_types

                if self.sample_format == 'natural-pair':
                    raise NotImplementedError
                elif self.sample_format == 'natural-pair-v2':
                    out = self.nt.sanitize(
                        sentence=sent, entity_names=en, entity_types=et, error_prefix=err_pref, filename=filename,
                        resolve_failure=resolve_extraction_failure)
                    if not out.success:
                        # `triple2sample` modifies the sentence, i.e. dropping enclosing quotes & inner brackets,
                        # but the processing code may not reach that point, so drop the puncs again to be safe
                        handle_broken(d_log_=out.d_log, sentence=out.sentence)
                        continue
                    ner_sample = out.sample
                    data.append(ner_sample)
                    sents.append(ner_sample.sentence)
                    if logprobs:
                        lp_out = logprob.ner_sample2logprobs(
                            sample=ner_sample, sample_str=sample.sample_raw, sample_span=sample.span, span2logprob=s2lp,
                            entities_str=entities_raw, pattern_entity=self.pattern_extract_entity,
                            entity_sep=self.entity_sep, entity_pair_map=self.entity_pair_map
                        )
                        lps.append(lp_out.logprob)
                        et_lps.append(lp_out.entity_logprobs)
                else:  # [`natural-inline`, `natural-inline-v2`]
                    assert all(en_ in sent for en_ in en)  # a stricter check by the format construction
                    assert not check.entities_overlapping(sentence=sent, entity_names=en).overlap
                    raise NotImplementedError
                    # d.update(entity_names=tuple(en), entity_types=tuple(et))
                    # data.append(NerReadableExample(**d))
            elif self.sample_format in ['bio-list', 'bio-list-v2']:
                toks, tags = out.tokens, out.tags

                # handle edge case, many times, missing NER tag for last punctuation token, `.`
                # TODO: as a heuristic, may add wrong label...
                if len(toks) == len(tags) + 1 and toks[-1] == '.':
                    tags.append('O')
                    d_log = dict(tokens=toks, tags=tags)
                    msg = f'Edge case: missing NER tag for last punctuation token [{pl.i(".")}], ' \
                          f'added {pl.i("O")} tag for Sample {sample_ord}: [{pl.fmt(d_log)}]'
                    self.ec(msg=msg, kind='missing-last-tag')

                if len(toks) == len(tags):  # sanity check annotation exists for each token
                    data.append(NerBioExample.from_tokens_n_tags(tokens=toks, tags=tags))
                else:
                    d_log = {'tokens': toks, 'tags': tags, '#tokens': len(toks), '#tags': len(tags)}
                    msg = f'{err_pref} Number of tokens and tags don\'t match w/ {pl.fmt(d_log)}'
                    self.ec(msg=msg, kind='length-mismatch')
                    edge_cases.append(d_log)
            else:
                assert self.sample_format == 'bio-line'
                toks, tags = out.tokens, out.tags
                # by demo format construction. len(toks) == len(tags)
                data.append(NerBioExample.from_tokens_n_tags(tokens=toks, tags=tags))

        if len(data) == 0:
            msg = f'No samples decoded from completion:\n{pl.i(completion)}\n'
            if filename:
                msg = f'{msg}\nand file name: {pl.i(filename)}'
            self.ec(msg=f'{msg}, decoding format may be wrong.', kind='no-sample-decoded', failed=True)
        return ReadableSamples(
            n_sample_found=n_sample_found, samples=data, sentences=sents, logprobs=lps if logprobs else None, entity_logprobs=et_lps,
            edge_cases=edge_cases, failed_sentences=failed_sentences, num_no_entity=n_non_entity, num_dropped=n_drop
        )


class NerDatasetWriter:
    """
    Process GPT completions to NER dataset in BIO format ready for training, write them to local disk
    """

    def __init__(
            self, dataset_name: str = None, completion2samples: Completion2Samples = None,
            sample_format: str = 'natural-pair', as_passage: bool = False,
            allowed_entity_types: Union[bool, List[str]] = None,
            drop_samples_wo_entities: Union[bool, float] = False,
            detect_enum: bool = False, lowercase_entity_type: bool = None, logger: logging.Logger = None
    ):
        """
        :param dataset_name: Name of the dataset
        :param completion2samples: See `Completion2Samples`
        :param sample_format: See `demo2prompt.py::PromptConstructor`
        :param as_passage: See `demo2prompt::PromptConstructor`
        :param allowed_entity_types: See `demo2prompt::PromptConstructor`
        :param drop_samples_wo_entities: See `Completion2Samples`
        :param detect_enum: dynamically determine if `has_enum_prefix` is True, see `Completion2Samples`
            intend for `natural-inline` only
        """
        self.dataset_name = dataset_name or 'conll2003'
        dnm_out = self.dataset_name.replace('-', '_')
        self.dset_path = os_join('data_generation', 'data', dnm_out)

        self.allowed_entity_types = None
        if isinstance(allowed_entity_types, list):
            self.allowed_entity_types = allowed_entity_types
        elif allowed_entity_types is True:
            self.allowed_entity_types = sconfig(f'datasets.{dataset_name}.readable-entity-types')
        self.lowercase_entity_type = lowercase_entity_type or False
        self.drop_samples_wo_entities = drop_samples_wo_entities
        self.detect_enum = detect_enum

        self.logger = logger or _logger
        c2s_args = dict(
            dataset_name=dataset_name, sample_format=sample_format, allowed_entity_types=self.allowed_entity_types,
            lowercase_entity_type=self.lowercase_entity_type, drop_samples_wo_entities=drop_samples_wo_entities, logger=self.logger)
        c2s = completion2samples or Completion2Samples(**c2s_args)
        self.has_enum = has_enum = c2s.has_enum_prefix
        self.d_c2s = {has_enum: c2s}
        self.ec = c2s.ec
        if detect_enum:
            assert sample_format in ['natural-pair-v2', 'natural-inline', 'natural-inline-v2', 'bio-list-v2']
            c2s_ = Completion2Samples(**c2s_args, has_enum_prefix=not has_enum, ec=self.ec)  # share one edge case logger
            self.d_c2s[not has_enum] = c2s_

        ca(sample_format=sample_format)
        self.sample_format = sample_format

        self.data_format = sample_fmt2data_fmt(sample_format=sample_format)

        self.as_passage = as_passage
        if as_passage:
            assert dataset_name is not None
            if dataset_name == 'conll2003':
                # e.g.
                #   **Title**: <title of article>
                #
                #   **Article**:
                #   <content of article, arbitrary lines>
                #
                #   **sentences**:
                #   samples just like normal completions
                #
                #   [End of Sentences]  // optional
                if sample_format == 'natural-pair-v2':
                    # pattern to match a single new article group
                    s = r'(\*\*|\*)?'  # for edge case, the bold-facing chars may be missing, or just a single `*`
                    self.pattern_passage_search = [
                        re.compile(
                            # edge cases:
                            #   1. whitespace after `Article` prefix
                            #   2. whitespace after `Sentences` prefix
                            #   3. `Title`, `Article`, `Sentences` boldface includes `:`
                            rf'{s}Title{s}:{s} (?P<title>.*)\n\n{s}Article{s}:{s}( )?\n(\n)?(?P<article>.*)\n\n{s}Sentences{s}:{s}( )?\n(?P<samples>.*)(\n\n\[End of Sentences])?',
                            re.IGNORECASE | re.DOTALL
                        ),
                        # edge case, sentence prefix missing, sentences directly follow article
                        re.compile(
                            rf'{s}Title{s}: (?P<title>.*)\n\n{s}Article{s}:\n(?P<samples>.*)(\n\n\[End of Sentences])?',
                            re.IGNORECASE | re.DOTALL
                        )
                    ]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        d_log = {
            'dataset-name': self.dataset_name, 'sample-format': self.sample_format, 'data-format': self.data_format,
            'as-passage': self.as_passage, 'allowed-entity-types': self.allowed_entity_types,
            'lowercase-entity-type': self.lowercase_entity_type, 'drop-samples-wo-entities': self.drop_samples_wo_entities,
            'detect-enum': self.detect_enum
        }
        self.logger.info(f'{pl.i(self.__class__.__qualname__)} initialized w/ {pl.i(d_log, indent=1)}')

    @property
    def meta(self) -> Dict:
        return dict(
            dataset_name=self.dataset_name, sample_format=self.sample_format, data_format=self.data_format,
            detect_enum=self.detect_enum, has_enum_prefix=self.has_enum,
            drop_samples_wo_entities=self.drop_samples_wo_entities
        )

    def __call__(
            self, completions_dir_name: completions.CompletionDirectory, expected_samples_per_completion: int = None,
            output_dir_name: str = None, lowercase: bool = False, logprobs: bool = None, resolve_extraction_failure: bool = False
    ):
        out = dataset_name2data_dir(
            dataset_name=self.dataset_name, output_dir='NER-Dataset', output_postfix=output_dir_name,
            timestamp='short-date'
        )
        output_path, base_path = out.path, out.base_path
        d_log = {
            'class-name': self.__class__.__qualname__, 'metadata': self.meta, 'output-path': output_path,
            'completions-dir-name': completions_dir_name, 'expected-samples-per-completion': expected_samples_per_completion,
            'lowercase': lowercase, 'logprobs': logprobs, 'resolve-extraction-failure': resolve_extraction_failure
        }
        out = completions.process_completions_init(
            completion_base_path=base_path, completions_dir_name=completions_dir_name, output_path=output_path, completion_type='NER',
            logger=self.logger, init_log=d_log, logprobs=logprobs
        )
        completions.log_prompt_eg(dir_name=completions_dir_name, base_path=base_path, logger=self.logger)
        d_log_count = {'#completions': len(out.filepaths)}

        t = Timer()
        examples, sents, n_sample_found, n_no_entity, n_drop = [], [], 0, 0, 0
        edge_cases, challenging_sentences = [], defaultdict(list)
        lps, entity_lps = None, None
        if logprobs:
            lps, entity_lps = [], []
        for c in out.iter:
            cpl, fnm, p_fnm = c.content, c.filename, c.pretty_filename

            if self.as_passage:  # break up each completion file to individual passages
                ms: List[re.Match] = patterns.find_non_overlap_matches(
                    pattern=self.pattern_passage_search, text=cpl, return_matches=True)
                if len(ms) > 1:
                    raise NotImplementedError
                completion_group = [m.group('samples') for m in ms]
            else:
                completion_group = [cpl]
            for c_ in completion_group:
                has_enum = completions.completion_has_enum_prefix(completion=c_) if self.detect_enum else self.has_enum
                out = self.d_c2s[has_enum](
                    completion=c_, n_samples=expected_samples_per_completion, filename=fnm, logprobs=c.logprobs,
                    resolve_extraction_failure=resolve_extraction_failure)
                if len(out.sentences) != out.n_sample_found:
                    from stefutil import sic
                    sic(out.sentences)
                    raise NotImplementedError
                if resolve_extraction_failure and out.n_sample_found != expected_samples_per_completion:
                    from stefutil import sic
                    sic(out.samples)
                    raise NotImplementedError
                n_sample_found += out.n_sample_found
                examples += out.samples
                sents += out.sentences
                if logprobs:
                    lps += out.logprobs
                    entity_lps += out.entity_logprobs

                edge_cases += out.edge_cases
                for fail_kind, sents__ in out.failed_sentences.items():
                    challenging_sentences[fail_kind] += sents__

                n_no_entity += out.num_no_entity
                n_drop += out.num_dropped
        d_log_count['#sample-found'] = n_sample_found

        os.makedirs(output_path, exist_ok=True)
        if len(edge_cases) > 0:
            edge_path = os_join(output_path, 'edge_cases.json')
            _logger.warning(f'{pl.i(len(edge_cases))} found edge cases written to {pl.i(stem(edge_path))}')
            with open(edge_path, 'w') as f:
                json.dump(edge_cases, f, indent=4)
        return dataset.finish_ner_processing(
            samples=examples, sentences=sents, challenging_sentences=challenging_sentences, 
            logprobs=logprobs, sample_logprobs=lps, entity_logprobs=entity_lps,  
            dedup=True, lowercase=lowercase, output_path=output_path, d_log=d_log_count, time=t,
            ec=self.ec, logger=self.logger, dataset_name=self.dataset_name, data_format=self.data_format,
            entity_types=self.allowed_entity_types)
