"""
Modifications on a generated NER sample
    Intended for pipeline on conversion to valid NER samples
"""

import itertools
from copy import deepcopy
from typing import Dict, Tuple, List, Union, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter

from tqdm import tqdm

from stefutil import *
from src.util import patterns
from src.util import sample_check as check
from src.util.ner_example import *
from src.util.sample_formats import *
from src.data_util.prettier import EdgeCases, sdpc


__all__ = [
    'entities_differ_in_case_only',
    'merge_entities_on_separator',
    'enclose_in_quote', 'sanitize_quotes',
    'drop_enclosing_quotes', 'drop_enclosing_brackets',
    'drop_entities_enclosing_puncs', 'drop_brackets_in_sentence', 'drop_puncs_in_sentence',
    'duplicate_multi_occurring_entities', 'reorder_entities'
]


_logger = get_logger('Sample-Edit')


def merge_entities_on_separator(entities: List[str], entity_sep: str = None, entity_pair_map: EntityPairTemplate = None) -> List[str]:
    # for edge case: an entity span contains comma itself
    assert entity_sep == ','
    assert isinstance(entity_pair_map, EntityPairEncloseType)
    assert entity_pair_map.open_char == '(' and entity_pair_map.close_char == ')'
    entities_ = []
    i = 0
    while i < len(entities):
        e = entities[i]
        if '(' in e and e[-1] == ')':  # a whole entity span
            entities_.append(e)
        elif len(e.strip()) > 0:
            # drop empty entities, e.g. `Named Entities: [Beyoncé (person), ]`
            e_ = e
            while e_[-1] != ')' and i + 1 < len(entities):
                i += 1
                e_ = f'{e_}, {entities[i]}'
            entities_.append(e_)
        i += 1
    return entities_


def entities_differ_in_case_only(entity_names: Union[List[str], Tuple[str]] = None, sentence: str = None, ec: EdgeCases = None) -> bool:
    if check_single_char_appearance(entity_names=entity_names, sentence=sentence):
        if ec:
            d_log = dict(sentence=sentence, entity_names=entity_names)
            msg = f'Edge case: `A` is an annotated entity and `a` appears in the sentence w/ {pl.i(d_log, indent=1)}'
            ec(msg=msg, kind='A-as-entity-&-a-in-sentence', args=d_log)
        return True
    en_lc = [e.lower() for e in entity_names]
    return len(entity_names) == len(set(en_lc)) and len(entity_names) != len(set(entity_names))


def enclose_in_quote(s: str) -> str:
    """
    Enclose a string in quotes
    """
    # handle cases where the sentence itself is double-quoted, or contain double quotes, use single quotes
    quote = "'" if '"' in s else '"'
    return f'{quote}{s}{quote}'
    # # A sentence may even have both double quotes and single quotes, pick the less frequent one
    # # quote = min(["'", '"'], key=lambda x: s.count(x))
    # count_single, count_double = s.count("'"), s.count('"')
    # sic(s, count_single, count_double)
    # if count_single == 0 and count_double == 0:
    #     quote = '"'
    # else:  # in case of tie, prefer single quote
    #     quote = "'" if count_single <= count_double else '"'
    # return f'{quote}{s}{quote}'


def drop_enclosing_quotes(s: str) -> str:
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    elif s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    else:
        return s


def drop_enclosing_brackets(s: str) -> str:
    if s[0] == '[' and s[-1] == ']':
        return s[1:-1]
    else:
        return s



@dataclass
class DupMultiOccurEntitiesOutput:
    entity_names: List[str] = None
    entity_types: List[str] = None
    d_log: Dict[str, Any] = None


def duplicate_multi_occurring_entities(
        entity_names: List[str] = None, entity_types: List[str] = None, entity_name2count: Dict[str, int] = None,
        d_log: Dict[str, Any] = None, allow_limit: int = 3
):
    """
    For duplicating entities N times if it occurs N times in the sentence

    :param entity_names: Entity names before duplication
    :param entity_types: Entity types corresponding to `entity_names`
    :param entity_name2count: Entity name to #occurrence in the sentence
    :param d_log: A dict of logging info to be updated
    :param allow_limit: If the entity occurs more than `allow_limit` times, raise an error
    """
    enms, ets, c = entity_names, entity_types, entity_name2count

    # Assume the entity name has the same type for all occurrences
    en2et = dict(zip(enms, ets))  # since entity names are all distinct, this is well-defined
    if max(c.values()) > allow_limit:
        sic(enms, ets, c)
        raise NotImplementedError

    enms_ = []  # add the missing entity names up until reaching #occurrence
    for enm, count in c.items():
        if count > 1:
            enms_ += [enm] * (count - 1)
    # add duplicated entities to the front; will be reordered later in `reorder_entities`
    enms = enms_ + deepcopy(enms)
    d_log['entity_names_after_add'] = enms
    if ets is not None:
        ets = [en2et[enm] for enm in enms_] + deepcopy(ets)
        d_log['entity_types_after_add'] = ets
    return DupMultiOccurEntitiesOutput(entity_names=enms, entity_types=ets, d_log=d_log)





@dataclass
class ValidOrderingOutput:
    order: Tuple[int] = None
    elms_ordered: List[str] = None


def _get_valid_ordering(
        sentence: str = None, entity_names: List[str] = None, entity_types: List[str] = None,
        ignore_case: bool = False, insert_if_more: int = 8, _inner_call_dup: bool = False
) -> ValidOrderingOutput:
    # trying all permutations can be computationally prohibitive for too many entity names
    # e.g.
    #   "The International Space Station, a multinational space laboratory, is a collaboration between NASA, Roscosmos (Russia's space agency), ESA (European Space Agency), JAXA (Japan's space agency), and CSA (Canadian Space Agency)."
    #   Entity Names: [International Space Station (space station), NASA (organization), Roscosmos (organization), ESA (organization), JAXA (organization), CSA (organization), Russia (location), European Space Agency (organization), Japan (location), Canadian Space Agency (organization)]
    # so if too many entity names
    #   1. for the first `insert_if_more` entities, try all permutations
    #   2. iteratively insert the remaining entities, one by one, and greedily pick the first valid ordering
    insert_if_more = insert_if_more or 8
    distinct_enm = len(set(entity_names)) == len(entity_names)  # each entity name is distinct
    # check if any duplicate entity name has different entity types
    # TODO: if duplicate entity names and each correspond to different entity types, this will fail
    dup_groups = None
    if not distinct_enm:
        # group the same entity names, store their corresponding entity types and index
        dup_groups: Optional[Dict[str, List[Tuple[str, int]]]] = defaultdict(list)
        c = Counter(entity_names)
        dup_enms = {enm for enm, cnt in c.items() if cnt > 1}
        for i, enm in enumerate(entity_names):
            if enm in dup_enms:  # TODO: verify
                dup_groups[enm].append((entity_types[i] if entity_types is not None else None, i))
        # check if any duplicate entity name has different entity types
        if any(len({et for (et, idx) in lst}) > 1 for lst in dup_groups.values()):
            sic(sentence, entity_names, entity_types, dup_groups)
            raise NotImplementedError

    def _find_matches(enms: List[str]):
        return patterns.find_matches(text=sentence, keywords=enms, ignore_case=ignore_case, search_in_order=True, suppress_error=True)

    if len(entity_names) > insert_if_more:
        # TODO: to ensure passing the non-distinct-ordering-count check, all duplicates must be in the inner call
        l_enms = len(entity_names)
        _logger.warning(f'Too many entity names for exhaustive search, '
                        f'trying greedy search instead for entity range [{pl.i(insert_if_more)}:{pl.i(l_enms)})')

        _logger.info(f'Exhaustive search for entity range [{pl.i(0)}:{pl.i(insert_if_more)}]')
        ets = entity_types[:insert_if_more] if entity_types else None
        out = _get_valid_ordering(
            sentence=sentence, entity_names=entity_names[:insert_if_more], entity_types=ets,
            ignore_case=ignore_case,
            # the subset of entity names may be unique, but not all the entity names, this disables assertion
            _inner_call_dup=not distinct_enm
        )
        past_order, past_enms = out.order, out.elms_ordered
        it = tqdm(entity_names[insert_if_more:], desc='Greedy Search for Ordering')
        # for enm in entity_names[insert_if_more:]:
        for enm in it:
            found = False
            insert_idxs = list(range(len(past_order) + 1))
            for idx in insert_idxs:
                enms_try = past_enms[:idx] + [enm] + past_enms[idx:]
                order_try = tuple(enms_try.index(enm) for enm in enms_try)
                if _find_matches(enms_try).success:  # prep for next iter
                    past_order = order_try
                    past_enms = enms_try
                    found = True
                    break
            assert found
        ordering = tuple(entity_names.index(past_enms[i]) for i in past_order)  # re-order the ordering to match the original entity names
        # raise NotImplementedError
        return ValidOrderingOutput(order=ordering, elms_ordered=past_enms)
    else:
        n_enm = len(entity_names)
        all_ordering: List[Tuple[int]] = list(itertools.permutations(range(n_enm)))
        working = []
        it = tqdm(all_ordering, desc='Exhaustive Search for Ordering')
        it.set_postfix({'#entity': pl.i(n_enm)})
        # for ordering in all_ordering:
        for ordering in it:
            if _find_matches([entity_names[i] for i in ordering]).success:
                working.append(ordering)
        # TODO: if the same entity name appears multiple times, there will be multiple valid orderings
        if distinct_enm:
            if not _inner_call_dup:
                if len(working) != 1:
                    sic(sentence, entity_names, entity_types, len(working))
                assert len(working) == 1  # sanity check
        else:
            if not len(working) > 1:
                sic(distinct_enm, sentence, entity_names, entity_types, working)
            assert len(working) > 1
            # sanity check that the multiple working ordering only differ in the duplicate entity names
            # => all orderings are equivalent
            idx2enm = {idx: enm for enm, lst in dup_groups.items() for et, idx in lst}
            _working = [
                [(idx2enm[i]) if i in idx2enm else i for i in ordering] for ordering in working
            ]
            if not all(_working[0] == _work for _work in _working[1:]):
                sic(sentence, entity_names, entity_types, working, _working)
            assert all(_working[0] == _work for _work in _working[1:])
        ordering = working[0]
        return ValidOrderingOutput(order=ordering, elms_ordered=[entity_names[i] for i in ordering])


@dataclass
class ReorderEntityOutput:
    reordered: bool = None
    entity_names: List[str] = None
    entity_types: List[str] = None


def reorder_entities(
        sentence: str = None, entity_names: List[str] = None, entity_types: List[str] = None, ignore_case: bool = False,
        insert_if_more: int = None
) -> ReorderEntityOutput:
    """
    :return: Entity name and entity types reordered to match the order in `text`, if re-ordering is needed
    """
    # By exhaustive search on all possible permutations of `entity_names`, find the permutation that matches the order in `text`
    _reorder = False
    # try:
    #     _ms = find_matches(text=sentence, keywords=entity_names, ignore_case=ignore_case, search_in_order=True)
    # except ValueError:  # need re-ordering iff `ValueError` definitely raised
    #     _reorder = True
    # one-to-one mapping for match fail and re-ordering
    if not patterns.find_matches(
            text=sentence, keywords=entity_names, ignore_case=ignore_case, search_in_order=True, suppress_error=True).success:
        _reorder = True

    if _reorder:
        out = _get_valid_ordering(
            sentence=sentence, entity_names=entity_names, entity_types=entity_types, ignore_case=ignore_case, insert_if_more=insert_if_more)
        ordering, enms = out.order, out.elms_ordered
        # re-order entity types to match the re-ordered entity names
        ets = [entity_types[i] for i in ordering] if entity_types is not None else None
        return ReorderEntityOutput(reordered=True, entity_names=enms, entity_types=ets)
    else:
        return ReorderEntityOutput(reordered=False)


@dataclass
class DropEntitiesPuncsOutput:
    entity_names: List[str] = None
    entity_names_modified: List[str] = None  # original version before dropping


def drop_entities_enclosing_puncs(
        entity_names: List[str] = None, dataset_name: str = 'conll2003', ec: EdgeCases = None, drop: Union[bool, str] = True,
        d_log: Dict[str, Any] = None
) -> DropEntitiesPuncsOutput:
    """
    LLM-annotated named entities may be included in double quotes or brackets, drop them if requested
    """
    # if quotes are at the edge of the entity spans, drop them
    # TODO: should check that quotes are not included in any dataset ground truth before doing this
    if drop is True:
        drop = 'quote'
    ca.assert_options(display_name='Entity Name Punc Drop Type', val=drop, options=[False, 'quote', 'both'])

    enms = entity_names
    mods = []
    hp = [check.has_punc_on_edge(enm) for enm in enms]
    if any(hp.has_quote or hp.has_brackets for hp in hp):
        if dataset_name not in ['conll2003-no-misc', 'wiki-gold-no-misc', 'mit-movie', 'mit-restaurant']:
            sic(dataset_name, entity_names)
            raise NotImplementedError

        quote_one_side, quote_both_side, brackets = [], [], []
        for enm in enms:
            q = check.has_punc_on_edge(enm)
            if q.has_quote:
                if not q.on_both_side:
                    quote_one_side.append(enm)
                else:
                    quote_both_side.append(enm)
                    if drop in ['quote', 'both']:
                        mods.append(enm)
            if q.has_brackets:
                brackets.append(enm)
                if drop in ['both']:
                    mods.append(enm)
        if ec:
            d_log = d_log or dict()
            if len(quote_one_side) > 0 or len(quote_both_side) > 0:
                spans_w_quote = quote_one_side + quote_both_side
                d_log['spans_with_quote_one_side'] = quote_one_side
                msg = f'Edge case: quote found on at least one side of entity spans w/ {sdpc(d_log)}'
                ec(msg=msg, kind='entity-span-quote-on-side', args=dict(spans=spans_w_quote))

            kd = 'entity-span-drop-puncs'
            if drop in ['quote', 'both'] and len(quote_both_side) > 0:
                spans_w_quote = [enm for enm in enms if check.has_punc_on_edge(enm).has_quote]
                d_log['spans_with_quote_both_side'] = quote_both_side
                msg = f'Edge case: enclosing quotes dropped from entity spans w/ {sdpc(d_log)}'
                assert all(check.has_punc_on_edge(enm).on_both_side for enm in spans_w_quote)  # sanity check
                ec(msg=msg, kind=kd, args=dict(spans=spans_w_quote))
                enms = [drop_enclosing_quotes(enm) for enm in enms]
            if drop in ['both'] and len(brackets) > 0:
                msg = f'Edge case: enclosing brackets dropped from entity spans w/ {sdpc(d_log)}'
                spans_w_brackets = [enm for enm in enms if check.has_punc_on_edge(enm).has_brackets]
                ec(msg=msg, kind=kd, args=dict(spans=spans_w_brackets))
                enms = [drop_enclosing_brackets(enm) for enm in enms]
    return DropEntitiesPuncsOutput(entity_names=enms, entity_names_modified=mods)


@dataclass
class UpperInsideOutput:
    found: bool = None
    words: List[str] = None


def upper_inside(sentence: str) -> UpperInsideOutput:
    """
    Checks whether any word in the sentence contains an uppercase letter inside the word
    """
    ret = []
    # words = sentence.split()
    words = punc_tokenize(sentence=sentence)  # split on any whitespace & punctuation
    for word in words:
        if any(c.isupper() for c in word[1:]) and not word.isupper():  # if word is all caps, ignore
            ret.append(word)
    found = len(ret) > 0
    return UpperInsideOutput(found=found, words=ret if found else None)


def drop_brackets_in_sentence(sentence: str = None, pattern_emph: patterns.Patterns = None, ec: EdgeCases = None, **kwargs) -> str:
    sent = sentence
    if len(patterns.find_match(text=sent, pattern=pattern_emph)) > 0:
        ori = sent
        # drop the emphasized entity type by replacing regex matches w/ just the named entity
        emphs = []
        for pattern in pattern_emph:
            emphs_ = [(m.group(), m.group('et')) for m in patterns.find_match(text=sent, pattern=pattern)]
            emphs += [f'{pl.i(ori, c="y")} => {pl.i(et, c="g")}' for (ori, et) in emphs_]
            sent = pattern.sub(r' \g<et> ', sent)  # ensure whitespace before and after emphasized term
            sent = patterns.drop_consecutive_space(text=sent)

            # sanity check dropping emphasis didn't accidentally join 2 words together,
            #   by checking for uppercase letters in the middle of a word
            ui = upper_inside(sentence=sent)
            if ui.found and ec:
                d_log = dict(original=ori, modified=sent, emphasized=emphs, upper_inside_words=ui.words, **kwargs)
                msg = f'Uppercase letter inside word after dropping emphasis w/ {pl.i(d_log)}'
                ec(msg=msg, kind='drop-emph-word', args=dict(words=ui.words))

        if ec:
            d_log = dict(original=ori, modified=sent, emphasized=emphs, **kwargs)
            msg = f'Entity-enclosing punctuations dropped from sentence w/ {pl.i(d_log)}'
            ec(msg=msg, kind='drop-emph', args=dict(emphasized=emphs), disable_std_log=True)
    return sent.strip()


def drop_puncs_in_sentence(
        sentence: str = None, pattern_emph: patterns.Patterns = None, ec: EdgeCases = None, d_log: Dict[str, Any] = None, **kwargs
) -> str:
    if len(sentence) < 3:
        if ec:
            msg = f'Edge case: sentence is too short w/ {sdpc(d_log)}'
            ec(msg=msg, kind='sentence-too-short', args=d_log)
        return sentence

    sent = sentence
    if pattern_emph is not None:
        # drop bracket highlights within sentence
        sent = drop_brackets_in_sentence(sentence=sent, pattern_emph=pattern_emph, ec=ec, **kwargs)
    if check.has_punc_on_edge(sent).on_both_side:  # drop enclosing quotes & brackets from sentence
        msg = f'Edge case: sentence is enclosed in double quotes w/ {sdpc(d_log)}'
        ec(msg=msg, kind='sentence-enclosed-quote', disable_std_log=True)
        d_log['sentence_quote_dropped'] = sent = drop_enclosing_quotes(sent)
    return sent


non_ascii_quote2ascii_quote = {
    '‘': "'", '’': "'", '“': '"', '”': '"'
}


def sanitize_quotes(text: str = None, ec: EdgeCases = None, **kwargs) -> str:
    """
    LLM generated quotes are not in ASCII, e.g. `‘` instead of `'`, `“` instead of `"`, so replace them
    """
    if any(c in text for c in non_ascii_quote2ascii_quote):
        count = Counter((c for c in text if c in non_ascii_quote2ascii_quote))
        # sanity check double quotes should be paired, ending single quote `’` may have more counts than `‘`
        # if not count['‘'] == count['’'] and count['“'] == count['”']:
        #     sic(text, count)
        assert count['‘'] <= count['’'] and count['“'] == count['”']

        for c, c_ in non_ascii_quote2ascii_quote.items():
            text = text.replace(c, c_)
        if ec:
            kwargs['counts'] = count
            msg = f'Edge case: non-ASCII quote(s) found w/ {pl.i(kwargs)}'
            ec(msg=msg, kind='swap-non-ascii-quote', args=kwargs)
    return text
