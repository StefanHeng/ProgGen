"""
Checks on a generated NER sample
    Intended for pipeline on conversion to valid NER samples
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

from stefutil import punc_tokenize
from src.util.util_ import spans_overlap
from src.util import patterns


__all__ = [
    'have_word_overlap', 'has_punc_on_edge',
    'entities_in_sentence', 'get_non_overlapping_keyword_counts',
    'EntitiesOverlappingOutput', 'entities_overlapping',
]


def have_word_overlap(span1: str = None, span2: str = None) -> bool:
    # find all token split by puncs, to check for overlaps
    words_span, words_span_crt = punc_tokenize(sentence=span1), punc_tokenize(sentence=span2)
    return len(set(words_span) & set(words_span_crt)) > 0


quote_chars = ['"', "'"]


@dataclass
class HasEdgePuncOutput:
    has_quote: bool = None
    on_both_side: bool = None
    has_brackets: bool = None


def has_punc_on_edge(text: str) -> HasEdgePuncOutput:
    """
    :return: whether the edge of a string has a quote/bracket
    """
    s, e = text[0], text[-1]
    has_q = s in quote_chars or e in quote_chars
    both_q = s in quote_chars and e in quote_chars
    if has_q and both_q:
        assert s == e  # sanity check same quote char
    bas_both_b = (s, e) == ('[', ']')
    return HasEdgePuncOutput(has_quote=has_q, on_both_side=both_q, has_brackets=bas_both_b)


@dataclass
class EntitiesFoundInSentenceOutput:
    all_found: bool = None
    entities_found: List[str] = None
    entities_not_found: List[str] = None


def entities_in_sentence(sentence: str = None, entity_names: List[str] = None, ignore_case: bool = True) -> EntitiesFoundInSentenceOutput:
    """
    Syntactic sugar for `natural-pair` formats
    :return: If all entities are in sentence
    """
    enm2n_found = {e: len(patterns.find_match(text=sentence, keyword=e, ignore_case=ignore_case, strict=False)) for e in entity_names}

    all_found = all(n > 0 for n in enm2n_found.values())
    # # sanity check compatibility
    # assert all_found == all(len(find_match(text=sentence, keyword=e, ignore_case=ignore_case, strict=False)) > 0 for e in entity_names)

    found = [e for e, n in enm2n_found.items() if n > 0]
    not_found = [e for e, n in enm2n_found.items() if n == 0]

    return EntitiesFoundInSentenceOutput(all_found=all_found, entities_found=found, entities_not_found=not_found)


def get_non_overlapping_keyword_counts(sentence: str, keywords: List[str], ignore_case: bool = False) -> Dict[str, int]:
    """
    Count the exact match occurrence of each entity name in a sentence
    """
    # match longer entities first, in case entity names overlap
    # e.g.
    #   "The new iPhone 12 comes in four different models: iPhone 12 mini, iPhone 12, iPhone 12 Pro, and iPhone 12 Pro Max."
    #   Entity Names: ['iPhone 12 mini', 'iPhone 12', 'iPhone 12 Pro', 'iPhone 12 Pro Max']
    #   Entity Types: ['product', 'product', 'product', 'product']
    kws = sorted(keywords, key=lambda x: len(x), reverse=True)
    ret = {enm: 0 for enm in kws}
    for kw in kws:
        ms = patterns.find_match(text=sentence, keyword=kw, ignore_case=ignore_case, match_whole_words=True)
        if ms:  # each span of text can be used once, drop these spans from the sentence
            offset = 0
            for m in ms:
                # shifting the span when there are multiple matches; insert artificial space cos matching whole words may include space
                sentence = sentence[:m.start() - offset] + ' ' + sentence[m.end() - offset:]
                sentence = patterns.drop_consecutive_space(sentence)
                offset += m.end() - m.start()
            ret[kw] += len(ms)
    return ret


@dataclass
class EntitiesOverlappingOutput:
    overlap: bool = None
    matches: List[re.Match] = None
    entity_matches: List[List[re.Match]] = None  # possible spans for each entity by entity list position
    entity_spans: List[List[Tuple[int, int]]] = None


def entities_overlapping(
        sentence: str, entity_names: List[str], ignore_case: bool = False, search_in_order: bool = True
) -> EntitiesOverlappingOutput:
    """
    :param sentence: Text to search for `keywords`
    :param entity_names: List of keywords to search for in `text`
    :param ignore_case: If true, ignore casing when searching for `keywords`
    :param search_in_order: See `_find_exact_matches`
    :return: True if any of the keywords in `keywords` overlaps with each other

    Caller need to ensure all entities are in the sentence
    """
    if len(entity_names) <= 1:
        return EntitiesOverlappingOutput(overlap=False)
    else:
        try:
            out = patterns.find_matches(
                text=sentence, keywords=entity_names, ignore_case=ignore_case, search_in_order=search_in_order,
                return_all=not search_in_order
            )
            ms, et_ms = out.matches, out.keyword_matches  # the matches for each entity
            et_spans = [[m.span('keyword') for m in ms] for ms in et_ms]
        except ValueError:  # certain entity not found
            # sanity check use case
            assert entities_in_sentence(sentence=sentence, entity_names=entity_names, ignore_case=ignore_case).all_found
            assert search_in_order  # the only possible cause of match-not-found ValueError
            return EntitiesOverlappingOutput(overlap=True)
        if search_in_order:  # in this case, each entity always have 1 match by construction
            spans = [m.span('keyword') for m in ms]
            assert spans == sorted(spans)  # when searching in order, spans should be sorted already
            overlap = spans_overlap(spans=spans)
        else:
            # in this case, each entity may have multiple matches
            #   so try all possibilities, if one of them has no overlap, then return no overlap
            # sanity check at least 1 match for each entity
            assert all(len(matches) > 0 for matches in et_ms)
            # each element is the potential span config for all entities
            #   first, add the candidate spans for the 1st entity
            enm1, ms1 = entity_names[0], et_ms[0]
            spans_pool = [[m.span('keyword')] for m in ms1]  # duplicate the pool for each match span
            for matches in et_ms[1:]:
                # if K spans for the current entity, duplicate the current pool K times,
                #   and add append each of the K spans to each of the candidate spans
                n_mch = len(matches)
                if n_mch > 1:
                    spans_pool_ = []
                    for m in matches:  # duplicate the pool, add a different match span for each duplicate
                        spans_pool_ += [spans + [m.span('keyword')] for spans in spans_pool]
                    spans_pool = spans_pool_
                else:
                    # add the single match span to each of the candidate spans
                    mch = matches[0]
                    spans_pool = [spans + [mch.span('keyword')] for spans in spans_pool]
            # check if any of the candidate spans has no overlap
            overlap = all(spans_overlap(spans=spans) for spans in spans_pool)
        return EntitiesOverlappingOutput(overlap=overlap, matches=ms, entity_matches=et_ms, entity_spans=et_spans)


if __name__ == '__main__':
    from stefutil import sic

    def check_overlap():
        from dataclasses import asdict

        sent = "I can't make up my mind if I want to watch a comedy or a romantic comedy, what do you suggest?"
        enms = ['comedy', 'romantic comedy']
        ovl = entities_overlapping(sentence=sent, entity_names=enms, ignore_case=True, search_in_order=False)
        sic(asdict(ovl))
    # check_overlap()

    def check_found():
        sent = "Is there a movie featuring the song 'Can't Stop the Feeling!' by Justin Timberlake?"
        enms = ["Can't Stop the Feeling!", "Justin Timberlake"]
        out = entities_in_sentence(sentence=sent, entity_names=enms, ignore_case=True)
        sic(out)

        mch = patterns.find_match(text=sent, keyword=enms[0], ignore_case=True, strict=False)
        sic(mch)
    check_found()
