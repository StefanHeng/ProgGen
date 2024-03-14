import json
import random
from os.path import join as os_join
from copy import deepcopy
from typing import List, Dict, Tuple, Iterable, Union, Any
from dataclasses import dataclass
from collections import Counter, defaultdict

from stefutil import *
from src.util.util_ import *
from src.util import patterns


__all__ = [
    'unpretty', 'detokenize',
    'ner_labels2tag2index', 'ner_labels2tags',
    'split_text_with_terms',

    'check_single_char_appearance',
    'NerBioExample', 'NerReadableExample', 'NerSpanExample', 'NerExample', 'bio2readable', 'readable2bio',
    'ReadableBioTag', 'DatasetLoader',
]


_logger = get_logger(__name__)


def unpretty(txt: str) -> str:
    """
    ensures each whole word is split into distinct tokens, drops the changes for human readability from `tokens2sentence`
    """
    return ' '.join(punc_tokenize(txt))


def detokenize(tokens: Iterable[str]) -> str:
    """
    Join tokens into a sentence
    """
    if not hasattr(detokenize, 'detokenizer'):
        import sacremoses  # lazy import to save time
        detokenize.detokenizer = sacremoses.MosesDetokenizer()
    return detokenize.detokenizer.detokenize(tokens)


def find_keyword_positions(text, keyword) -> List[Tuple[str, int]]:
    positions = []
    start = 0
    while start < len(text):
        start = text.find(keyword, start)
        if start == -1:
            break
        positions.append((keyword, start))
        start += len(keyword)
    return positions


@dataclass
class SplitTermsOutput:
    segments: List[str] = None
    tokens: List[str] = None
    labels: List[str] = None


def split_text_with_terms(
        input_text: str, term_list: Union[List[str], Tuple[str]], terms_n_types: List[Tuple[str, str]],
        ignore_case: bool = False, strict: bool = False
) -> SplitTermsOutput:
    """
    :param input_text: Text to split
    :param term_list: List of terms to split `input_text` by, should be in order of appearance in `input_text`
    :param terms_n_types: List of (term, entity type) pairs, ordered by term occurrence
            A dictionary may not work since the same term may appear multiple times and have different entity types
    :param ignore_case: If true, ignore casing when searching for terms in `input_text`
    :param strict: If true, raise error if any term in `term_list` is not found in `input_text`
    """
    if ignore_case:
        assert all(t.casefold() in input_text.casefold() for t in term_list)
    else:
        assert all(t in input_text for t in term_list)

    segments = []
    if strict:  # ensure each term is found in `input_text`
        # partition `input_text` into segments by `term_list`, in that order
        for term in term_list:
            start, matched, input_text = patterns.partition(input_text, term, ignore_case=ignore_case)
            if start.strip():
                segments.append(start.strip())
            segments.append(matched)
        if input_text.strip():
            segments.append(input_text.strip())
    else:
        if ignore_case:
            raise NotImplementedError
        input_text = deepcopy(input_text)
        positions = []
        for term in term_list:
            if term and term in input_text:
                position = find_keyword_positions(input_text, term)
                positions += position

        terms_to_split = sorted(positions, key=lambda x: x[-1])  # Sort terms by length to avoid partial matches
        while input_text:
            term_found = False
            for term in terms_to_split:
                if term[0] in input_text:
                    before, matched, input_text = input_text.partition(term[0])
                    before = before.strip()
                    if before:
                        # output.append(before.strip())
                        segments.append(before)
                    segments.append(matched.strip())
                    term_found = True
                    break

            if not term_found:
                segments.append(input_text.strip())
                break

    labels = []
    tokens = []
    terms_n_types = deepcopy(terms_n_types)
    if ignore_case:
        terms_n_types = [(enm.casefold(), et) for enm, et in terms_n_types]

    _n_tag_added = 0
    for seg in segments:
        split_tokens = punc_tokenize(seg)  # split on all punctuations
        tokens += split_tokens

        if ignore_case:
            seg = seg.casefold()
        seg_is_entity = seg == terms_n_types[0][0] if len(terms_n_types) > 0 else False
        if seg_is_entity:
            # pop the 1st element one by one, can do this since `segments` and `terms_n_types` are both in occurrence order
            (enm, et) = terms_n_types.pop(0)
            labels += [f"I-{et}" if i != 0 else f"B-{et}" for i in range(len(split_tokens))]
            _n_tag_added += 1
        else:
            labels += ["O" for _ in range(len(split_tokens))]
    assert all(t != '' for t in tokens)  # sanity check
    if _n_tag_added != len(term_list):
        sic(_n_tag_added, len(term_list), term_list, terms_n_types, input_text, segments)
    # sanity check all entity names are indeed added; exhausted all terms
    assert _n_tag_added == len(term_list) and len(terms_n_types) == 0
    return SplitTermsOutput(segments=segments, tokens=tokens, labels=labels)


def ner_labels2tag2index(entity_types: List[str] = None) -> Dict[str, int]:
    """
    :param entity_types: List of NER entity type labels in readable format
    :return: Dictionary mapping NER entity type labels in BIO format to index
    """
    ret = dict(O=0)
    idx = 1
    for n in entity_types:
        ret[f"B-{n}"] = idx
        idx += 1
        ret[f"I-{n}"] = idx
        idx += 1
    return ret


def ner_labels2tags(entity_types: List[str]) -> List[str]:
    ret = ['O']
    for n in entity_types:
        ret += [f'B-{n}', f'I-{n}']
    return ret


@dataclass(eq=True, frozen=True)
class NerExample:
    sentence: str = None

    def get_entity_types(self) -> Union[Tuple[str], List[str]]:
        """
        :return: entity types in the sample in occurrence order
        """
        raise NotImplementedError

# NerExample = Union[NerBioExample, NerReadableExample, NerSpanExample]


# Ending w/ `Example` are ready for training
@dataclass(eq=True, frozen=True)
class NerBioExample(NerExample):
    sentence: str = None
    tokens: Tuple[str] = None
    ner_tags: Tuple[str] = None

    @classmethod
    def from_tokens_n_tags(cls, tokens: List[str], tags: List[str]) -> 'NerBioExample':
        return cls(sentence=detokenize(tokens), tokens=tuple(tokens), ner_tags=tuple(tags))

    @classmethod
    def from_json(cls, d: Union[str, Dict[str, Any]]) -> 'NerBioExample':
        if isinstance(d, str):
            d = json.loads(d)
        ner_tags = d.get('ner_tags', d.get('labels'))  # get tags from key `labels` or `ner_tags`
        tokens = d['tokens']
        return cls.from_tokens_n_tags(tokens=tokens, tags=ner_tags)

    def get_entity_types(self) -> List[str]:
        return [t[2:] for t in self.ner_tags if t.startswith('B-')]

    def to_readable(self) -> 'NerReadableExample':
        return bio2readable(self)

    def get_entity_span_indices(self) -> List[Tuple[int, int]]:
        """
        get the inclusive starting and ending indices corresponding to each entity span
        """
        start_idxs = [i for i, tag in enumerate(self.ner_tags) if tag.startswith('B-')]
        end_idxs = []
        n_tag = len(self.ner_tags)
        for start_idx in start_idxs:
            tag = self.ner_tags[start_idx][2:]
            found = False
            for i in range(start_idx + 1, n_tag):
                if self.ner_tags[i] != f'I-{tag}':  # found the end of the span
                    end_idxs.append(i - 1)
                    found = True
                    break
            if not found:  # the last token is part of an entity span
                end_idxs.append(n_tag - 1)
        entity_idxs = list(zip(start_idxs, end_idxs))

        assert len(entity_idxs) == len(start_idxs) == len(end_idxs)  # sanity check all entity annotations are found
        for (s1, e1), (s2, e2) in zip(entity_idxs[:-1], entity_idxs[1:]):  # sanity check entity spans are increasing & non-overlapping
            assert s1 <= e1 < s2 <= e2
        return entity_idxs


@dataclass
class MultiOccurEntityInfo:
    has_multi_occur_entity: bool = None
    has_multi_occur_entity_diff_type: bool = None
    multi_occur_entity_names: List[str] = None
    multi_occur_entity_types: List[str] = None


@dataclass(eq=True, frozen=True)
class NerReadableExample(NerExample):
    sentence: str = None
    entity_names: Tuple[str, ...] = None
    entity_types: Tuple[str, ...] = None

    @classmethod
    def from_d(cls, d: Union[str, Dict[str, Any]] = None, **kwargs) -> 'NerReadableExample':
        ret = dict()
        if isinstance(d, str):
            d = json.loads(d)
        if d:
            ret.update(d)
        if kwargs:
            ret.update(kwargs)
        # entity names and types can be empty, but sentence should be provided
        assert {'sentence'} <= set(ret.keys()) <= {'sentence', 'entity_names', 'entity_types'}
        ret['entity_names'] = tuple(ret.get('entity_names', []))
        ret['entity_types'] = tuple(ret.get('entity_types', []))
        return cls(**ret)

    def to_split_on_puncs(self) -> 'NerReadableExample':
        """
        Split sentences and entity names on punctuation marks, intended for string comparison w/ `NerBioExample`s
        """
        # so that entities should still be part of the sentence
        sent, enms = unpretty(self.sentence), [unpretty(en) for en in self.entity_names]
        return NerReadableExample(sentence=sent, entity_names=tuple(enms), entity_types=self.entity_types)

    def get_entity_types(self) -> Tuple[str]:
        return self.entity_types

    def to_lower(self) -> 'NerReadableExample':
        """
        Convert all text to lower case
        """
        sent, enms_ = self.sentence.lower(), [en.lower() for en in self.entity_names]
        return NerReadableExample(sentence=sent, entity_names=tuple(enms_), entity_types=self.entity_types)

    def get_multi_occur_entity_info(self) -> MultiOccurEntityInfo:
        enm2c = Counter(self.entity_names)
        multi_occur_enms = [enm for enm, cnt in enm2c.items() if cnt > 1]

        enm2ets = defaultdict(set)
        for enm, et in zip(self.entity_names, self.entity_types):
            enm2ets[enm].add(et)
        multi_occur_enms_diff_type = [enm for enm, ets in enm2ets.items() if len(ets) > 1]

        idx_multi = [i for i, enm in enumerate(self.entity_names) if enm in multi_occur_enms]
        enm_multi = [self.entity_names[i] for i in idx_multi]
        et_multi = [self.entity_types[i] for i in idx_multi]
        return MultiOccurEntityInfo(
            has_multi_occur_entity=len(multi_occur_enms) > 0,
            has_multi_occur_entity_diff_type=len(multi_occur_enms_diff_type) > 0,
            multi_occur_entity_names=enm_multi,
            multi_occur_entity_types=et_multi
        )

    def to_bio(self, ignore_case: bool = None) -> NerBioExample:
        """
        Convert to BIO format
        """
        return readable2bio(self, ignore_case=ignore_case)


@dataclass
class NerSpan:
    content: str = None
    entity_type: str = None


@dataclass(eq=True, frozen=True)
class NerSpanExample(NerExample):
    sentence: str = None
    spans: List[NerSpan] = None

    @property
    def entity_types(self) -> List[str]:
        return [s.entity_type for s in self.spans]


def bio2readable(
        x: NerBioExample, label_map: Dict[str, str] = None, join_tokens_by_space: bool = False, use_original_sentence: bool = True
) -> NerReadableExample:
    """
    Convert token-level NER data in BIO format to more human-readable natural language format
    """
    toks, tags = x.tokens, x.ner_tags
    assert len(toks) == len(tags)

    entity_names: List[str] = []
    entity_types: List[str] = []
    for i, t in enumerate(tags):
        if t == 'O':
            continue
        prefix, lb = t.split('-')
        if prefix == 'B':
            entity_types.append(lb)
            entity_names.append(toks[i])
        else:
            assert prefix == 'I'
            assert entity_types[-1] == lb
            entity_names[-1] += ' ' + toks[i]
    if label_map:
        entity_types = [label_map[et] for et in entity_types]
    sent = x.sentence if use_original_sentence else detokenize(toks)
    return NerReadableExample(sentence=sent, entity_names=tuple(entity_names), entity_types=tuple(entity_types))


def check_single_char_appearance(entity_names: Union[List[str], Tuple[str]] = None, sentence: str = None) -> bool:
    # for an edge case: `A` is an annotated entity and `a` appears in the sentence, e.g.
    #   `1. "What is the plot of the Clint Eastwood movie A that has a fantasy quest theme?"
    #   Named Entities: [Clint Eastwood (Actor), Fantasy quest (Genre), A (Title)]`
    #   `2. "I'm looking for a movie with a soundtrack as powerful as M"
    #   Named Entities: [powerful (Review), M (Title)]`
    for c_, c in [('A', 'a'), ('M', 'm'), ('S', 's')]:
        if c_ in entity_names and len(patterns.find_match(text=sentence, keyword=c)) > 0:
            return True
    return False


# a version w/o edge case logging for `readable2bio`
#   see the complete version in `src/dataset_util/generate.py`
def _entities_differ_in_case_only(entity_names: Union[List[str], Tuple[str]] = None, sentence: str = None) -> bool:
    # if 'A' in entity_names and len(find_match(text=sentence, keyword='a')) > 0:
    if check_single_char_appearance(entity_names=entity_names, sentence=sentence):
        return True
    en_lc = [e.lower() for e in entity_names]
    return len(entity_names) == len(set(en_lc)) and len(entity_names) != len(set(entity_names))


def readable2bio(x: NerReadableExample, ignore_case: bool = None) -> NerBioExample:
    if ignore_case is None:
        ignore_case = not _entities_differ_in_case_only(entity_names=x.entity_names, sentence=x.sentence)

    enms = [unpretty(n) for n in x.entity_names]

    enm2ets = defaultdict(set)
    for enm, et in zip(enms, x.entity_types):
        enm2ets[enm].add(et)
    lst_terms = list(zip(enms, x.entity_types))
    split_args = dict(input_text=unpretty(x.sentence), term_list=enms, terms_n_types=lst_terms, ignore_case=ignore_case, strict=True)
    out = split_text_with_terms(**split_args)
    tokens, ner_tags = out.tokens, out.labels

    # if len(tokens) != len(punc_tokenize(x.sentence)):
    #     sic(x)
    #     sic(tokens, ner_tags, punc_tokenize(x.sentence))
    #     sic(len(tokens), len(ner_tags), len(punc_tokenize(x.sentence)))
    assert len(tokens) == len(ner_tags) == len(punc_tokenize(x.sentence))  # sanity check
    return NerBioExample(sentence=x.sentence, tokens=tuple(tokens), ner_tags=tuple(ner_tags))


_none_entity_type = '__none__'


def bio2consecutive_spans(x: NerBioExample) -> NerSpanExample:
    """
    Intended for demo in format `natural-inline`

    breaks down NER-marked sentences into consecutive spans
    """
    toks, tags = x.tokens, x.ner_tags
    assert len(toks) == len(tags)
    ret = []
    curr_content, curr_type = '', _none_entity_type
    prev_type = None
    n_tok = len(toks)
    for i, (tok, tag) in enumerate(zip(toks, tags)):
        if tag == 'O':
            curr_type = _none_entity_type
        else:
            prefix, lb = tag.split('-')
            assert prefix in ('B', 'I')
            curr_type = lb
        if curr_type != prev_type:
            if curr_content:
                ret.append(NerSpan(content=curr_content, entity_type=prev_type))
            curr_content = tok
        else:
            sep = ' '
            if tok == '.' and i + 1 == n_tok:
                sep = ''  # appending last period token, no whitespace
            curr_content += sep + tok
        prev_type = curr_type
    if curr_content:
        ret.append(NerSpan(content=curr_content, entity_type=prev_type))
    for r in ret:
        if r.entity_type == _none_entity_type:
            r.entity_type = None
    return NerSpanExample(sentence=x.sentence, spans=ret)


class ReadableBioTag:
    """
    Map BIO tag labels to natural-language version
    """
    def __init__(self, label_map: Dict[str, str] = None):
        self.label_map = label_map

    def __call__(self, tag: str) -> str:
        if tag == 'O':
            return tag
        prefix, lb = tag.split('-')
        if prefix == 'B':
            return f'B-{self.label_map[lb]}'
        else:
            assert prefix == 'I'
            return f'I-{self.label_map[lb]}'


def _drop_misc(tags: List[str]) -> List[str]:
    """
    swap `MISC` tags to `O` tags
    """
    return [t if 'MISC' not in t else 'O' for t in tags]


@dataclass
class SamplePairOutput:
    n_shot_samples: List[NerExample] = None
    n_samples: List[NerExample] = None


class DatasetLoader:
    from_hf = ['conll2003', 'conll2003-no-misc', 'ncbi-disease']
    from_local_bio_file = ['job-desc', 'job-stack', 'mit-movie', 'mit-restaurant', 'wiki-gold', 'wiki-gold-no-misc']

    def __init__(
            self, dataset_name: str = 'conll2003', split: str = 'train', data_format: str = 'readable', join_tokens_by_space: bool = None,
            cache_n_shot: bool = True, cache_get_n: bool = True
    ):
        ca(dataset_name=dataset_name, original_data_format=data_format)
        import datasets  # lazy import to save time

        self.dataset_name = dataset_name
        self.split = split
        d_dset = sconfig(f'datasets.{dataset_name}')
        self.labels, self.readable_labels = d_dset['entity-types'], d_dset['readable-entity-types']
        self.label_map = d_dset.get('label2readable-label')
        self.rbt = ReadableBioTag(label_map=self.label_map) if self.label_map is not None else None
        if dataset_name in DatasetLoader.from_hf:
            ca.assert_options(display_name='Dataset split', val=split, options=['train', 'validation', 'test'])
            dnm_load = dataset_name
            if dnm_load.endswith('-no-misc'):
                dnm_load = dnm_load[:-len('-no-misc')]
            dnm_load = dnm_load.replace('-', '_')
            self.dset: datasets.Dataset = datasets.load_dataset(dnm_load)[split]

            self.tags = self.dset.features['ner_tags'].feature.names
            if dataset_name == 'conll2003-no-misc':  # following prior papers, drop the MISC category
                self.tags = _drop_misc(self.tags)
            self.i2l = self.dset.features['ner_tags'].feature.int2str
        elif dataset_name in DatasetLoader.from_local_bio_file:
            ca.assert_options(display_name='Dataset split', val=split, options=['train', 'val', 'test'])
            dir_nm = dataset_name
            if dataset_name == 'mit-movie':
                dir_nm = os_join('mit-movie', 'eng')  # use the `eng` partition
            elif dataset_name == 'wiki-gold-no-misc':
                dir_nm = 'wiki-gold'
            path = os_join(pu.proj_path, 'original-dataset', dir_nm, f'{split}.jsonl')

            self.tags = ner_labels2tags(entity_types=self.labels)
            self.dset: List[NerBioExample] = self._load_from_jsonl(path=path)
        else:
            raise NotImplementedError
        self.join_tokens_by_space = join_tokens_by_space if join_tokens_by_space is not None else False

        self.cache_n_shot = cache_n_shot
        self.cache_get_n = cache_get_n
        self.cache_n_shot_args2samples = dict() if cache_n_shot else None
        self.cache_get_n_args2samples = dict() if cache_get_n else None

        self.data_format = data_format
        d_log = {
            'dataset-name': dataset_name, 'split': split, 'data-format': data_format,
            'labels': self.labels, 'readable-labels': self.readable_labels, 'label-map': self.label_map, '#labels': len(self.labels),
            'ner-tags': self.tags, '#tags': len(self.tags),
            'join-tokens-by-space': join_tokens_by_space
        }
        _logger.info(f'Constructed {pl.i(self.__class__.__name__)} w/ {pl.i(d_log, indent=1)}')

    def _load_from_jsonl(self, path: str) -> List[NerBioExample]:
        with open(path, 'r') as f:
            data = [json.loads(ln) for ln in f.readlines()]

        def _load_single(d: Dict[str, Any]) -> NerBioExample:
            if self.rbt:
                tags = d['ner_tags']
                if self.dataset_name == 'wiki-gold-no-misc':
                    tags = _drop_misc(tags)
                d['ner_tags'] = [self.rbt(t) for t in tags]
            d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
            return NerBioExample.from_json(d)
        return [_load_single(d) for d in data]

    def __len__(self):
        return len(self.dset)

    def _get_single(self, idx: int) -> NerExample:
        if self.dataset_name in DatasetLoader.from_hf:
            import datasets  # lazy import to save time
            self.dset: datasets.Dataset
            entry: Dict[str, Any] = self.dset[idx]
            toks = entry['tokens']
            tags: List[str] = [self.i2l(i) for i in entry['ner_tags']]

            if self.dataset_name == 'conll2003-no-misc':
                tags = _drop_misc(tags)

            tags = [self.rbt(t) for t in tags]
            bio = NerBioExample.from_tokens_n_tags(tokens=toks, tags=tags)
        elif self.dataset_name in DatasetLoader.from_local_bio_file:
            bio = self.dset[idx]
        else:
            raise NotImplementedError(f'dataset name={self.dataset_name}')
        if self.data_format == 'readable':
            # to human-readable types
            readable = bio2readable(bio, join_tokens_by_space=self.join_tokens_by_space)
            return readable
        elif self.data_format == 'bio':
            # to human-readable types
            return bio
        else:
            assert self.data_format == 'span'
            eg = bio2consecutive_spans(bio)
            return eg

    def __getitem__(self, idxs: Union[int, List[int], Tuple[int], slice]) -> Union[NerExample, List[NerExample]]:
        if isinstance(idxs, int):
            return self._get_single(idxs)
        elif isinstance(idxs, slice):
            return [self._get_single(i) for i in range(*idxs.indices(len(self)))]
        else:
            assert isinstance(idxs, (list, tuple))
            return [self._get_single(i) for i in idxs]

    def get_n(
            self, n: int = 100, return_samples: bool = True, exclude: List[int] = None, shuffle: bool = True, seed: int = 42
    ) -> Union[List[int], List[NerExample]]:
        """
        Get random samples from the dataset
            Intended for getting random subsets except n-shot demo samples that will be served as unlabeled sentences
        """
        if self.cache_get_n:
            exclude_ = tuple(exclude) if exclude else None
            k = (n, return_samples, exclude_, shuffle, seed)
            if k in self.cache_get_n_args2samples:
                return self.cache_get_n_args2samples[k].copy()  # so that the cached version is not modified
            else:
                ret = self._get_n(n=n, return_samples=return_samples, exclude=exclude, shuffle=shuffle, seed=seed)
                self.cache_get_n_args2samples[k] = ret
                return ret.copy()
        else:
            return self._get_n(n=n, return_samples=return_samples, exclude=exclude, shuffle=shuffle, seed=seed)

    def _get_n(
            self, n: int = 100, return_samples: bool = True, exclude: List[int] = None, shuffle: bool = True, seed: int = 42
    ) -> Union[List[int], List[NerExample]]:
        idxs = list(range(len(self)))
        if shuffle:
            random.seed(seed)
            random.shuffle(idxs)
            random.seed()
        ret = idxs[:n]
        if exclude:
            assert all(isinstance(i, int) for i in exclude)  # sanity check
            exclude = set(exclude)
            ret = [i for i in ret if i not in exclude]

            i_idx = n
            while len(ret) < n:
                idx = idxs[i_idx]
                if idx not in exclude:
                    ret.append(idx)
                i_idx += 1
        return self[ret] if return_samples else ret

    def n_shot(
            self, n: int = 5, return_samples: bool = True, include_none_samples: bool = False, shuffle: bool = True, seed: int = 42
    ) -> Union[List[int], List[NerExample]]:
        """
        Gets n-shot examples for each entity type

        Algorithm modified from Template-free Prompt Tuning for Few-shot NER, in NAACL 2022
            We don't shuffle the dataset for a more realistic setting, the first n samples are selected

        Exactly n occurrences will be seen for each entity type,
            so output length can be smaller than (n x #entity-types), when a single sentence has multiple entity types

        :param n: Number of occurrences for each entity type
        :param return_samples: If true, return the actual samples, otherwise return the indices
        :param include_none_samples: If true, include samples w/ no entity types, capped at n
        :param shuffle: If true, shuffle the dataset before selecting samples
        :param seed: Random seed for shuffling
        """
        if self.cache_n_shot:
            k = (n, return_samples, include_none_samples, shuffle, seed)
            if k in self.cache_n_shot_args2samples:
                return self.cache_n_shot_args2samples[k].copy()
            else:
                ret = self._n_shot(
                    n=n, return_samples=return_samples, include_none_samples=include_none_samples, shuffle=shuffle, seed=seed)
                self.cache_n_shot_args2samples[k] = ret
                return ret.copy()
        else:
            return self._n_shot(n=n, return_samples=return_samples, include_none_samples=include_none_samples, shuffle=shuffle, seed=seed)

    def _n_shot(
            self, n: int = 5, return_samples: bool = True, include_none_samples: bool = False, shuffle: bool = False, seed: int = 42
    ):
        idxs = list(range(len(self)))
        if shuffle:
            # **ideally**, we don't do this cos having access to the entire labeled dataset is unrealistic
            random.seed(seed)
            random.shuffle(idxs)
            random.seed()
        ret = []
        et2added_count = {et: 0 for et in self.readable_labels}
        if include_none_samples:
            et2added_count[_none_entity_type] = 0
        for i in idxs:
            eg = self[i]
            c = Counter(eg.get_entity_types())
            if include_none_samples:  # TODO: check in, is this a setting we can try?
                add = True
                if len(c) == 0:
                    c[_none_entity_type] = 1  # this ensures cap at n for samples w/ no entity types
            else:
                add = len(c) > 0
            add = add and all(et2added_count[et] + n_et <= n for et, n_et in c.items())
            if add:
                ret.append(i)
                for et, n_et in c.items():
                    et2added_count[et] += n_et
            if all(v == n for v in et2added_count.values()):
                break
        i2ets = {i: self[i].get_entity_types() for i in ret}
        c = Counter(sum((list(v) for v in i2ets.values()), start=[]))
        assert all(v == n for v in c.values())  # sanity check
        if include_none_samples:
            assert sum(len(ets) == 0 for ets in i2ets.values()) == n  # sanity check
        return self[ret] if return_samples else ret

    def get_few_demo_samples(
            self, n_demo: int = 5, demo_type: str = 'n-shot', shuffled_n_shot: bool = True, shuffle: bool = False, **kwargs
    ) -> Union[List[int], List[NerExample], List[NerReadableExample]]:
        ca.assert_options(display_name='Demo Sample Retrieval Type', val=demo_type, options=['n', 'n-shot'])
        if demo_type == 'n-shot':
            samples = self.n_shot(n=n_demo, shuffle=shuffled_n_shot, **kwargs)
        else:
            assert demo_type == 'n'
            samples = self[:n_demo]
        if shuffle:
            random.shuffle(samples)  # shuffle order for diversity
        return samples

    def get_few_demo_and_n_samples(self, n_shot_args: Dict[str, Any] = None, n_args: Dict[str, Any] = None) -> SamplePairOutput:
        """
        Get n-shot demo samples and some more randomly shuffled samples
        """
        d = dict(shuffle=True, return_samples=False)
        n_shot_args_ = d.copy()
        n_shot_args_.update(n_shot_args or dict())
        idxs = self.get_few_demo_samples(**n_shot_args_)

        n_args_ = {**dict(n=100, exclude=idxs), **d}
        n_args_.update(n_args or dict())
        idxs_n = self.get_n(**n_args_)
        return SamplePairOutput(n_shot_samples=self[idxs], n_samples=self[idxs_n])


if __name__ == '__main__':
    sic.output_width = 128

    # dnm = 'conll2003'
    # dnm = 'conll2003-no-misc'
    # dnm = 'job-desc'
    # dnm = 'mit-movie'
    dnm = 'mit-restaurant'
    # dnm = 'job-stack'
    # dnm = 'wiki-gold'
    # dnm = 'wiki-gold-no-misc'
    # dnm = 'ncbi-disease'
    inc_n = dnm not in ['mit-movie', 'mit-restaurant']

    def check_load():
        d_fmt = 'readable'
        # d_fmt = 'bio'
        # d_fmt = 'span'
        cl = DatasetLoader(dataset_name=dnm, data_format=d_fmt)
        sic(cl.labels, cl.readable_labels, cl.label_map, cl.tags)
        sic(cl[:5])
    # check_load()

    def check_local_dset_load():
        # fmt = 'readable'
        fmt = 'bio'
        dl = DatasetLoader(dataset_name=dnm, split='train', data_format=fmt)
        n = 5
        samples = dl[:n]
        sic(samples)

        # until how much samples can I see an example of each entity type?
        check_entity_coverage = False
        # check_entity_coverage = True
        if check_entity_coverage:
            assert fmt == 'readable'
            ets = dl.readable_labels

            def check_cover(data, best: int = None):
                data: List[NerReadableExample]
                ets_shown = {et for eg in data for et in eg.entity_types}
                curr = len(set(ets) & ets_shown)
                update = False
                if curr > best:
                    best = curr
                    update = True
                return sorted(ets_shown), best, update

            n_sample, samples = 0, []
            covered, n_cover, upd = check_cover(samples, best=0)
            while len(covered) < len(ets):
                n_sample += 1
                samples = dl[:n_sample]
                covered, n_cover, upd = check_cover(samples, best=n_cover)
                if upd:
                    sic(n_sample, n_cover, covered)
    # check_local_dset_load()

    def check_n_shot():
        # sh = False
        sh = True
        d_fmt = 'readable'
        # d_fmt = 'bio'
        dl = DatasetLoader(dataset_name=dnm, split='train', data_format=d_fmt)

        # n = 5
        # n = 2
        n = 1
        samples = dl.n_shot(n=n, return_samples=True, include_none_samples=inc_n, shuffle=sh, seed=42)
        sic(samples, len(samples))
    # check_n_shot()

    def check_get_n():
        dl = DatasetLoader(dataset_name=dnm, split='train', data_format='readable')
        n_shot_idxs = dl.n_shot(n=1, include_none_samples=inc_n, return_samples=False)
        unlabeled_idxs = dl.get_n(n=100, exclude=n_shot_idxs, return_samples=False)
        # sic(n_shot_idxs, unlabeled_idxs)
        assert set(n_shot_idxs) & set(unlabeled_idxs) == set()  # sanity check mutually exclusive
        assert set(dl[n_shot_idxs]) & set(dl[unlabeled_idxs]) == set()
        sic(len(n_shot_idxs), len(unlabeled_idxs))
        sic(dl[unlabeled_idxs[:10]])
    # check_get_n()

    def check_ncbi_broken():
        import datasets
        dset = datasets.load_dataset('ncbi_disease', split='test')
        sample = dset[940]
        sic(sample)

        dset = DatasetLoader(dataset_name='ncbi-disease', split='test', data_format='bio')
        sample = dset[540]
        # samples = dset[:]
        # for i, eg in enumerate(samples):
        #     if eg.sentence == '':
        #         sic(i, eg)
        # assert all(eg.sentence != '' for eg in samples)
        # sample = samples[940]
        sic(sample)
    # check_ncbi_broken()

    def check_quote():
        """
        Check if a dataset contains sentences & named entities w/ enclosing quotes
        """
        from src.util.sample_check import has_punc_on_edge

        dl = DatasetLoader(dataset_name=dnm, split='test', data_format='readable')

        n_sent_has_quote = 0
        n_en_has_quote = 0
        for sample in dl:
            sample: NerReadableExample
            if has_punc_on_edge(sample.sentence).on_both_side:
                sic(sample.sentence)
                n_sent_has_quote += 1

            ens_w_quote = [en for en in sample.entity_names if has_punc_on_edge(en).on_both_side]
            if len(ens_w_quote) > 0:
                sic(sample.sentence, ens_w_quote)
                n_en_has_quote += 1
        sic(len(dl), n_sent_has_quote, n_en_has_quote)
    check_quote()

    def check_readable2bio():
        eg = NerReadableExample(
            sentence="what year did the director of the movie 'c' win an award for best foreign film?",
            entity_names=('c', 'best foreign film'), entity_types=('Title', 'Review')
        )
        bio = eg.to_bio()
        sic(bio)
    # check_readable2bio()
