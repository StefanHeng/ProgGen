"""
Compute statistics on synthetic datasets from `data_generation`
"""

import os
import json
import logging
from os.path import join as os_join
from typing import List, Dict, Set, Union, Any
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from stefutil import get_logger, to_percent, describe, ca, sic, pl, punc_tokenize, stem
from src.util import pu, sconfig
from src.util.ner_example import detokenize, NerReadableExample, bio2readable, NerBioExample
from src.data_util.prettier import highlight_span_in_sentence, sdpc


__all__ = ['get_word_weights', 'NerDatasetStats']


_logger = get_logger('Dataset Stats')


def get_word_weights(documents: List[str]) -> Dict[str, float]:
    """
    Given a list of documents, get weights of words via TF-IDF
    """
    from stefutil import TextPreprocessor

    lst_toks = TextPreprocessor()(documents)

    from sklearn.feature_extraction.text import TfidfVectorizer  # lazy import to save time
    vect = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=3)
    mat = vect.fit_transform((' '.join(toks) for toks in lst_toks))
    words = vect.get_feature_names_out()

    scores = np.asarray(mat.sum(axis=0)).squeeze().tolist()  # Sum the scores across all documents
    assert len(scores) == len(words)  # sanity check
    return {w: s for w, s in zip(words, scores)}


def dataset2test_file_name(dataset_name: str = None) -> str:
    # if 'conll2003' in dataset_name:
    #     return 'test'
    if dataset_name in ['job-desc']:
        return 'test-all'
    elif dataset_name in ['job-stack', 'wiki-gold', 'wiki-gold-no-misc', 'mit-movie', 'mit-restaurant']:
        return 'bio-test-all'
    elif dataset_name in ['ncbi-disease'] or 'conll2003' in dataset_name:
        return 'bio-test'
    else:
        raise NotImplementedError


def dataset_dir_name2path(dataset_name: str = 'conll2003', dir_name: str = None, sub_dir: str = None) -> str:
    if any(k in dir_name for k in ['test', 'train', 'val']):  # original dataset
        return os_join(pu.proj_path, 'original-dataset', dataset_name, f'{dir_name}.jsonl')
    else:
        _dnm = dataset_name.replace('-', '_')
        if sub_dir is not None:
            _dnm = os_join(_dnm, sub_dir)
        return os_join(pu.proj_path, 'generated_data', _dnm, dir_name, 'readable-train.json')


def word_overlap(source_corpus: List[str], target_corpus: List[str], pretty: bool = True, top_n: int = 20) -> Dict[str, Any]:
    """
    Given a source corpus and a target corpus, compute the word overlap between them
    """
    # compute weights of words respectively
    w_s, w_t = get_word_weights(source_corpus), get_word_weights(target_corpus)
    s_s, s_t = set(w_s.keys()), set(w_t.keys())
    inter = s_s.intersection(s_t)
    # get overlap w.r.t each corpus
    ratio_s = sum(w_s[w] for w in inter) / sum(w_s.values())
    ratio_t = sum(w_t[w] for w in inter) / sum(w_t.values())
    ret: Dict[str, Any] = dict(ratio_s=ratio_s, ratio_t=ratio_t, harmonic_mean=2 * ratio_s * ratio_t / (ratio_s + ratio_t))
    if pretty:
        ret = {k: round(v * 100, 2) for k, v in ret.items()}
    if top_n:
        ret[f'source top {top_n} words'] = sorted(w_s, key=w_s.get, reverse=True)[:top_n]
        ret[f'target top {top_n} words'] = sorted(w_t, key=w_t.get, reverse=True)[:top_n]
    return ret


def overlapping_words(
        source_corpus: List[str], target_corpus: List[str], top_n: int = 20, top_both: bool = True, show_counts: bool = True
) -> Dict[str, Any]:
    """
    Given 2 corpus, return the top words appeared based on counts
    """
    from stefutil import TextPreprocessor

    tp = TextPreprocessor(tokenize_scheme='chunk')
    toks_s = sum(tp(source_corpus), start=[])
    toks_t = sum(tp(target_corpus), start=[])
    c_s, c_t = Counter(toks_s), Counter(toks_t)
    inter = set(c_s.keys()).intersection(set(c_t.keys()))
    if top_both:
        word2count = {w: (c_s[w], c_t[w]) for w in inter}
        if top_n:  # get top by harmonic mean of counts
            def counts2score(x):
                c1, c2 = word2count[x]
                return 2 * c1 * c2 / (c1 + c2)
                # return min(c1, c2)
            word2count = {w: word2count[w] for w in sorted(inter, key=counts2score, reverse=True)[:top_n]}
        return word2count if show_counts else list(word2count.keys())
    else:
        if top_n:
            inter_s = {w: c_s[w] for w in sorted(inter, key=c_s.get, reverse=True)[:top_n]}
            inter_t = {w: c_t[w] for w in sorted(inter, key=c_t.get, reverse=True)[:top_n]}
        else:
            inter_s = {w: c_s[w] for w in inter}
            inter_t = {w: c_t[w] for w in inter}
        return dict(source_count=inter_s, target_count=inter_t)


@dataclass
class VectorOverlapOutput:
    cosine_sims: np.ndarray = None
    d_log: Dict[str, Any] = None


@dataclass
class StatsSummaryOutput:
    summary: Dict[str, Any] = None
    entity_dist: Dict[str, Any] = None
    test_set_overlap: Dict[str, Any] = None
    test_set_vector_overlap: VectorOverlapOutput = None


class NerDatasetStats:
    def __init__(
            self, dataset_name: str = 'conll2003', dataset_path_or_examples: Union[str, List[NerReadableExample]] = None,
            entity_types: Union[bool, List[str]] = None, show_in_entity_order: bool = True, encode_batch_size: int = None,
            test_stats: 'NerDatasetStats' = None, verbose: bool = False, unique_entity_vector: bool = False, dir_name: str = None
    ):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path_or_examples

        self.dir_name = dir_name
        if isinstance(dataset_path_or_examples, str):
            self.dir_name = stem(os.path.dirname(dataset_path_or_examples))
            if dataset_path_or_examples.endswith('.jsonl'):
                is_ori = True
            else:
                assert dataset_path_or_examples.endswith('.json')
                is_ori = False
            if is_ori:
                dset = [json.loads(line) for line in open(dataset_path_or_examples, 'r')]
            else:
                with open(dataset_path_or_examples, 'r') as f:
                    dset = json.load(f)['examples']

            def map_eg(x: Dict[str, Any]) -> NerReadableExample:
                if is_ori:  # in `bio` format
                    return bio2readable(NerBioExample.from_json(x))
                else:
                    return NerReadableExample(**x)
            self.egs = [map_eg(eg) for eg in dset]
        else:
            assert isinstance(dataset_path_or_examples, list)
            self.egs = dataset_path_or_examples

        if entity_types is True:
            entity_types = sconfig(f'datasets.{dataset_name}.readable-entity-types')
        self._entity_types = entity_types
        self._entity_names = None
        self.show_in_entity_order = show_in_entity_order
        self._entity_hierarchy = None
        self._entity_types_all = sorted(self.entity_hierarchy().keys())
        self._vocab = None

        self.encoder, self._sentence_vectors, self._entity_vectors = None, None, None
        self.unique_entity_vector = unique_entity_vector
        self.encode_batch_size = encode_batch_size or 32

        self._test_stats = test_stats
        self.verbose = verbose

    def encode(self, texts: List[str] = None, sample_kind: str = 'sentence') -> np.ndarray:
        if self.encoder is None:  # lazy load
            self.encoder = SbertEncoder()

        fnm = f'{stem(os.path.dirname(self.dataset_path))}/{stem(self.dataset_path)}'  # use the direct parent directory name
        desc = f'Encoding {pl.i(fnm)} {sample_kind}'
        return self.encoder(texts, batch_size=self.encode_batch_size, desc=desc)

    @property
    def sentence_vectors(self) -> np.ndarray:
        if self._sentence_vectors is None:
            self._sentence_vectors = self.encode(texts=[d.sentence for d in self.egs], sample_kind='sentences')
        return self._sentence_vectors

    @property
    def entity_vectors(self) -> np.ndarray:
        if self._entity_vectors is None:
            if self.unique_entity_vector:
                txts = self.entity_names
            else:  # weigh by occurrence count
                txts = sum([list(d.entity_names) for d in self.egs], start=[])
            self._entity_vectors = self.encode(texts=txts, sample_kind='entities')
        return self._entity_vectors

    @property
    def test_stats(self) -> 'NerDatasetStats':
        if self._test_stats is None:
            dir_nm = dataset2test_file_name(self.dataset_name)
            self._test_stats = NerDatasetStats.from_dir_name(dataset_name=self.dataset_name, dir_name=dir_nm)

            if self.verbose:
                d_log = {'vocab-size': len(self._test_stats.vocab), '#entity-named-tokens': len(self._test_stats.entity_name_toks)}
                _logger.info(f'Test dataset stats initialized w/ {pl.i(d_log)}')
        return self._test_stats

    @test_stats.setter
    def test_stats(self, val: 'NerDatasetStats'):
        self._test_stats = val

    @classmethod
    def from_dir_name(
            cls, dataset_name: str = 'conll2003', dir_name: str = None, path_args: Dict[str, Any] = None, entity_types: bool = True,
            **kwargs
    ) -> 'NerDatasetStats':
        dset_path = dataset_dir_name2path(dataset_name=dataset_name, dir_name=dir_name, **(path_args or dict()))
        return cls(dataset_name=dataset_name, dataset_path_or_examples=dset_path, entity_types=entity_types, **kwargs)

    @property
    def vocab(self) -> Set[str]:
        # distinct, cased words
        if self._vocab is None:
            vocab = set()
            for d in self.egs:
                vocab.update(punc_tokenize(d.sentence))
            self._vocab = vocab
        return self._vocab

    @property
    def entity_name_toks(self) -> Set[str]:
        # distinct, cased words
        toks = set()
        for d in self.egs:
            for en in d.entity_names:
                toks.update(punc_tokenize(en))
        return toks

    def entity_hierarchy(
            self, with_counts: bool = True, alphabetical_order: bool = True, with_context: bool = False, color: bool = True
    ) -> Dict[str, Any]:
        """
        :param with_counts: If True, include counts for each entity
        :param alphabetical_order: If True, sort entity names in alphabetical order
        :param with_context: If True, include the sentence context for each entity
        :param color: If True, highlight the entity in the context
        :return: unique entity types mapped to entity names
        """
        hier = dict()
        if self.show_in_entity_order and self._entity_types is not None:
            for et in self._entity_types:
                hier[et] = [] if with_counts else set()
        for d in self.egs:
            for enm, et in zip(d.entity_names, d.entity_types):
                if with_counts:
                    hier[et].append(enm)
                else:
                    hier[et].add(enm)
        if with_counts:
            # ensures ordering by count
            ret = dict()
            for et, enms in hier.items():
                # if not alphabetical_order:
                #     ret[et] = dict(Counter(enms).most_common())
                # else:
                #     # sort entity of the same count
                #     # first, group these entities by count
                #     lst = defaultdict(list)
                #     for enm, count in Counter(enms).items():
                #         lst[count].append(enm)
                #     # then, sort each group & merge together
                #     ret[et] = sum([sorted(lst[count]) for count in sorted(lst.keys(), reverse=True)], start=[])
                pairs = Counter(enms).most_common()
                if alphabetical_order:  # sort first by count, descending, then by entity name, ascending
                    pairs = sorted(pairs, key=lambda x: (-x[1], x[0]))
                ret[et] = dict(pairs)
            if with_context:
                def highlight(sentence: str = None, entity_span: str = None) -> str:
                    return highlight_span_in_sentence(
                        sentence=sentence, span=entity_span, allow_multi_occurrence=True, pref='[', post=']', color=color, debug_for_test=True)
                # given (entity type => entity names), construct (entity type => entity names => contexts
                ret_ = defaultdict(lambda: defaultdict(list))
                for d in self.egs:
                    for enm, et in zip(d.entity_names, d.entity_types):
                        sent = d.sentence
                        if color:
                            if not (enm in sent or enm.lower() in sent.lower()):
                                # span not found in sentence, must be due to whitespace after sentence `detokenized`
                                enm = detokenize([enm])
                                assert enm in sent or enm.lower() in sent.lower()
                            sent = highlight(sentence=sent, entity_span=enm)
                        ret_[et][enm].append(sent)
                ret = {et: {enm: ret_[et][enm] for enm in enms} for et, enms in ret.items()}  # now keep the same ordering as `ret`
                # if just 1 context, index into the 1-element list
                ret = {et: {enm: ctx[0] if len(ctx) == 1 else ctx for enm, ctx in enms.items()} for et, enms in ret.items()}
        else:
            if not alphabetical_order or with_context:
                raise NotImplementedError
            ret = {k: sorted(v) for k, v in hier.items()}  # will always do alphabetical ordering
        if color:
            ret = {pl.i(et, c='g'): {pl.i(enm, c='b'): ctx for enm, ctx in enms.items()} for et, enms in ret.items()}
        return ret

    @property
    def entity_types(self) -> List[str]:
        if self._entity_types:
            return self._entity_types
        else:
            return sorted(self.entity_hierarchy().keys())

    @property
    def entity_names(self) -> List[str]:
        if not self._entity_names:
            self._entity_names = sorted(set([en for d in self.egs for en in d.entity_names]))
        return self._entity_names

    def entity_counts(self, show_interested_only: bool = False, distinct_entity: bool = False) -> Dict[str, Dict[str, int]]:
        """
        :return: Counts of occurrences of entity types and entity names
        """
        c_enms = Counter([enm for d in self.egs for enm in d.entity_names])
        if not distinct_entity:
            c_ets = Counter([et for d in self.egs for et in d.entity_types])  # counter will be printed in order of occurrence
        else:
            # only count distinct occurrences of each entity type
            et2enms = defaultdict(set)
            for d in self.egs:
                for et, enm in zip(d.entity_types, d.entity_names):
                    et2enms[et].add(enm)
            c_ets = {et: len(enms) for et, enms in et2enms.items()}

        if show_interested_only and self.show_in_entity_order and self._entity_types is not None:
            c_ets = {et: c_ets[et] for et in self._entity_types}
        return dict(entity_types=c_ets, entity_names=c_enms)

    @property
    def avg_entity_per_example(self) -> float:
        """
        :return: Average number of entities per example
        """
        return np.mean([len(d.entity_names) for d in self.egs if len(d.entity_names) > 0])  # drop examples w/ no entity

    @property
    def avg_n_words(self) -> float:
        """
        :return: Average length of sentence
        """
        # return np.mean([len(d.sentence) for d in self.egs])
        return np.mean([len(d.sentence.split()) for d in self.egs])

    @property
    def token_summary(self):
        """
        R like 5-number summary of sentence token length
        """
        return describe(vals=[len(punc_tokenize(d.sentence)) for d in self.egs], round_dec=2)

    def summary(self, with_all_entity_types: bool = True, compare_with_test: Union[bool, str] = True) -> StatsSummaryOutput:
        n = len(self.egs)
        ets = self.entity_types

        sents_no_ent = [d.sentence for d in self.egs if len(d.entity_names) == 0]
        n_no_ent = len(sents_no_ent)

        ec_out = self.entity_counts()
        enm2count, et_count = ec_out['entity_names'], ec_out['entity_types']
        et_count_uniq = self.entity_counts(distinct_entity=True)['entity_types']
        total_all = sum(et_count.values())
        total_all_uniq = sum(et_count_uniq.values())
        et_count = {e: et_count[e] for e in ets}
        et_count_uniq = {e: et_count_uniq[e] for e in ets}

        total_interested = sum(et_count.values())
        total_interested_uniq = sum(et_count_uniq.values())

        def to_perc(x, **kwargs):
            return to_percent(x, decimal=1, **kwargs)
        et_count_norm = {k: to_perc(v/total_all, append_char=None) for k, v in et_count.items()}  # normalize to percentages
        et_count_norm_uniq = {k: to_perc(v/total_all_uniq, append_char=None) for k, v in et_count_uniq.items()}
        # sum everything and get ratio relative to *all* entity types
        if total_interested < total_all:
            et_count_norm['[total]'] = to_perc(total_interested / total_all, append_char=None)
        if total_interested_uniq < total_all_uniq:
            et_count_norm_uniq['[total]'] = to_perc(total_interested_uniq / total_all_uniq, append_char=None)
        # get average occurrence of each entity
        #   1) normalize across all entities into a single metric
        #   2) normalize across all entities of the same entity type
        avg_enm_occ = np.mean(list(enm2count.values()))  # already distinct elements in the counter, micro-average
        avg_enm_occ_et = {et: np.mean([enm2count[enm] for enm in enms]) for et, enms in self.entity_hierarchy(color=False).items()}

        enms_w_comma = [en for en in self.entity_names if ',' in en]
        ratio_comma = len(enms_w_comma) / len(self.entity_names)

        lst_multi_occur = [(i, eg.get_multi_occur_entity_info()) for i, eg in enumerate(self.egs)]
        n_multi_occur_entity = sum(info.has_multi_occur_entity for _, info in lst_multi_occur)
        n_multi_occur_entity_diff_type = sum(info.has_multi_occur_entity_diff_type for _, info in lst_multi_occur)
        sample_multi_occur_entity = [self.egs[i] for i, info in lst_multi_occur if info.has_multi_occur_entity]
        sample_multi_occur_entity_diff_type = [self.egs[i] for i, info in lst_multi_occur if info.has_multi_occur_entity_diff_type]

        d = {
            'dataset-dir-name': self.dir_name,
            '#examples': n,

            '#entity types': len(self._entity_types_all),
            '#distinct entity names': len(self.entity_names),
            '#entity names': sum(len(d.entity_names) for d in self.egs),
            'entity names w/ comma': enms_w_comma,
            '%comma in entity names': to_perc(ratio_comma),

            'vocab size': len(self.vocab),
            'avg #entities per example': round(self.avg_entity_per_example, 2),
            'avg #words per example': round(self.avg_n_words, 2),

            '#samples w/o entity': n_no_ent,
            '%samples w/o entity': to_perc(n_no_ent / n)
        }
        if (len(sents_no_ent) / n) < 0.1:  # otherwise, too much to show
            d['samples w/o entity'] = sents_no_ent

        d.update({
            '#samples w/ multi-occurring entities': n_multi_occur_entity,
            '%samples w/ multi-occurring entities': to_perc(n_multi_occur_entity / n),
            'samples w/ multi-occurring entities': [sdpc(s, as_str=False) for s in sample_multi_occur_entity],

            '#samples w/ multi-occurring entities of different types': n_multi_occur_entity_diff_type,
            '%samples w/ multi-occurring entities of different types': to_perc(n_multi_occur_entity_diff_type / n),
            'samples w/ multi-occurring entities of different types': [sdpc(s, as_str=False) for s in sample_multi_occur_entity_diff_type]
        })
        if with_all_entity_types and set(ets) != set(self._entity_types_all):
            d['all entity types'] = self._entity_types_all
        d_et = {
            'interested entity types': ets,
            'entity type dist': et_count,
            'entity types dist (%)': et_count_norm,
            'distinct entity type dist': et_count_uniq,
            'distinct entity types dist (%)': et_count_norm_uniq,
            'avg entity name occurrence': round(avg_enm_occ, 2),
            'type-wise avg entity name occurrence': {k: round(v, 2) for k, v in avg_enm_occ_et.items()}
        }
        d_ts, d_ts_vect = None, None
        if compare_with_test:
            d_ts = self.test_token_overlap()
            if compare_with_test == 'vector':
                d_ts_vect = self.test_vector_overlap()
        return StatsSummaryOutput(summary=d, entity_dist=d_et, test_set_overlap=d_ts, test_set_vector_overlap=d_ts_vect)

    def test_token_overlap(self) -> Dict[str, Any]:
        """
        get overlapping vocab and entity name tokens
        """
        vocab_ts, en_ts, en_toks_ts = self.test_stats.vocab, self.test_stats.entity_names,  self.test_stats.entity_name_toks

        ovl_v = vocab_ts & self.vocab
        ovl_enm = set(en_ts) & set(self.entity_names)
        ovl_e_tok = en_toks_ts & self.entity_name_toks
        sents, enms = [d.sentence for d in self.egs], [' '.join(d.entity_names) for d in self.egs]

        def to_perc(x):
            return to_percent(x, decimal=1)
        ret = {
            '#overlap sentence words': len(ovl_v),
            '#overlap entity name words': len(ovl_e_tok),
            '#overlap entity names': len(ovl_enm),
            'sentence overlap ratio': to_perc(len(ovl_v) / len(vocab_ts)),
            'entity name overlap ratio': to_perc(len(ovl_enm) / len(en_ts)),
            'entity name token overlap ratio': to_perc(len(ovl_e_tok) / len(en_toks_ts))
        }

        overlap = False
        # overlap = True
        if overlap:
            sents_ts, enms_ts = [d.sentence for d in self.test_stats.egs], [' '.join(d.entity_names) for d in self.test_stats.egs]
            ret.update({
                # 'sentence weighted-overlap': word_overlap(source_corpus=sents, target_corpus=sents_ts),
                # 'entity name weighted-overlap': word_overlap(source_corpus=enms, target_corpus=enms_ts),
                # 'overlapping words': overlapping_words(source_corpus=sents, target_corpus=sents_ts, top_n=50),
                'overlapping entity names': overlapping_words(source_corpus=enms, target_corpus=enms_ts, top_n=50)
            })
        return ret

    def test_vector_overlap(self, kind: str = 'sentence') -> VectorOverlapOutput:
        """
        Get a sense of distance on the sentences between generated data & test set

        Compute max cosine sim for each generated sentence w/ test set
        Compute ratio of test sentences that has a closest generated sentence mapped
        """
        ca(plot_type=kind)
        if kind == 'sentence':
            vects, ts_vects = self.sentence_vectors, self.test_stats.sentence_vectors
        else:  # `entity`
            vects, ts_vects = self.entity_vectors, self.test_stats.entity_vectors
        # batched cosine similarity between 2 lists of vectors
        # normalize
        vects = vects / np.linalg.norm(vects, axis=1, keepdims=True)
        ts_vects = ts_vects / np.linalg.norm(ts_vects, axis=1, keepdims=True)
        cos_sims = vects @ ts_vects.T

        # get indices of most-similar test sentence for each generated sentence
        idxs = np.argmax(cos_sims, axis=1)
        n_unique = len(set(idxs))  # get number of unique test sentences that has a closest generated sentence mapped

        sims = cos_sims[np.arange(len(cos_sims)), idxs]
        n_ts_sent = len(ts_vects)
        sim_sum = describe(sims*100, round_dec=1)
        sim_sum.pop('count')
        ts_mapped = {'count': n_unique, 'total': n_ts_sent, 'ratio': round((n_unique / n_ts_sent) * 100, 2)}
        return VectorOverlapOutput(cosine_sims=sims, d_log={'cosine-sim-summary': sim_sum, 'uniq-test-sentence-mapped': ts_mapped})

    def log_stats(
            self, compare_with_test: Union[bool, str] = True, logger: logging.Logger = None, with_contexts: bool = False, **logger_kwargs
    ):
        logger = logger or _logger
        out = self.summary(compare_with_test=compare_with_test)
        logger.info(
            f'NER Dataset Stats: {pl.i(out.summary, indent=True)}, '
            f'Entity Distributions: {pl.i(out.entity_dist, indent=1, align_keys=1)}', **logger_kwargs)

        if compare_with_test:
            logger.info(f'Test Set Token overlap: {pl.i(out.test_set_overlap, indent=1)}', **logger_kwargs)
            if compare_with_test == 'vector':
                logger.info(f'Test Set Vector overlap: {pl.i(out.test_set_vector_overlap.d_log, indent=1)}', **logger_kwargs)

        logger.info(f'NER Dataset Entity counts: {pl.i(self.entity_hierarchy(), indent=True, value_no_color=True)}', **logger_kwargs)
        if with_contexts:
            et_hier = self.entity_hierarchy(with_context=True)
            logger.info(f'NER Dataset Entities w/ contexts: {pl.i(et_hier, indent=True, value_no_color=True)}', **logger_kwargs)


if __name__ == '__main__':
    from src.generate.step_wise import STEP_WISE_DNM

    # dnm = 'conll2003'
    # dnm = 'conll2003-no-misc'
    # dnm = 'job-desc'
    # dnm = 'mit-movie'
    dnm = 'mit-restaurant'
    # dnm = 'job-stack'
    # dnm = 'wiki-gold-no-misc'
    # dnm = 'ncbi-disease'
    # ori = False
    ori = True

    # n_demo = 5
    # n_demo = 2
    n_demo = 1

    dc = False
    # dc = True
    # de = False
    de = True
    psg = False
    # psg = True
    step = False
    # step = True
    agg = False
    # agg = True
    # cot = False
    cot = True
    sic(dnm, ori)
    sic(dc, dc, psg, step)
    if step:
        sic(agg, cot)

    if dnm in ['job-stack', 'ncbi-disease']:
        tr = 'bio-train-1k'
    elif dnm in ['wiki-gold', 'wiki-gold-no-misc']:
        # tr = 'bio-train-all'
        tr = 'bio-train-1.1k'
    else:
        assert dnm in ['conll2003', 'conll2003-no-misc', 'mit-movie', 'mit-restaurant']
        # tr = 'train-1k'
        tr = 'bio-train-1350'  # as a better reference w/ ~1.5K generated data
    # if dnm in ['mit-movie']:
    tr_full = 'train-all'
    # else:
    #     raise NotImplementedError
    ts = dataset2test_file_name(dataset_name=dnm)
    sic(tr, tr_full, ts)

    def load_stats(dir_nm: str = None, **kwargs):
        pa_args = dict(sub_dir=STEP_WISE_DNM) if step else None
        return NerDatasetStats.from_dir_name(dataset_name=dnm, dir_name=dir_nm, path_args=pa_args, verbose=True, **kwargs)

    def log_stats(lst_dnms: Union[str, List[str]], **kwargs):
        if isinstance(lst_dnms, str):
            lst_dnms = [lst_dnms]
        for d_nm in lst_dnms:
            ds = load_stats(dir_nm=d_nm, **kwargs)
            # cxt = False
            cxt = True
            ds.log_stats(compare_with_test=d_nm != ts, with_contexts=cxt)

    # log_stats(tr)
    # log_stats(tr_full)
    log_stats(ts)
