import os
import json
import logging
from os.path import join as os_join
from copy import deepcopy
from typing import Dict, Tuple, List, Union, Any, Callable
from dataclasses import dataclass, asdict, astuple
from collections import Counter, defaultdict

from stefutil import *
from src.util import *
from src.util.ner_example import *
from src.data_util.prettier import *
from src.data_util.stats import *
from src.data_util.logprob import *


__all__ = [
    'DSample', 'SampleLogMap', 'de_duplicate_samples', 'de_duplicate_ner_samples',
    'NerDataset', 'NerDatasetOutput', 'samples2train_dataset', 'write_dataset', 'write_processed_dataset',
    'REDO_CHALLENGING_DIR_NM', 'log_n_save_challenging_samples',
    'NerProcessOutput', 'finish_ner_processing',
    'AnnotationDiffOutput', 'analyze_diff', 'compare_datasets',
    'CompareSamplePair', 'CompareSamplesOutput', 'compare_samples',
    'LoadNerSamplesOutput', 'dataset_dir_name2ner_samples',
    'MERGED_DIR_NAME', 'merge_datasets'
]


_logger = get_logger('Dataset Util')


@dataclass
class DedupOutput:
    samples: Union[List[NerExample], List[NerReadableExample], List[str]] = None
    sample2logprob: Dict[NerExample, float] = None
    sample2entity_logprobs: Dict[NerExample, List[float]] = None
    n_dup_drop: int = None


DSample = Dict[str, Any]
SampleLogMap = Callable[[Union[NerExample, DSample]], DSample]


def de_duplicate_samples(
        samples: List = None, logprobs: List[float] = None, entity_logprobs: List[List[float]] = None,
        logger: logging.Logger = None, sample_log_map: SampleLogMap = None, sample_kind: str = 'sample', **log_kwargs
) -> DedupOutput:
    """
    if duplicates on at sample level, keep only one copy
    """
    logger = logger or _logger

    assert all(isinstance(x, type(samples[0])) for x in samples[1:])  # sanity check elements are of the same type

    n_sample_ori = len(samples)
    n_uniq = len(set(samples))
    sample2logprob, sample2entity_logprobs = None, None

    if n_uniq < len(samples):
        # log duplicate samples
        c = Counter(samples)
        dup_samples = [k for k, v in c.items() if v > 1]
        avg_dup = sum(v for v in c.values() if v > 1) / len(dup_samples)
        d_log = {'#drop': len(samples) - n_uniq, 'avg #duplicates per group': round(avg_dup, 4)}
        n_dup = len(dup_samples)
        if isinstance(samples[0], NerExample):
            assert all(isinstance(x, NerExample) for x in dup_samples)  # sanity check
            dup_samples: List[Dict] = [asdict(x) for x in dup_samples]
        else:
            assert all(isinstance(x, (dict, str)) for x in dup_samples)  # `str` for de-duplicating sentences
        if sample_log_map:
            dup_samples = [sample_log_map(x) for x in dup_samples]
        kd = sample_kind
        kd_pl = f'{kd}s'
        logger.warning(f'Found {pl.i(n_dup)} group(s) of duplicate {pl.i(kd_pl)} '
                       f'and only one {pl.i(kd)} kept w/ {pl.i(d_log)}: {pl.i(dup_samples, indent=2)}', **log_kwargs)

        if logprobs:
            assert len(samples) == len(logprobs) == len(entity_logprobs)  # sanity check
            # get the logprob for each sample; for samples in the same duplicate group, take average
            s2lp = defaultdict(list)
            s2elp = defaultdict(list)
            for s, lp, e_lp in zip(samples, logprobs, entity_logprobs):
                s2lp[s].append(lp)
                e_lp: List[float]
                s2elp[s].append(e_lp)  # this can be empty, if the sentence have no entities

            # for sample-wise logprob, just take the average
            sample2logprob = {s: sum(lps) / len(lps) for s, lps in s2lp.items()}

            # for the same duplicate group, keep only 1 version of the entity logprobs;
            #   use the list of entities w/ the smallest average logprob, i.e. prefer the highest uncertainty
            def resolve_entity_logprobs(lps: List[List[float]]) -> List[float]:
                n = len(lps)
                assert n > 0  # there should at least 1 list added, by construction; that list added can be empty tho
                # check how many lists are non-empty
                idx_non_empty = [i for i, x in enumerate(lps) if len(x) > 0]
                if len(idx_non_empty) > 0:
                    # sic(lps, len(lps))
                    lps = [lps[i] for i in idx_non_empty]
                    return min(lps, key=lambda x: sum(x) / len(x))
                else:  # all lists are empty
                    return []
            # debug_empty = {s: lps for s, lps in s2elp.items() if any(len(x) == 0 for x in lps)}
            # sic(debug_empty)
            # sample2entity_logprobs = {s: min(lps, key=lambda x: sum(x) / len(x)) for s, lps in s2elp.items()}
            sample2entity_logprobs = {s: resolve_entity_logprobs(lps) for s, lps in s2elp.items()}
        samples = list(dict.fromkeys(samples))  # drop duplicates and keep original order
    else:
        if logprobs:
            sample2logprob = dict(zip(samples, logprobs))
            sample2entity_logprobs = dict(zip(samples, entity_logprobs))
    return DedupOutput(
        samples=samples, n_dup_drop=n_sample_ori - len(samples), sample2logprob=sample2logprob, sample2entity_logprobs=sample2entity_logprobs)


def de_duplicate_ner_samples(
        samples: List[NerExample] = None, logprobs: List[float] = None, entity_logprobs: List[List[float]] = None,
        logger: logging.Logger = None, sample_log_map: SampleLogMap = None, **log_kwargs
) -> DedupOutput:
    """
    drop duplicate NER samples
    """
    assert all(isinstance(eg, type(samples[0])) for eg in samples[1:])  # check examples are of the same class
    logger = logger or _logger

    n_sample_ori = len(samples)
    out = de_duplicate_samples(
        samples=samples, logprobs=logprobs, entity_logprobs=entity_logprobs,
        logger=logger, sample_log_map=sample_log_map, sample_kind='sample', **log_kwargs)
    samples, s2lp, s2elp = out.samples, out.sample2logprob, out.sample2entity_logprobs

    # then, if sentence level duplicates, drop all
    sents = [eg.sentence for eg in samples]
    c = Counter(sents)
    dup_sents = {s for s, v in c.items() if v > 1}
    if len(dup_sents) > 0:
        # log duplicate samples
        dup_sent2samples = defaultdict(dict)
        sent2running_count = defaultdict(int)
        for s in samples:
            sent = s.sentence
            if sent in dup_sents:
                d = asdict(s)
                del d['sentence']
                c = sent2running_count[sent]
                sent2running_count[sent] += 1
                # the `tokens` key for `NerBioExample` would be the same for all duplicates, one is enough
                d = {(k if k == 'tokens' else f'{k}-{c+1}'): v for k, v in d.items()}
                dup_sent2samples[sent].update(d)
        dup_samples = [{**dict(sentence=sent), **v} for sent, v in dup_sent2samples.items()]
        if sample_log_map:
            dup_samples = [sample_log_map(x) for x in dup_samples]
        n_drop = sum(sent2running_count.values())
        d_log = {'#drop': n_drop, 'avg #duplicates per group': round(n_drop / len(dup_sent2samples), 4)}
        logger.warning(f'Found {pl.i(len(dup_samples))} group(s) of sentences w/ differing annotations and '
                       f'all samples dropped w/ {pl.i(d_log)}: {pl.i(dup_samples, indent=2)}', **log_kwargs)

        samples = [eg for eg in samples if eg.sentence not in dup_sents]
        if logprobs is not None:
            assert s2lp is not None and s2elp is not None  # sanity check
            s2lp = {eg: s2lp[eg] for eg in samples}  # get logprob for only the kept samples
            s2elp = {eg: s2elp[eg] for eg in samples}
    return DedupOutput(samples=samples, n_dup_drop=n_sample_ori - len(samples), sample2logprob=s2lp, sample2entity_logprobs=s2elp)


def write_dataset(samples: List[Dict[str, Any]] = None, output_filename: str = None, kind: str = None, for_train: bool = False, **kwargs):
    ca.assert_options(display_name='NER Data Kind', val=kind, options=['readable', 'bio'])

    if kind == 'readable':
        def map_sample(x: Dict[str, Any]) -> Dict[str, Any]:
            return sdp(x, as_str=False)
    else:
        assert kind == 'bio'

        def map_sample(x: Dict[str, Any]) -> Dict[str, Any]:
            return x

    if kind != 'bio':  # too much indentation from list of tokens & tags => too large file => ignore
        with open(f'{output_filename}.json', 'w') as f:
            kwargs.update(examples=samples)
            json.dump(kwargs, f, indent=4)
    with open(f'{output_filename}.jsonl', 'w') as f:
        for s in samples:
            if for_train and kind == 'bio':
                s = deepcopy(s)
                s['labels'] = s.pop('ner_tags')  # for training pipeline, the BIO-tags are indexed via `labels`
            f.write(f'{json.dumps(s)}\n')
    # also writes a prettier logged version for easier manual review
    with open(f'{output_filename}.log', 'w') as f:
        lst_log = [map_sample(s) for s in samples]
        f.write(pl.nc(lst_log, indent=2))


@dataclass
class NerDataset:
    bio: List[Dict[str, Any]] = None  # for training
    readable: List[Dict[str, Any]] = None  # for readability
    n_dup_drop: int = None  # #duplicate samples dropped
    sample2logprob: Dict[NerExample, float] = None
    sample2entity_logprobs: Dict[NerExample, List[float]] = None


@dataclass
class NerDatasetOutput:
    original: NerDataset = None
    lowercased: NerDataset = None


def samples2train_dataset(
        examples: Union[List[NerExample], List[NerReadableExample]],
        logprobs: List[float] = None, entity_logprobs: List[List[float]] = None,
        data_format: str = 'natural', ignore_case: bool = None,
        write: Union[bool, str] = False, write_fnm: Union[str, Dict[str, str]] = None, output_path: str = None,
        dedup: bool = False, sample_log_map: Union[bool, SampleLogMap] = True,
        lowercase: bool = None, logger: logging.Logger = None
) -> NerDatasetOutput:
    """
    Convert NER data sample to training data format in BIO, including de-duplication and file write

    :param examples: list of NER samples to convert to training data format
    :param logprobs: list of log probability corresponding to each sample by index
    :param entity_logprobs: list of entity log probability list corresponding to each sample and each entity annotation by index
    :param data_format: the format of the input samples, in [`natural`, `bio`]
    :param ignore_case: whether to ignore case when converting from natural to BIO format
        If `None`, automatically ignore case if it will not influence the BIO format conversion
    :param write: whether to write the dataset to file
        If a string, the kind of dataset to write, in [`readable`, `bio`, `both`]
    :param write_fnm: the filename to write the dataset to, corresponding to the `write` kind
    :param output_path: the directory to write the dataset to
    :param dedup: whether to de-duplicate the samples
    :param sample_log_map: A map function that converts a sample to a prettier version for logging
    :param lowercase: whether to lower-case the samples
        If True, both the original version and the lower-cased version will be written to file
            The lower-cased version will be written to a subdirectory `lowercase`
    :param logger: logger
    """
    ca(data_format=data_format)
    logger = logger or _logger  # allow passing in custom logger for writing to file

    if sample_log_map is True:
        def sample_log_map(x: Dict[str, Any]) -> Dict[str, Any]:
            return sdpc(x, as_str=False)

    def _run_single(lc: bool = None):
        examples_ = deepcopy(examples)
        if data_format == 'natural':
            assert all(isinstance(x, NerReadableExample) for x in examples_)
        else:  # `bio`
            assert all(isinstance(x, NerBioExample) for x in examples_)
        extra = dict()
        if lc:
            assert all(isinstance(x, NerReadableExample) for x in examples_)
            examples_: List[NerReadableExample]
            examples_ = [x.to_lower() for x in examples_]
            logger.info(f'{pl.i(len(examples_))} samples lower-cased')
            extra = dict(block='stdout')

        s2lp, s2e_lp = None, None
        if logprobs:
            assert dedup  # so that definitely get a one-to-one mapping: sample => logprob
        if dedup:
            de_dup_args: Dict[str, Any] = dict(logger=logger, sample_log_map=sample_log_map, extra=extra)
            if not lc:  # Intended for sending samples back to LLM, the lower-cased version is not needed
                de_dup_args.update(logprobs=logprobs, entity_logprobs=entity_logprobs)
            out = de_duplicate_ner_samples(samples=examples_, **de_dup_args)
            examples_, n_dup_drop = out.samples, out.n_dup_drop
            if logprobs:
                s2lp = out.sample2logprob
                s2e_lp = out.sample2entity_logprobs
        else:
            n_dup_drop = None

        lst_readable, lst_bio = [], []
        if data_format == 'natural':
            for eg in examples_:
                eg: NerReadableExample
                lst_readable.append(asdict(eg))
                lst_bio.append(asdict(eg.to_bio(ignore_case=ignore_case)))
        else:  # `bio`
            for eg in examples_:
                eg: NerBioExample
                bio = asdict(eg)
                bio['labels'] = bio.pop('ner_tags')
                bio = {k: bio[k] for k in ['tokens', 'labels', 'sentence']}  # re-order for consistency with `natural` format

                readable = bio2readable(eg)
                lst_readable.append(asdict(readable))
                lst_bio.append(bio)
        if write:
            if isinstance(write, str):
                write_kind = write
                ca.assert_options(display_name='Write NER Dataset type', val=write_fnm, options=['readable', 'bio', 'both'])
            else:
                write_kind = 'both'
            fnm_map = dict(readable='readable', bio='bio')

            write_fnm_ = None
            if write_fnm is not None:
                write_fnm_ = write_fnm if isinstance(write_fnm, dict) else dict(readable=write_fnm, bio=write_fnm)
            for k, v in (write_fnm_ or dict()).items():
                fnm_map[k] = f'{fnm_map[k]}-{v}'

            fnm_key = dict(readable=['readable'], bio=['bio'], both=['readable', 'bio'])[write_kind]
            fnms = []
            for k in fnm_key:
                fnm_ = fnm_map[k]
                out_path = output_path
                if lc:
                    out_path = os_join(out_path, 'lowercase')
                os.makedirs(out_path, exist_ok=True)
                write_dataset(samples=lst_bio if k == 'bio' else lst_readable, output_filename=os_join(out_path, fnm_), kind=k)
                fnms.append(fnm_)
            logger.info(f'{pl.i(len(lst_bio))} samples written to {pl.i(fnms, indent=1)} at {pl.i(stem(output_path, top_n=3))}', **extra)
        return NerDataset(bio=lst_bio, readable=lst_readable, n_dup_drop=n_dup_drop, sample2logprob=s2lp, sample2entity_logprobs=s2e_lp)
    if not lowercase:
        return NerDatasetOutput(original=_run_single())
    else:
        return NerDatasetOutput(original=_run_single(), lowercased=_run_single(lc=True))


def write_tag2id(entity_types: List[str], path: str, logger: logging.Logger = None, **logger_kwargs):
    path_tag2id = os_join(path, 'tag_to_id.json')
    with open(path_tag2id, 'w') as f:
        json.dump(ner_labels2tag2index(entity_types=entity_types), f, indent=4)
    (logger or _logger).info(f'{pl.i("tag2id")} written to {pl.i(stem(path_tag2id))}', **logger_kwargs)


def write_processed_dataset(
        ner_dataset: NerDataset, dataset_name: str = None, entity_types: List[str] = None,
        output_path: str = None, logger: logging.Logger = None, **logger_kwargs
):
    logger = logger or _logger
    if entity_types is None:
        entity_types = sum((list(eg['entity_types']) for eg in ner_dataset.readable), start=[])
        entity_types = sorted(set(entity_types))

    for kd, samples in [('readable', ner_dataset.readable), ('bio', ner_dataset.bio)]:
        out_path = os_join(output_path, f'{kd}-train')
        write_dataset(samples=samples, output_filename=out_path, kind=kd, for_train=True, dataset_name=dataset_name, entity_types=entity_types)
    logger.info(f'{pl.i(len(ner_dataset.bio))} training samples written to {pl.i(stem(output_path))}', **logger_kwargs)
    write_tag2id(entity_types=entity_types, path=output_path, logger=logger, **logger_kwargs)

    lst_readable = [NerReadableExample.from_d(d) for d in ner_dataset.readable]
    ets = sconfig(f'datasets.{dataset_name}.readable-entity-types')
    nds = NerDatasetStats(dataset_name=dataset_name, dataset_path_or_examples=lst_readable, entity_types=ets, dir_name=stem(output_path))
    nds.log_stats(logger=logger, **logger_kwargs)
    return entity_types


REDO_CHALLENGING_DIR_NM = 'redo-challenging'  # for re-annotate challenging 1-stage samples w/ 2-stage Gen


def log_n_save_challenging_samples(
        challenging_samples: Dict[str, List[Any]] = None, sample_kind: str = 'sample',
        output_path: str = None, logger: logging.Logger = None, sample_map: Callable = None, **kwargs
):
    """
    write challenging/uncertain samples to be re-annotated/classified

    keep the reason for challenging in the file output, for manual inspection
    """
    k_pl = f'{sample_kind}s'
    ch_samples = sum((samples for samples in challenging_samples.values()), start=[])
    # may contain duplicates since some samples may be challenging for multiple reasons
    ch_samples = list(dict.fromkeys(ch_samples))
    if sample_map:
        ch_samples = [sample_map(samp) for samp in ch_samples]
        challenging_samples = {k: [sample_map(samp) for samp in v] for k, v in challenging_samples.items()}

    kd2c = {k: len(v) for k, v in challenging_samples.items()}
    challenging_samples[k_pl] = ch_samples
    with open(output_path, 'w') as f:
        json.dump(challenging_samples, f, indent=4)
    d_log = {f'#challenging-{k_pl}': sum(kd2c.values()), f'#unique-challenging-{k_pl}': len(ch_samples), 'reason-counts': kd2c}
    logger.info(f'Challenging {k_pl} written w/ {pl.i(d_log, indent=2)}', **kwargs)


@dataclass
class NerProcessOutput:
    dataset: NerDataset = None
    samples: List[NerReadableExample] = None


def finish_ner_processing(
        samples: List[NerReadableExample], sentences: List[str] = None, challenging_sentences: Dict[str, List[str]] = None,
        logprobs: bool = False, sample_logprobs: List[float] = None, entity_logprobs: List[List[float]] = None,
        ec: EdgeCases = None, logger: logging.Logger = None,
        data_format: str = 'natural', dataset_name: str = 'conll2003', entity_types: List[str] = None,
        dedup: bool = False, lowercase: bool = False, output_path: str = None,
        d_log: Dict[str, Any] = None, time: Union[str, Timer] = None
) -> NerProcessOutput:
    logger = logger or _logger
    if ec and ec.have_edge_case:
        logger.info(ec.summary())
    d_log['#sample-extracted'] = len(samples)

    if data_format != 'natural':
        raise NotImplementedError
    to_dset_args: Dict[str, Any] = dict(data_format=data_format, logger=logger, ignore_case=None, dedup=dedup, lowercase=lowercase)
    if logprobs:
        to_dset_args.update(logprobs=sample_logprobs, entity_logprobs=entity_logprobs)
    out = samples2train_dataset(examples=samples, **to_dset_args)
    ori, lc_ = out.original, out.lowercased

    def finish_single(dset: NerDataset = None, lc: bool = None, out_path: str = None):
        extra = None
        if lc:
            out_path = os_join(out_path, 'lowercased')
            os.makedirs(out_path, exist_ok=True)
            extra = dict(block='stdout')
        write_processed_dataset(
            ner_dataset=dset, dataset_name=dataset_name, entity_types=entity_types, output_path=out_path, logger=logger, extra=extra)
        
        n_written = len(dset.readable)
        ret = dict()
        if not lc:
            # these are intended for re-sending to LLM, mostly on processing 1-stage samples,
            #   so don't need to save the lower-cased version
            if sentences:
                # sents = deepcopy(sentences)
                sents = sentences
                if lowercase:
                    sents = [s.lower() for s in sents]
                # de-duplicate the sentences & write to file
                sents = de_duplicate_samples(samples=sents, logger=logger, extra=extra).samples
                with open(os_join(out_path, 'sentences.json'), 'w') as f:
                    json.dump(dict(sentences=sents), f, indent=4)
                ret['#unique-sentence'] = len(sents)

            if logprobs:
                ec_ = EdgeCases(logger=logger)  # for logging entity triple mapping
                # save & log sample-wise logprobs
                s2lp, s2elp = dset.sample2logprob, dset.sample2entity_logprobs
                samples_w_lp = [{**asdict(sample_), **dict(logprob=lp)} for sample_, lp in s2lp.items()]
                samples_sorted = log_n_save_samples_w_logprob(
                    samples_w_logprob=samples_w_lp, output_path=out_path, logger=logger, extra=extra)

                # log & save each (sentence, entity name, entity type) triplets w/ logprob
                triples_w_logprob = []
                for sample, et_lps in s2elp.items():
                    sample: NerReadableExample
                    sent, enms, ets = sample.sentence, sample.entity_names, sample.entity_types
                    assert len(enms) == len(ets) == len(et_lps)  # sanity check

                    # Edge cases:
                    #   1> if the sample contains multi-occurring entities, need to add additional index to differentiate,
                    #   2> each entity may be distinct, but one entity may contain a subset that's exactly another entity
                    #       e.g. X: `The Golden Gate Bridge is a suspension bridge spanning the Golden Gate`
                    #           Entities: `Golden Gate Bridge`, `Golden Gate`
                    #       In this case, to select the proper entity span to highlight for LLM correction, need to distinguish
                    #           For that span who's the subset, e.g. `Golden Gate`, need to know which of the match is the correct one
                    #               e.g. `Golden Gate` index = 1 for the 2nd match
                    enm2c = Counter(enms)
                    multi_occur = any(c > 1 for c in enm2c.values())
                    if multi_occur:
                        # track the current index for each multi-occurring entity
                        enm2multi_occ_idx = dict((enm, 0) for enm, c in enm2c.items() if c > 1)
                    else:
                        enm2multi_occ_idx = dict()

                    # get the super-string entities for each entity
                    entity_sub2super = defaultdict(list)
                    # iterate all possible ordered pairs
                    # n_enm = len(enms)
                    # for i, j in [(i, j) for i in range(n_enm) for j in range(n_enm)]:
                    #     enm_sub, enm_super = enms[i], enms[j]
                    for enm_sub in enms:  # note we include the current entity
                        if enm_sub in entity_sub2super:
                            # must be due to multi-occurring entities, and this entity's relative index is already processed
                            assert multi_occur  # sanity check
                            continue
                        for enm_super in enms:
                            # check for substring and not exact match
                            ms = patterns.find_match(text=enm_super, keyword=enm_sub)  # make sure super-string by exact word boundary
                            n_mch = len(ms)
                            # if n_mch not in [0, 1]:
                            #     sic(sent, enms)
                            #     sic(enm_sub, enm_super, ms)
                            # assert n_mch in [0, 1]  # sanity check at most 1 match
                            # surprisingly, the above may not be the case, e.g.
                            #   X: `The Hindu Vishwa Hindu Parishad, or VHP,
                            #       is a Hindu nationalist organization in India that aims to promote and protect Hindu culture and traditions.`
                            #   Entities: [**Hindu Vishwa Hindu Parishad**, VHP, **Hindu**, India, **Hindu**]
                            if n_mch > 0:  # the current entity will also be added, to figure out the relative ordering
                                entity_sub2super[enm_sub].append(enm_super)
                    # now filter out matches if just contains the current entity, also filters out the case where
                    #   all the super-string-entities are is really just the same entity appearing multiple times
                    entity_sub2super = {enm: lst for enm, lst in entity_sub2super.items() if lst != [enm] * len(lst)}
                    has_sub = len(entity_sub2super) > 0

                    ms_enms = None
                    if has_sub:  # for sanity check, see below
                        out_ms = patterns.find_matches(text=sent, keywords=enms, ignore_case=True, search_in_order=True, return_all=False)
                        assert out_ms.success  # sanity check
                        ms_enms = out_ms.matches

                    # if multi_occur and has_sub:  # sanity check these are mutually exclusive, for easy logic
                    #     assert len(entity_sub2super) == 1 and len(enm2multi_occ_idx) == 1
                    #     enm_c, enm_sub = list(enm2multi_occ_idx.keys())[0], list(entity_sub2super.keys())[0]
                    #     if enm_c == enm_sub:
                    #         sic(sent, enms, ets)
                    #         sic(enm2multi_occ_idx, entity_sub2super)
                    #         raise NotImplementedError
                    #     assert enm_c != enm_sub

                    for i, (enm, et, lp) in enumerate(zip(enms, ets, et_lps)):
                        d = dict(sentence=sent, span=enm, entity_type=et, average_logprob=lp)
                        if has_sub and enm in entity_sub2super:
                            ms = patterns.find_match(text=sent, keyword=enm)
                            enms_super = entity_sub2super[enm]
                            # sanity check the entity itself and its super-string-entities all found again
                            #   Equality is not necessarily the case, for edge case above, e.g.
                            #       entities: `Hindu Vishwa Hindu Parishad` & `Hindu`
                            assert len(ms) >= len(enms_super)
                            # the corresponding index for the current entity is already determined, from `entity_sub2super`,
                            #   which is the relative position of the current entity in the list
                            idxs_match = [i for i, e in enumerate(enms_super) if e == enm]
                            n_mch = len(idxs_match)
                            if n_mch == 1:
                                idx_rel = idxs_match[0]
                            else:
                                assert n_mch > 1  # sanity check, must be due to multi-occurring entities
                                assert multi_occur and enm in enm2multi_occ_idx  # sanity check
                                assert n_mch == enm2c[enm]  # sanity check
                                # so get the relative index within the same multi-occurring entity list
                                idx_same_enm = enm2multi_occ_idx[enm]  # will be incremented later, see below
                                idx_rel = idxs_match[idx_same_enm]
                            # note that the index relative to sentence is not relative index of the entity in the list
                            #   For edge case above, e.g. entities: `Hindu Vishwa Hindu Parishad` & `Hindu`
                            idx_curr = 0
                            entities_before = enms_super[:idx_rel]
                            for enm_super in entities_before:
                                n_add = len(patterns.find_match(text=enm_super, keyword=enm))
                                assert n_add >= 1  # sanity check
                                idx_curr += n_add
                            # **note** this still assumes all occurrences of the current entity is extracted, as exact match or super-string
                            #   sanity check this is indeed the case by comparing the span indices
                            span_from_sent = ms_enms[i].span('keyword')
                            mch_curr = patterns.find_match(text=sent, keyword=enm)[idx_curr]
                            span_from_process = mch_curr.span('keyword')
                            # if span_from_sent != span_from_process:
                            #     sic(sent, enms)
                            #     sic(enm, enms_super)
                            #     sic(idxs_match, idx_rel, idx_curr)
                            #     sic(span_from_sent, span_from_process)
                            assert span_from_sent == span_from_process  # sanity check

                            # d['index_super'] = idxs_match[idx_rel]
                            d['index_super'] = idx_curr
                            ec_(msg=f'Index added for super-string entity w/ {pl.i(d)}', kind='super-string-entity-add-index', args=d)

                        if multi_occur and enm in enm2multi_occ_idx:
                            d['index'] = enm2multi_occ_idx[enm]
                            # sic(d, enm2c)
                            # raise NotImplementedError
                            enm2multi_occ_idx[enm] += 1
                            ec_(msg=f'Index added for multi-occurring entity w/ {pl.i(d)}', kind='multi-occur-entity-add-index', args=d)

                        triples_w_logprob.append(d)
                if ec_.have_edge_case:
                    logger.info(ec_.summary())
                # sanity check each triple is unique
                triples_tup = [
                    (d['sentence'], d['span'], d['entity_type'], d.get('index', 0), d.get('index_super', 0)) for d in triples_w_logprob]
                assert len(triples_tup) == len(set(triples_tup))
                log_n_save_triples_w_logprob(
                    triples_w_logprob=triples_w_logprob, entity_types=entity_types, output_path=out_path,
                    logger=logger, top_n_log='all', extra=extra)

                if challenging_sentences:
                    # select a subset of uncertain samples, i.e. low logprob, to be re-annotated, capped by logprob & count
                    count_cap = round(n_written * 0.2)  # 20% of the samples
                    lp_thresh = -5e-2
                    ch_sents = [sample_d['sentence'] for sample_d in samples_sorted if sample_d['logprob'] < lp_thresh]
                    n_ch = len(ch_sents)
                    msg_ = f'Challenging sentences w/ logprob < {pl.i(lp_thresh)}: {pl.i(ch_sents, indent=1)}'
                    if n_ch > count_cap:
                        ch_sents = ch_sents[:count_cap]  # not sorted by logprob, so can do this
                        msg_ = f'{msg_}, capped to smallest {pl.i(count_cap)} sentences'
                    _logger.info(msg_, extra=extra)
                    challenging_sentences['low-logprob'] += ch_sents
            if challenging_sentences and len(challenging_sentences) > 0:
                output_fnm = os_join(out_path, 'challenging-sentences.json')  # to be re-annotated w/ 2-stage gen
                log_n_save_challenging_samples(
                    challenging_samples=challenging_sentences, sample_kind='sentence', output_path=output_fnm, logger=_logger, extra=extra)
        ret.update({'#duplicate-sample-dropped': dset.n_dup_drop, '#sample-written': n_written})
        return ret

    if not lowercase:
        assert lc_ is None
        d_log.update(finish_single(dset=ori, out_path=output_path))
    else:
        assert lc_ is not None
        for lc__, dset_, kd in [(False, ori, 'original'), (True, lc_, 'lowercased')]:
            d_log[kd] = finish_single(dset=dset_, lc=lc__, out_path=output_path)

    # add output path
    d_log = dict(output_path=stem(output_path, top_n=3), **d_log)
    msg = f'Processed samples w/ {pl.i(d_log, indent=1)}'
    if time:
        if isinstance(time, Timer):
            time = time.end()
        msg = f'{msg} in {pl.i(time)}'
    logger.info(msg)
    samples = [NerReadableExample.from_d(d) for d in ori.readable]
    return NerProcessOutput(dataset=ori, samples=samples)


@dataclass
class LoadNerSamplesOutput:
    samples: List[NerReadableExample] = None
    sentences: List[str] = None
    sentence2sample: Dict[str, NerReadableExample] = None
    sample_logprobs: List[Dict[str, Any]] = None  # an NER sample dict w/ logprob
    entity_logprobs: List[Dict[str, Any]] = None  # each annotation (sentence, entity, type) triple w/ logprob
    path: str = None


def dataset_dir_name2ner_samples(
        dataset_name: str = None, dataset_dir_name: str = None, logprobs: bool = False, **kwargs
) -> LoadNerSamplesOutput:
    path = dataset_dir_name
    if not os.path.isdir(path):
        path = dataset_name2data_dir(dataset_name=dataset_name, input_dir=dataset_dir_name, **kwargs).path
    assert os.path.isdir(path)

    with open(os_join(path, 'readable-train.json'), 'r') as f:
        egs = json.load(f)['examples']
    ret = [NerReadableExample.from_d(**eg) for eg in egs]

    sents = [eg.sentence for eg in ret]
    assert len(ret) == len(set(ret))  # sanity check no duplicates
    assert len(sents) == len(set(sents))

    sample_lps, et_lps = None, None
    if logprobs:
        with open(os_join(path, 'logprobs-sample.json'), 'r') as f:
            sample_lps = json.load(f)
        with open(os_join(path, 'logprobs-triple.json'), 'r') as f:
            et_lps = json.load(f)
    return LoadNerSamplesOutput(
        samples=ret, sentences=sents, sentence2sample=dict(zip(sents, ret)), sample_logprobs=sample_lps, entity_logprobs=et_lps, path=path)


@dataclass
class AnnotationDiffOutput:
    span_is_different: bool = None
    different_span: Union[str, List[str]] = None
    type_is_different: bool = None
    different_type: Union[str, List[str]] = None


def analyze_diff(
        eg1: NerReadableExample = None, eg2: NerReadableExample = None, logger: logging.Logger = None, ignore_case: bool = True,
        color: bool = False, allow_no_diff: bool = False
) -> AnnotationDiffOutput:
    """
    :return: Whether the two examples differ in span and/or type
    """
    logger = logger or _logger

    if ignore_case:
        eg1, eg2 = eg1.to_lower(), eg2.to_lower()
    if allow_no_diff:
        # eq = eg1.to_lower() == eg2.to_lower() if ignore_case else eg1 == eg2
        if eg1 == eg2:
            return AnnotationDiffOutput(span_is_different=False, type_is_different=False)
    else:
        assert eg1 != eg2  # sanity check the point of this function
    assert eg1.sentence == eg2.sentence  # sanity check same sentence
    enms_1, enms_2 = eg1.entity_names, eg2.entity_names
    if ignore_case:
        enms_1_, enms_2_ = [enm.lower() for enm in enms_1], [enm.lower() for enm in enms_2]
        # diff_span = enms_1_ != enms_2_
        # sanity check no 2 entities differ in case
        assert len(set(enms_1)) == len(set(enms_1_)) and len(set(enms_2)) == len(set(enms_2_))
        enms_1, enms_2 = enms_1_, enms_2_
    # else:
    diff_span = enms_1 != enms_2  # can do this since the order of entities is fixed

    if diff_span:  # get the difference in span
        # may not always be a strict subset of the other
        set_enms1, set_enms2 = set(enms_1), set(enms_2)

        enms1_more, enms2_more = [], []
        if set_enms1 == set_enms2:  # the difference should be in count, for multi-occurring entities
            assert len(enms_1) != len(enms_2)
            # find that multi-occurring entity
            c1, c2 = Counter(enms_1), Counter(enms_2)
            # get the difference in counts between the two
            diff_count = {enm: abs(c1[enm] - c2[enm]) for enm in set_enms1 if c1[enm] != c2[enm]}
            assert len(diff_count) == 1  # sanity check just 1 multi-occurring entity
            enm, diff = diff_count.popitem()  # the sign of the difference tells which sample has more
            if diff > 0:
                enms1_more = [enm]
            else:
                enms2_more = [enm]
        else:
            only_in_1, only_in_2 = set_enms1 - set_enms2, set_enms2 - set_enms1
            n1, n2 = len(only_in_1), len(only_in_2)
            # if not (n1 > 0 or n2 > 0):
            #     sic(eg1, eg2, n1, n2, set_enms1, set_enms2, enms_1, enms_2, diff_span)
            assert n1 > 0 or n2 > 0  # sanity check

            if n1 > 0:
                # get the additional span in the original ordering
                enms1_more = [enm for enm in enms_1 if enm not in set_enms2]
            if n2 > 0:
                enms2_more = [enm for enm in enms_2 if enm not in set_enms1]
        n1, n2 = len(enms1_more), len(enms2_more)
        diff_enms = []
        if n1 > 0:
            if color:
                # enms = [pl.i(enm, c='y') for enm in enms]
                enms1_more = [pl.i(enms1_more, c='y')]
            diff_enms += enms1_more
        else:
            diff_enms += ['__NA__']

        if n1 > 0 or n2 > 0:  # insert a separation to indicate which sample the span is from
            diff_enms.append('=>')
        if n2 > 0:
            if color:
                # enms = [pl.i(enm, c='g') for enm in enms]
                enms2_more = [pl.i(enms2_more, c='g')]
            diff_enms += enms2_more
        else:
            diff_enms += ['__NA__']
        if color:
            diff_enms = pl.i(diff_enms, sep=' ')
    else:
        diff_enms = None

    # map entity span to type
    enm2et_1, enm2et_2 = dict(zip(enms_1, eg1.entity_types)), dict(zip(enms_2, eg2.entity_types))

    # within each sample, same span occurred many times may not have the same type, keep all distinct types in occurrence for comparison
    enm2cnt_1, enm2cnt_2 = Counter(enms_1), Counter(enms_2)  # get multi-occurred spans
    mul_1, mul_2 = [enm for enm, cnt in enm2cnt_1.items() if cnt > 1], [enm for enm, cnt in enm2cnt_2.items() if cnt > 1]
    it = [(enm, eg1, True) for enm in mul_1] + [(enm, eg2, False) for enm in mul_2]
    for enm, eg, is_1 in it:  # find the corresponding types for each occurrence
        if ignore_case:
            enm = enm.lower()
        idx = [i for i, e in enumerate(eg.entity_names) if e == enm]
        types = [eg.entity_types[i] for i in idx]
        types = tuple(dict.fromkeys(types))  # remove duplicates
        if len(types) > 1:
            d_log = dict(sample=1 if is_1 else 2, sentence=eg.sentence, span=enm, types=types)
            logger.info(f'Found multiple types for multi-occurred span w/ {pl.fmt(d_log)}')
        else:
            if len(types) != 1:
                sic(eg, enm, types, ignore_case)
            assert len(types) == 1
            types = types[0]
        d = (enm2et_1 if is_1 else enm2et_2)
        assert enm in d  # sanity check overwriting prior, single type
        d[enm] = types  # keep all types

    diff_type = False
    diff_pairs = []
    # for each span and type, check for the corresponding type in the other sample; keep in mind one sample
    enms = list(dict.fromkeys(enms_1 + enms_2))  # union of all spans
    # declare as different type if span is in both samples but type is different
    for enm in enms:
        et_1, et_2 = enm2et_1.get(enm, None), enm2et_2.get(enm, None)
        if et_1 is not None and et_2 is not None and et_1 != et_2:
            diff_type = True
            if color:
                enm = pl.i(enm, c='y')
                et_1 = pl.i(et_1, c='r')
                et_2 = pl.i(et_2, c='g')
            else:
                et_1, et_2 = pl.nc(et_1), pl.nc(et_2)
            diff_pairs.append(f'{enm} => {et_1} | {et_2}')
    if not (diff_span or diff_type):
        sic(eg1, eg2, diff_span, diff_type)
    assert diff_span or diff_type  # sanity check
    if not diff_type:
        diff_pairs = None
    return AnnotationDiffOutput(span_is_different=diff_span, type_is_different=diff_type, different_span=diff_enms, different_type=diff_pairs)


@dataclass
class CompareSamplePair:  # as input to `compare_samples`
    sample1: NerReadableExample = None
    sample2: NerReadableExample = None
    d_log: Dict[str, Any] = None


@dataclass
class CompareSamplesOutput:
    d_counts: Dict[str, int] = None  # stats on differences in entity span and type
    samples_diff_log: List[Dict[str, Any]] = None  # log of the pair of entity annotations & potential differences


def compare_samples(
        samples: List[CompareSamplePair] = None,
        samples1: List[NerReadableExample] = None, samples2: List[NerReadableExample] = None,
        d_logs: List[Dict[str, Any]] = None, logger: logging.Logger = None,
        prefix1: str = 'ori', prefix2: str = 'new', ignore_case: bool = True,
        allow_no_diff: bool = False, color: bool = True, verbose: bool = False, msg: str = None
):
    """
    Compare 2 lists of samples, pairs of samples in both lists should be potentially different annotations for the same sentence
    """
    it_d_log = None

    if samples is not None:
        n = len(samples)
        assert not (samples1 is not None or samples2 is not None)
        samples1 = [s.sample1 for s in samples]
        samples2 = [s.sample2 for s in samples]
        # check if `d_logs` is also provided
        if samples[0].d_log is not None:
            assert all(s.d_log is not None for s in samples)  # sanity check
            assert d_logs is None
            d_logs = [s.d_log for s in samples]

    else:
        assert samples1 is not None and samples2 is not None
        n = len(samples1)
        assert n == len(samples2)  # sanity check

    if d_logs is not None:
        assert len(d_logs) == n  # sanity check
        it_d_log = iter(d_logs)

    logger = logger or _logger
    pref_1, pref_2 = f'Y ({prefix1})', f'Y ({prefix2})'

    samples_diff = []
    n_diff_span, n_diff_type, n_no_diff = 0, 0, 0
    at_ = atc if color else at
    for eg1, eg2 in zip(samples1, samples2):
        d_log = next(it_d_log, dict()) if it_d_log is not None else dict()

        # sanity check same sentence
        if ignore_case:
            if eg1.sentence.lower() != eg2.sentence.lower():
                sic(eg1, eg2)
            assert eg1.sentence.lower() == eg2.sentence.lower()
        else:
            assert eg1.sentence == eg2.sentence

        d = {'X': eg1.sentence, pref_1: at_(eg1), pref_2: at_(eg2)}

        diff = analyze_diff(eg1, eg2, logger=logger, ignore_case=ignore_case, allow_no_diff=allow_no_diff, color=color)
        if not allow_no_diff:
            if not (diff.span_is_different or diff.type_is_different):
                raise NotImplementedError  # TODO: e.g. track # w/ no diff
        if diff.span_is_different:
            diff_span = diff.different_span
            if not color:
                diff_span = pl.nc(diff_span)
            d['diff-span'] = diff_span
            n_diff_span += 1
        if diff.type_is_different:
            d['diff-type'] = pl.i(diff.different_type) if color else pl.nc(diff.different_type)
            n_diff_type += 1
        if not diff.span_is_different and not diff.type_is_different:
            n_no_diff += 1

        d.update(d_log)
        samples_diff.append(d)
    d_counts = {'#different-span': n_diff_span, '#different-type': n_diff_type, '#total': n - n_no_diff}
    if n_no_diff > 0:
        d_counts['#no-difference'] = n_no_diff
    d_log = {'#different-samples': d_counts, 'samples': samples_diff}
    if verbose:
        msg = msg or 'Compared samples'
        logger.info(f'{msg} w/ {pl.i(d_log, indent=3, align_keys=3)}')
    return CompareSamplesOutput(d_counts=d_counts, samples_diff_log=samples_diff)


def compare_datasets(
        dataset1: LoadNerSamplesOutput, dataset2: LoadNerSamplesOutput, logger: logging.Logger = None,
        dataset_1_prefix: str = None, dataset_2_prefix: str = None, ignore_case: bool = True
):
    logger = logger or _logger
    dataset_1_prefix = dataset_1_prefix or 'ori'
    dataset_2_prefix = dataset_2_prefix or 'new'

    egs_1, sent2eg_1 = dataset1.samples, dataset1.sentence2sample
    egs_2, sent2eg_2 = dataset2.samples, dataset2.sentence2sample
    pref_1, pref_2 = f'Y ({dataset_1_prefix})', f'Y ({dataset_2_prefix})'

    log_overlap_no_match, log_no_overlap = [], []
    n_ovl, n_match, n_in_1, n_in_2 = 0, 0, 0, 0
    # diff_in_span, diff_in_type = 0, 0
    sents = list(dict.fromkeys([eg.sentence for eg in egs_1 + egs_2]))  # union sentences in all examples
    sample_pairs_match = []
    for sent in sents:
        eg_1, eg_2 = sent2eg_1.get(sent, None), sent2eg_2.get(sent, None)
        d = dict(X=sent)

        if eg_1 is not None and eg_2 is not None:
            if ignore_case:
                eq = eg_1.to_lower() == eg_2.to_lower()
            else:
                eq = eg_1 == eg_2
            if not eq:
                # d.update({pref_1: at(eg_1), pref_2: at(eg_2)})   # TODO: refactor w/ `compare_samples`
                # log_overlap_no_match.append(d)
                #
                # diff = analyze_diff(eg_1, eg_2, logger=logger)
                # if diff.span_is_different:
                #     diff_in_span += 1
                #     d['diff-span'] = pl.nc(diff.different_span, sep=' ')  # snap to 1 line
                # if diff.type_is_different:
                #     diff_in_type += 1
                #     d['diff-type'] = pl.nc(diff.different_type)
                sample_pairs_match.append(CompareSamplePair(sample1=eg_1, sample2=eg_2))
            else:
                n_match += 1
            n_ovl += 1
        else:
            if eg_1 is not None:
                d.update({pref_1: at(eg_1)})
                n_in_1 += 1
            else:
                assert eg_2 is not None
                d.update({pref_2: at(eg_2)})
                n_in_2 += 1
            log_no_overlap.append(d)
    n_no_ovl = len(log_no_overlap)
    assert n_no_ovl == n_in_1 + n_in_2  # sanity check
    d_log = {
        '#unique sentences': len(sents),
        '#overlapping samples': {'matching': n_match, 'non-matching': n_ovl - n_match, 'total': n_ovl},
        # '#non-matching samples': {'different span': diff_in_span, 'different type': diff_in_type, 'total': len(log_overlap_no_match)},
        '#non-overlapping samples': {f'#{dataset_1_prefix}': n_in_1, f'#{dataset_2_prefix}': n_in_2, 'total': n_no_ovl},
        # 'non-matching samples': log_overlap_no_match,
        # 'no-overlap samples': log_no_overlap,
    }
    logger.info(pl.i(d_log, indent=1))
    compare_samples(samples=sample_pairs_match, logger=logger, prefix1=dataset_1_prefix, prefix2=dataset_2_prefix, verbose=True)
    return d_log


MERGED_DIR_NAME = 'merged'


@dataclass
class _Sample2LogprobsDictOutput:
    index_sample: int = None
    indices_entity: List[int] = None
    sample_w_logprob: Dict[str, Any] = None
    triples_w_logprob: List[Dict[str, Any]] = None


class _Sample2LogprobsDict:
    """
    A utility for finding the logprob entries corresponding to each processed NER sample
    """
    def __init__(self, sample_logprobs: List[Dict[str, Any]] = None, entity_logprobs: List[Dict[str, Any]] = None):
        self.sample_logprobs = sample_logprobs
        self.entity_logprobs = entity_logprobs

    def __call__(self, sample: NerReadableExample, source: str = None) -> _Sample2LogprobsDictOutput:
        t = astuple(sample)
        idxs_sample = [i for i, d in enumerate(self.sample_logprobs) if t == _Sample2LogprobsDict._sample_w_logprob2tuple(d)]
        assert len(idxs_sample) == 1  # sanity check exactly one match
        idx_sample = idxs_sample[0]
        ret_sample = self.sample_logprobs[idx_sample]

        sent, enms, ets = sample.sentence, sample.entity_names, sample.entity_types
        idxs_entity = []

        # check for multi-occurring entities for special handling
        enm2c = Counter(enms)
        multi_occur = any(c > 1 for c in enm2c.values())
        multi_enm2idx = None
        if multi_occur:
            multi_enm2idx = {enm: 0 for enm, c in enm2c.items() if c > 1}

        for enm, et in zip(enms, ets):
            idxs_entity_ = [
                i for i, d in enumerate(self.entity_logprobs) if (sent, enm, et) == _Sample2LogprobsDict._triple_w_logprob2tuple(d)]
            if len(idxs_entity_) != 1:
                assert len(idxs_entity_) > 1 and multi_occur  # the only difference should be the `index`, for multi-occurring entities
                lst = [self.entity_logprobs[i] for i in idxs_entity_]
                # get the corresponding index
                idx = multi_enm2idx[enm]
                multi_enm2idx[enm] += 1  # we can increment and match cos how the triples were written, see `finish_ner_processing`
                lp = lst[idx]
                assert lp['index'] == idx
                idxs_entity.append(idxs_entity_[idx])  # in the next iteration of the same span, will be the next index

                # lst_ = deepcopy(lst)
                # for d in lst_:
                #     d.pop('index')
                # # sanity check the dicts should equal each other now
                # if not all(lst_[0] == d for d in lst_[1:]):
                #     sic(lst, lst_)
                # assert all(lst_[0] == d for d in lst_[1:])
            else:
                idxs_entity += idxs_entity_
        if len(idxs_entity) != len(enms):
            sic(sent, enms, ets, idxs_entity, len(idxs_entity))
        assert len(idxs_entity) == len(enms)  # sanity check exactly one match for each entity
        ret_entities = [self.entity_logprobs[i] for i in idxs_entity]

        if source:
            ret_sample['source'] = source
            for d in ret_entities:
                d['source'] = source
        return _Sample2LogprobsDictOutput(
            index_sample=idx_sample, sample_w_logprob=ret_sample,
            indices_entity=idxs_entity, triples_w_logprob=ret_entities
        )

    @staticmethod
    def _sample_w_logprob2tuple(d: Dict[str, Any]) -> Tuple:
        return d['sentence'], tuple(d['entity_names']), tuple(d['entity_types'])

    @staticmethod
    def _triple_w_logprob2tuple(d: Dict[str, Any]) -> Tuple:
        return d['sentence'], d['span'], d['entity_type']


def merge_datasets(
        dataset_name: str = None, logprobs: bool = None, dataset_dir_names: List[str] = None, dataset_prefixes: List[str] = None,
        output_dir_name: str = None, lowercase: bool = None, logger: logging.Logger = None
):
    """
    challenging/uncertain samples are collected and re-sent to LLM to be annotated/classified again
        new sentences can be processing failures from original dataset, or just considered challenging

    This function merges multiple processed NER datasets (from `process_completions`) together
    For each sample that appears in the earlier dataset, i.e. same sentence,
         the original sample annotations are overriden w/ new ones
    """
    logger = logger or _logger
    t = Timer()
    data_dir = dict(dataset_name=dataset_name, sub_dir=MERGED_DIR_NAME)
    output_path = dataset_name2data_dir(**data_dir, output_dir='NER-Dataset', output_postfix=output_dir_name, timestamp='short-date').path
    d_log = {'dataset-dir-names': dataset_dir_names, 'output-dir-name': output_dir_name, 'lowercase': lowercase}
    add_file_handler(logger=logger, file_path=os_join(output_path, f'merge-sample.log'))
    logger.info(f'Merging re-classified samples w/ {pl.i(d_log, indent=1)}')

    if len(dataset_dir_names) != 2:
        raise NotImplementedError
    dnm1, dnm2 = dataset_dir_names

    out1 = dataset_dir_name2ner_samples(dataset_name=dataset_name, dataset_dir_name=dnm1, logprobs=logprobs)
    out2 = dataset_dir_name2ner_samples(dataset_name=dataset_name, dataset_dir_name=dnm2, logprobs=logprobs)
    s_sent1, s_sent2 = set(out1.sentences), set(out2.sentences)
    assert len(out1.samples) == len(s_sent1) and len(out2.samples) == len(s_sent2)  # sanity check no duplicate sentences
    n_ovl = len(s_sent1 & s_sent2)

    if dataset_prefixes:
        assert len(dataset_prefixes) == len(dataset_dir_names)  # sanity check
        pref1, pref2 = dataset_prefixes
    else:
        pref1, pref2 = 'ori', 'new'
    d_log_compare = compare_datasets(dataset1=out1, dataset2=out2, logger=logger, dataset_1_prefix=pref1, dataset_2_prefix=pref2)
    d_log_count = {'#samples-ori': len(out1.sentences), '#samples-new': len(out2.sentences), '#samples-with-sentence-overlap': n_ovl}

    sents = list(dict.fromkeys(out1.sentences + out2.sentences))
    ret = []
    if logprobs:
        ret_lps, ret_et_lps = [], []
        s2ld1 = _Sample2LogprobsDict(sample_logprobs=out1.sample_logprobs, entity_logprobs=out1.entity_logprobs)
        s2ld2 = _Sample2LogprobsDict(sample_logprobs=out2.sample_logprobs, entity_logprobs=out2.entity_logprobs)
    else:
        ret_lps, ret_et_lps = None, None
        s2ld1, s2ld2 = None, None
    for sent in sents:
        samp_ori, samp_new = out1.sentence2sample.get(sent), out2.sentence2sample.get(sent)

        # Always prefer the new sample if available, cases:
        #   1> sample not found in original dataset, due to e.g. wasn't able to extract NER sample
        #   2> sample is re-classified
        #       Note in such case we always pick the newer one, whether CLS labels are different
        #           So that logprobs will use the new one
        if samp_ori is None:
            assert samp_new is not None
            samp_add = samp_new
            s2ld, src = s2ld2, pref2
            # samp_new != samp_ori  # note this may not be the case
        else:
            samp_add = samp_ori
            s2ld, src = s2ld1, pref1
        ret.append(samp_add)

        if logprobs:  # find the corresponding logprobs and add
            s2ld_out = s2ld(sample=samp_add, source=src)
            ret_lps.append(s2ld_out.sample_w_logprob)
            ret_et_lps += s2ld_out.triples_w_logprob

    n_ovd = d_log_compare['#overlapping samples']['non-matching']
    n_add = d_log_compare['#non-overlapping samples'][f'#{pref2}']
    d_log_count.update({'#samples-overridden': n_ovd, '#samples-added': n_add})
    logger.info(f'{pl.i(n_ovd)} samples had entity annotations overridden ')

    if logprobs:  # also save the sample-wise and entity-wise logprobs
        log_n_save_samples_w_logprob(samples_w_logprob=ret_lps, output_path=output_path, logger=logger)
        ets = sconfig(f'datasets.{dataset_name}.readable-entity-types')
        log_n_save_triples_w_logprob(
            triples_w_logprob=ret_et_lps, entity_types=ets, output_path=output_path, logger=logger, top_n_log='all')

    return finish_ner_processing(
        samples=ret, logger=logger,  dataset_name=dataset_name, entity_types=sconfig(f'datasets.{dataset_name}.readable-entity-types'),
        dedup=True, lowercase=lowercase, output_path=output_path, d_log=d_log_count, time=t
    )


if __name__ == '__main__':
    # dnm = 'conll2003'
    # dnm = 'conll2003-no-misc'
    # dnm = 'job-desc'
    dnm = 'mit-movie'
    # dnm = 'mit-restaurant'
    # dnm = 'job-stack'
    # dnm = 'wiki-gold-no-misc'
    # dnm = 'ncbi-disease'

    def check_write_tag2id():
        output_path = os_join(pu.proj_path, 'original-dataset', dnm)

        assert dnm in ['conll2003', 'conll2003-no-misc', 'mit-movie']
        dl = DatasetLoader(dataset_name=dnm)
        labels = [dl.label_map[lb] for lb in dl.labels]
        sic(labels)
        # t2i = ner_labels2tag2index(labels)
        # sic(t2i)
        write_tag2id(entity_types=labels, path=output_path)
    check_write_tag2id()
