import os
import json
import random
import urllib.request
from os.path import join as os_join
from typing import List, Tuple, Dict, Union, Any
from dataclasses import asdict
from collections import Counter

from stefutil import get_logger, pl, stem
from src.util.ner_example import NerBioExample


__all__ = [
    'download_raw_file', 'download_raw_files', 'samples2jsonl',
    'load_conll_single', 'load_conll_style',
    'split_data', 'write_dataset', 'entity_type_dist', 'dataset_stats'
]


logger = get_logger(__name__)


def download_raw_file(url: str = None, output_path: str = None, output_filename: str = None):
    """
    download files from the web using `urlretrieve`
    """
    os.makedirs(output_path, exist_ok=True)
    file_path = os_join(output_path, output_filename)
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url=url, filename=file_path)
        logger.info(f'Downloaded from {pl.i(url)} to {pl.i(stem(file_path, top_n=3))}')


def download_raw_files(url2path: Dict[str, Dict[str, str]] = None, output_path: str = None):
    for k1, url_ in url2path.items():
        dset_path = os_join(output_path, 'raw', k1)
        for k2, url in url_.items():
            fnm = f'{k2}.txt'
            download_raw_file(url=url, output_path=dset_path, output_filename=fnm)


def samples2jsonl(samples: List[NerBioExample] = None, file_path: str = None):
    with open(file_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample)) + '\n')


def split_data(data: List[NerBioExample] = None, split_size: int = 0.1, seed: int = 42) -> Tuple[List[NerBioExample], List[NerBioExample]]:
    """
    :return: Split data into 2 partitions
    """
    random.seed(seed)
    random.shuffle(data)
    d2_sz = round(split_size * len(data))
    d2_idxs = set(random.sample(range(len(data)), d2_sz))  # sample random indices for val set
    return [data[i] for i in range(len(data)) if i not in d2_idxs], [data[i] for i in d2_idxs]


def has_upper(txt: str) -> bool:
    """
    :return: if txt has any uppercase letters
    """
    return any(c.isupper() for c in txt)


def load_conll_single(rows: List[str], token_first: bool = True) -> NerBioExample:
    # tag & token, separated by tab
    rows = [r.split('\t') for r in rows]
    assert len(rows) > 0
    toks, tags = zip(*rows)
    if not token_first:
        toks, tags = tags, toks
    ret = NerBioExample.from_tokens_n_tags(tokens=toks, tags=tags)
    return ret


def load_conll_style(file_path: str = None, token_first: bool = True, check_all_lower: bool = False) -> List[NerBioExample]:
    """
    Load a conll-style dataset, each token-tag pair on a new line, separated by tab
    """
    lines = [ln.strip() for ln in open(file_path, 'r').readlines()]
    idxs_sep = [i for i, line in enumerate(lines) if line == '']
    idxs_strt = [0] + [i + 1 for i in idxs_sep]
    idxs_end = [i for i in idxs_sep] + [len(lines) - 1]
    sents = [lines[i:j] for i, j in zip(idxs_strt, idxs_end)]

    assert len(sents[-1]) == 0
    sents = sents[:-1]  # drop the last newline

    ret = [load_conll_single(rows, token_first=token_first) for rows in sents]
    if check_all_lower:
        assert all(not has_upper(eg.sentence) for eg in ret)
    return ret


def write_dataset(
        train: List[NerBioExample] = None, dev: List[NerBioExample] = None, test: List[NerBioExample] = None,
        output_path: str = None, seed: int = 42
):
    if dev is None:
        train, val = split_data(data=train, seed=seed)  # split train into train and dev
    else:
        val = dev
    d_samples = dict(train=train, val=val, test=test)

    os.makedirs(output_path, exist_ok=True)
    for split, sps in d_samples.items():
        pa = os_join(output_path, f'{split}.jsonl')
        samples2jsonl(samples=sps, file_path=pa)
        logger.info(f'{pl.i(split)} split written to {pl.i(stem(pa, top_n=3))}')
    split2sz = {split: len(sps) for split, sps in d_samples.items()}
    split2sz['total'] = sum(split2sz.values())
    logger.info(f'Dataset sizes: {pl.i(split2sz)}')


def load_dataset_samples(dataset_path: str = None) -> List[NerBioExample]:
    return [NerBioExample(**json.loads(line)) for line in open(dataset_path, 'r')]


def entity_type_dist(
        data: Union[str, List[NerBioExample]] = None, entity_types: Union[List[str], Dict[str, str]] = None
) -> Dict[str, Dict[str, int]]:
    """
    :return: Entity type distribution
    """
    if isinstance(data, str):
        data = load_dataset_samples(dataset_path=data)
    if entity_types:
        if isinstance(entity_types, dict):
            et2et = entity_types
            entity_types = list(entity_types.keys())
        else:
            assert isinstance(entity_types, list)
            et2et = None

        toks = [t[2:] for d in data for t in d.ner_tags if t.startswith('B-')]  # drop `B-` prefix; the entire entity counts as 1
        assert all(t in entity_types for t in toks)  # sanity check no tags are ignored
        c = Counter(toks)
        c = {et: c[et] for et in entity_types}

        if et2et:
            c = {et2et[k]: v for k, v in c.items()}
    else:  # count all entity types
        c = dict(Counter(t for sample in data for t in sample.ner_tags))
        c = {t[2:]: v for t, v in c.items() if t.startswith('B-')}  # keep only keys that starts w/ `B-`
    # normalize and round to 2 decimal places percents
    c_norm = {k: round(v / sum(c.values()) * 100, 2) for k, v in c.items()}
    return {'#entity-types': len(c), 'counts': c, 'percents': c_norm}


def dataset_stats(dataset_path: str = None, splits: List[str] = None, sum_train_n_val: bool = True) -> Dict[str, Any]:
    splits = splits or ['train', 'val', 'test']
    split2samples = {split: load_dataset_samples(dataset_path=os_join(dataset_path, f'{split}.jsonl')) for split in splits}
    if sum_train_n_val:
        assert 'train' in splits and 'val' in splits
        split2samples['train'] += split2samples.pop('val')

    def _samples2entity_counts(data: List[NerBioExample] = None) -> int:
        return sum(t.startswith('B-') for sample in data for t in sample.ner_tags)
    return {
        '#sentence': {split: len(samples) for split, samples in split2samples.items()},
        '#entity': {split: _samples2entity_counts(samples) for split, samples in split2samples.items()},
        'test-set-dist': entity_type_dist(data=split2samples['test'])
    }
