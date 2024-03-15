"""
For mit-movie dataset from *Asgard A portable architecture for multilingual dialogue systems*
"""

import os
from os.path import join as os_join

from stefutil import get_logger, pl
from src.util import pu
from src.write_original_dataset.raw.util import download_raw_files, load_conll_style, write_dataset, entity_type_dist, dataset_stats


__all__ = ['write']


logger = get_logger(__name__)


URL_BASE = 'https://groups.csail.mit.edu/sls/downloads/movie'

URL = dict(
    eng=dict(
        train='engtrain.bio',
        test='engtest.bio'
    ),
    trivia10k13=dict(
        train='trivia10k13train.bio',
        test='trivia10k13test.bio'
    )
)

URL = {nm: {split: os_join(URL_BASE, fnm) for split, fnm in v.items()} for nm, v in URL.items()}


def write(dataset_name: str = 'mit-movie', seed: int = 42):
    dataset_path = os_join(pu.proj_path, 'original-dataset', dataset_name)
    os.makedirs(dataset_path, exist_ok=True)

    download_raw_files(url2path=URL, output_path=dataset_path)

    def convert2jsonl(name: str = 'eng'):
        dset_path = os_join(dataset_path, 'raw', name)
        tr_path = os_join(dset_path, 'train.txt')
        ts_path = os_join(dset_path, 'test.txt')

        tr_samples = load_conll_style(file_path=tr_path, token_first=False, check_all_lower=True)
        ts_samples = load_conll_style(file_path=ts_path, token_first=False, check_all_lower=True)
        write_dataset(train=tr_samples, test=ts_samples, output_path=os_join(dataset_path, name), seed=seed)
    convert2jsonl(name='eng')
    convert2jsonl(name='trivia10k13')


if __name__ == '__main__':
    dnm = 'mit-movie'

    write(dataset_name=dnm)

    def check_entity_type_dist(name: str = 'eng'):
        dataset_path = os_join(pu.proj_path, 'original-dataset', dnm)
        path = os_join(dataset_path, name, 'test.jsonl')
        print(pl.i(entity_type_dist(data=path)))
    # check_entity_type_dist()

    def check_stats(name: str = 'eng'):
        dataset_path = os_join(pu.proj_path, 'original-dataset', dnm)
        path = os_join(dataset_path, name)
        out = dataset_stats(dataset_path=path)
        print(pl.i(out))
        print(pl.fmt(out))
    # check_stats(name='eng')
    check_stats(name='trivia10k13')
