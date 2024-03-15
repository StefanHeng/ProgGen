"""
For mit-restaurant dataset from *Asgard A portable architecture for multilingual dialogue systems*
"""

import os
from os.path import join as os_join

from stefutil import get_logger, pl
from src.util import sconfig, pu
from src.write_original_dataset.raw.util import download_raw_file, load_conll_style, write_dataset, entity_type_dist


__all__ = ['write']


logger = get_logger(__name__)


URL_BASE = 'https://groups.csail.mit.edu/sls/downloads/restaurant'

_nm = 'restaurant'
URL = dict(
    train=f'{_nm}train.bio',
    test=f'{_nm}test.bio'
)
URL = {split: os_join(URL_BASE, fnm) for split, fnm in URL.items()}


def write(dataset_name: str = 'mit-restaurant', seed: int = 42):
    dataset_path = os_join(pu.proj_path, 'original-dataset', dataset_name)
    os.makedirs(dataset_path, exist_ok=True)

    _dset_path = os_join(dataset_path, 'raw')

    for split, url in URL.items():
        fnm = f'{split}.txt'
        download_raw_file(url=url, output_path=_dset_path, output_filename=fnm)

    tr_path = os_join(_dset_path, 'train.txt')
    ts_path = os_join(_dset_path, 'test.txt')

    tr_samples = load_conll_style(file_path=tr_path, token_first=False, check_all_lower=True)
    ts_samples = load_conll_style(file_path=ts_path, token_first=False, check_all_lower=True)
    write_dataset(train=tr_samples, test=ts_samples, output_path=dataset_path, seed=seed)


if __name__ == '__main__':
    dnm = 'mit-restaurant'

    # write(dataset_name=dnm)

    def check_entity_type_dist(split: str = 'test'):
        dataset_path = os_join(pu.proj_path, 'original-dataset', dnm)
        path = os_join(dataset_path, f'{split}.jsonl')
        ets = sconfig(f'datasets.{dnm}.label2readable-label')
        # ets = None
        dist = entity_type_dist(data=path, entity_types=ets)
        print(pl.i(dist))
    # check_entity_type_dist(split='train')
    check_entity_type_dist(split='test')
