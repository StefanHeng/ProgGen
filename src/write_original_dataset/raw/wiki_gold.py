"""
Processing the WikiGold dataset

The authors didn't seem to release the dataset
The most official version seems to be [this](https://github.com/juand-r/entity-recognition-datasets/tree/master/data/wikigold)
"""

import os
from os.path import join as os_join
from typing import List

from stefutil import *
from src.util import *
from src.util.ner_example import NerBioExample
from src.write_original_dataset.raw.util import *


__all__ = ['write']


# We use the processed files [here](https://github.com/yumeng5/RoSTER/tree/main/data/wikigold)

URL_BASE = 'https://raw.githubusercontent.com/yumeng5/RoSTER/main/data/wikigold'
URL = dict(
    train=dict(text='train_text.txt', label='train_label_true.txt'),
    val=dict(text='valid_text.txt', label='valid_label_true.txt'),
    test=dict(text='test_text.txt', label='test_label_true.txt')
)
URL = {split: {tp: os_join(URL_BASE, fnm) for tp, fnm in v.items()} for split, v in URL.items()}


def write(dataset_name: str = 'wiki-gold'):
    dataset_path = os_join(pu.proj_path, 'original-dataset', dataset_name)
    os.makedirs(dataset_path, exist_ok=True)

    download_raw_files(url2path=URL, output_path=dataset_path)

    def split2samples(split: str = 'train') -> List[NerBioExample]:
        path = os_join(dataset_path, 'raw', split)
        txt_path, lb_path = os_join(path, 'text.txt'), os_join(path, 'label.txt')
        txts = open(txt_path, 'r').readlines()
        lbs = open(lb_path, 'r').readlines()
        assert len(txts) == len(lbs)

        def get_single(txt: str, lb: str) -> NerBioExample:
            txt, lb = txt.strip(), lb.strip()
            toks, tags = txt.split(), lb.split()  # split by space
            assert len(toks) == len(tags)  # sanity check
            return NerBioExample.from_tokens_n_tags(tokens=toks, tags=tags)
        return [get_single(txt=txt, lb=lb) for txt, lb in zip(txts, lbs)]

    tr_samples = split2samples(split='train')
    vl_samples = split2samples(split='val')
    ts_samples = split2samples(split='test')
    write_dataset(train=tr_samples, dev=vl_samples, test=ts_samples, output_path=dataset_path)


if __name__ == '__main__':
    dnm = 'wiki-gold'

    write(dataset_name=dnm)

    def check_entity_type_dist():
        dataset_path = os_join(pu.proj_path, 'original-dataset', dnm)
        path = os_join(dataset_path, 'test.jsonl')
        dist = entity_type_dist(data=path, entity_types=sconfig(f'datasets.{dnm}.label2readable-label'))
        print(pl.i(dist))
    check_entity_type_dist()
