"""
Convert original dataset to our NER training format

Intended for writing validation and test sets for the NER training pipeline

Even for datasets already downloaded locally, still need to write it again for NER input format consistency

Intended for converting `ner_tags` to `labels`, and dropping duplicate samples
"""

import json
import random
from os.path import join as os_join
from typing import List, Tuple, Union

from stefutil import *
from src.util import *
from src.util.ner_example import *
from src.data_util.dataset import *


__all__ = ['write_train', 'write_val', 'write_test']


logger = get_logger(__name__)


# Load the data in the same format as the raw dataset
# this prevents 1) tokens => sentence with `tokens2sentence` and then 2) extracting tokens with `split_text_with_terms`
#   causes tokens not split properly
DATA_FORMAT = 'bio'


def dataset_name2output_path(dataset_name: str = None) -> str:
    return os_join(pu.proj_path, 'original-dataset', dataset_name)


def get_samples_n_fnm(dl: DatasetLoader = None, n: int = None, filename: str = None, split: str = None) -> Tuple[List[NerExample], str]:
    if n:
        assert n <= len(dl)
        random.seed(42)
        idxs = random.sample(range(len(dl)), n)

        samples = dl[idxs]

        assert filename is not None
        fnm = filename
    else:
        samples = dl[:]
        fnm = filename or f'{split}-all'
    return samples, fnm


def write_train(
        dataset_name: str = None,
        few: Union[bool, str] = True, n: int = None, filename: str = None, shuffle: bool = False, include_none_samples: bool = False,
        sentence_only: bool = None, exclude: List[int] = None
):
    """
    Allowed a few training samples from the original dataset

    :param dataset_name: dataset name
    :param few: whether to write a few samples or all the samples
    :param n: If given, write `n` original samples or pass `n` to get few-shot demo call
    :param filename: If given, write to `filename`
    :param shuffle: If True, shuffle the samples before writing
        Intended for few-shot samples
    :param include_none_samples: If True, include samples with no entities
        Intended for few-shot samples
    :param sentence_only: If True, only write the sentences
        Intended for LLM ablation experiments
    :param exclude: If given, exclude the samples with the given indices
        Intended for a setup including unlabeled sentences, where the labeled n-shot samples are excluded
    """
    dl = DatasetLoader(dataset_name=dataset_name, split='train', data_format=DATA_FORMAT)
    logger.info(f'Total #samples in train split: {pl.i(len(dl))}')

    if exclude:
        raise NotImplementedError

    if few:
        n = n or 5  # 5-shot or 5 demo samples
        tp = few if isinstance(few, str) else 'n'
        fnm = 'train-few' if tp == 'n' else f'train-{n}-shot'
        if shuffle:
            fnm = f'{fnm}-shuffled'
        if include_none_samples:
            fnm = f'{fnm}+neg'
        samples = dl.get_few_demo_samples(n_demo=n, demo_type=tp, shuffle=shuffle, include_none_samples=include_none_samples)
    else:
        samples, fnm = get_samples_n_fnm(dl=dl, n=n, filename=filename, split='train')
    out_path = dataset_name2output_path(dataset_name=dataset_name)
    if sentence_only:
        samples = de_duplicate_ner_samples(samples=samples).samples
        sents = [s.sentence for s in samples]
        with open(os_join(out_path, f'{fnm}-sentences.json'), 'w') as f:
            json.dump(sents, f, indent=4)
    else:
        samples2train_dataset(examples=samples, write=True, write_fnm=fnm, dedup=True, output_path=out_path, data_format=DATA_FORMAT)


def write_val(dataset_name: str = None, few: bool = True, n: int = None, filename: str = None):
    """
    Only get 5 val samples from original dataset for validation
    """
    if dataset_name in DatasetLoader.from_hf:
        split = 'validation'
    elif dataset_name in DatasetLoader.from_local_bio_file:
        split = 'val'
    else:
        raise NotImplementedError
    dl = DatasetLoader(dataset_name=dataset_name, split=split, data_format=DATA_FORMAT)
    logger.info(f'total #samples: {pl.i(len(dl))}')

    if few:
        n = n or 5
        samples, fnm = dl[:n], 'val-few' or filename
    else:
        samples, fnm = get_samples_n_fnm(dl=dl, n=n, filename=filename, split='val')
    out_path = dataset_name2output_path(dataset_name=dataset_name)
    samples2train_dataset(examples=samples, write=True, write_fnm=fnm, dedup=True, output_path=out_path, data_format=DATA_FORMAT)


def write_test(dataset_name: str = None):
    dl = DatasetLoader(dataset_name=dataset_name, split='test', data_format=DATA_FORMAT)
    logger.info(f'total #samples: {pl.i(len(dl))}')

    samples = dl[:]
    if dataset_name in DatasetLoader.from_hf:
        fnm = 'test'
    elif dataset_name in DatasetLoader.from_local_bio_file:
        fnm = 'test-all'
    else:
        raise NotImplementedError

    if dataset_name == 'ncbi-disease':  # bug???
        assert samples[940].sentence == ''
        del samples[940]
    assert all(s.sentence != '' for s in samples)
    out_path = dataset_name2output_path(dataset_name=dataset_name)
    samples2train_dataset(examples=samples, write=True, write_fnm=fnm, dedup=True, output_path=out_path, data_format=DATA_FORMAT)


if __name__ == '__main__':
    sic.output_width = 120

    # dnm = 'conll2003'
    # dnm = 'conll2003-no-misc'
    # dnm = 'job-desc'
    # dnm = 'mit-movie'
    dnm = 'mit-restaurant'
    # dnm = 'job-stack'
    # dnm = 'wiki-gold-no-misc'
    # dnm = 'ncbi-disease'

    # write_train(few='n')
    # write_train(few='n-shot')
    # write_train(few='n-shot', shuffle=True)
    # write_train(few='n-shot', n=2, shuffle=True, include_none_samples=True)
    # write_train(few='n-shot', n=1, shuffle=True, include_none_samples=True)
    # write_train(few='n-shot', n=1, shuffle=True, include_none_samples=False)
    # write_train(few=False, n=1_000, filename='train-1k')  # intended for equal-sample-size upper bound performance
    # write_train(few=False, n=1_350, filename='train-1350')  # 90% train split of 1.5K equal-sample-size comparison
    # write_train(few=False, n=None, filename='train-1.1k')  # since the entire train set of `wiki-gold` doesn't reach 1.35K
    # write_train(few=False, n=1_000, filename='train-1k', sentence_only=True)  # intended for LLM ablation experiments
    # write_train(few=False)  # intended for full-supervision comparison, getting upper bound performance

    # write_val()
    # write_val(few=False)
    # write_val(few=False, n=150, filename='val-150')  # 10% val split of 1.5K equal-sample-size comparison

    write_test(dataset_name=dnm)
