import os
import random
from os.path import join as os_join
from typing import Dict, List, Tuple, Union, TypeVar, Any
from dataclasses import dataclass

from stefutil import *
from src.util._paths import *


__all__ = [
    'sconfig', 'pu', 'save_fig',
    'sample_fmt2data_fmt', 'sample_fmt2original_data_fmt',
    'abbreviate_format', 'dataset_meta', 'dataset_name2data_dir', 'dataset_name2model_dir',
    'sample_single', 'sample_few',
    'span_pair_overlap', 'spans_overlap',
]


_logger = get_logger(__name__)


config_path: str = os_join(BASE_PATH, PROJ_DIR, PKG_NM, 'util', 'config.json')
sconfig = SConfig(config_file=config_path).__call__
pu = PathUtil(
    base_path=BASE_PATH, project_dir=PROJ_DIR, package_name=PKG_NM, dataset_dir=DSET_DIR, model_dir=MODEL_DIR,
    within_proj=True, makedirs=['plot', 'model']
)
pu.generated_data_path = os_join(pu.proj_path, 'generated_data')
save_fig = pu.save_fig


ca.cache_options(
    display_name='NER Dataset Name', attr_name='dataset_name', options=list(sconfig('datasets').keys())
)
ca.cache_options(
    display_name='NER Sample Demo/Completion format', attr_name='sample_format',
    options=['natural-pair', 'natural-pair-v2', 'natural-inline', 'natural-inline-v2', 'bio-list', 'bio-list-v2', 'bio-line']
)
ca.cache_options(
    display_name='NER Processed Data format', attr_name='data_format', options=['natural', 'bio']
)
ca.cache_options(
    display_name='NER Dataset Load format', attr_name='original_data_format', options=['readable', 'span', 'bio']
)
ca.cache_options(
    display_name='Prompt Insert Type', attr_name='insert', options=['defn', 'schema']
)
ca.cache_options(
    display_name='Visualize Sample Type', attr_name='plot_type', options=['sentence', 'entity']
)
ca.cache_options(display_name='Diverse Entity Scheme', attr_name='diverse_entity', options=['independent', 'seeded', 'mixed'])


def sample_fmt2data_fmt(sample_format: str) -> str:
    """
    Convert sample format to data format
    """
    ca(sample_format=sample_format)
    if 'natural' in sample_format:
        return 'natural'
    else:
        assert 'bio' in sample_format
        return 'bio'


def sample_fmt2original_data_fmt(sample_format: str) -> str:
    """
    Convert sample format to data format
    """
    ca(sample_format=sample_format)
    if sample_format in ['natural-pair', 'natural-pair-v2']:
        return 'readable'
    elif sample_format in ['natural-inline', 'natural-inline-v2']:
        return 'span'
    else:
        assert 'bio' in sample_format
        return 'bio'


def abbreviate_format(val: str = None) -> str:
    """
    Abbreviate strings for file path write
    """
    assert val is not None
    ca(sample_format=val)
    if val == 'natural-pair':
        return 'n-p'
    elif val == 'natural-pair-v2':
        return 'n-p2'
    elif val == 'natural-inline':
        return 'n-i'
    elif val == 'natural-inline-v2':
        return 'n-i2'
    elif val == 'bio-list':
        return 'b-l'
    elif val == 'bio-list-v2':
        return 'b-l2'
    else:
        assert val == 'bio-line'
        return 'b-ln'


def dataset_meta(
        sample_format: str = None, n_list: int = None,
        n_annotate: int = None, n_identify: int = None, n_classify: int = None, n_correct: int = None,
        with_unlabeled: Union[bool, int] = None,
        diverse_context: Union[bool, Dict[str, Any]] = None, drop_prob: float = None,
        diverse_entity: Union[bool, str] = None,
        as_passage: bool = None, subsample_demo: bool = False,
        insert: str = None, cot: bool = False, lowercase: bool = False,
        postfix: str = None,
) -> str:
    ret = dict()
    if sample_format:
        ret['fmt'] = abbreviate_format(sample_format)
    if n_list:
        ret['#l'] = n_list
    if n_annotate:
        ret['#a'] = n_annotate
    if n_identify:
        ret['#i'] = n_identify
    if n_classify:
        ret['#c'] = n_classify
    if n_correct:
        ret['#cr'] = n_correct
    if with_unlabeled:
        ret['#u-lb'] = with_unlabeled
    dc, de = diverse_context, diverse_entity
    if dc or de:
        if isinstance(dc, dict):
            if 'presets' in dc:
                dc['pst'] = dc.pop('presets')
            if 'group_attributes' in dc:
                dc['grp'] = dc.pop('group_attributes')
        if drop_prob:
            dc = dict(dp=drop_prob, **dc) if isinstance(dc, dict) else dict(dp=drop_prob)

        if isinstance(de, str):
            ca(diverse_entity=de)
            de = de[0]
            # de = 's' if de == 'seeded' else 'i'  # one of ['independent', 'seeded']
        if dc and not de:
            ret['dc'] = dc
        elif not dc and de:
            ret['de'] = de
        else:  # group key to denote overall diversity
            ret['d'] = dict(c=dc, e=de)
    if as_passage:
        ret['psg'] = True
    if insert:
        assert insert in [True, 'defn', 'schema']
        if insert is True:
            ret['ins'] = True
        else:
            ret['ins'] = 'def' if insert == 'defn' else 'scm'
    if cot:
        ret['cot'] = True
    if subsample_demo:
        ret['sub'] = True
    if lowercase:
        ret['lc'] = True

    if postfix:
        return f'{pl.pa(ret)}_{postfix}' if ret != dict() else postfix
    else:
        return pl.pa(ret) if ret != dict() else None


@dataclass
class DirectoryOutput:
    path: str = None
    base_path: str = None


def dataset_name2data_dir(
        dataset_name: str = 'conll2003', sub_dir: str = None,
        input_dir: str = None,
        output_dir: str = None, output_postfix: str = None,
        timestamp: Union[bool, str] = True
):
    assert not (input_dir is not None and output_dir is not None)  # at most one of them can be given
    dirs = [pu.generated_data_path, dataset_name.replace('-', '_')]
    if sub_dir is not None:
        dirs.append(sub_dir)
    base_path = os_join(*dirs)
    if input_dir is not None:
        dirs.append(input_dir)
    elif output_dir is not None:
        if timestamp:
            fmt = timestamp if isinstance(timestamp, str) else 'short-full'
            output_dir = f'{now(for_path=True, fmt=fmt)}_{output_dir}'
        if output_postfix:
            output_dir = f'{output_dir}_{output_postfix}'
        dirs.append(output_dir)
    ret = os_join(*dirs)
    if output_dir is not None:
        os.makedirs(ret, exist_ok=True)
    ret: str
    base_path: str
    return DirectoryOutput(path=ret, base_path=base_path)


def dataset_name2model_dir(dataset_name: str = 'conll2003', model_dir: str = None):
    base_path = os_join(pu.model_path, dataset_name.replace('-', '_'))

    path = base_path
    if model_dir:
        path = os_join(base_path, model_dir)
        os.makedirs(path, exist_ok=True)
    return DirectoryOutput(path=path, base_path=base_path)


T = TypeVar('T')


def sample_single(lst: List[T], generator: Union[int, random.Random] = None) -> T:
    """
    Syntactic sugar
    """
    if not isinstance(lst, list) and len(lst) > 0:
        raise ValueError(f'Input must be a non-empty list, but got {pl.i(lst)}')
    gen = get_random_generator(generator=generator)
    return gen.sample(lst, 1)[0]


def sample_few(lst: List[T], min_: int = 0, max_: int = None, generator: Union[int, random.Random] = None) -> List[T]:
    """
    Randomly sample a few elements from a list
    The size to sample is also randomly sampled
    """
    n = len(lst)
    max_ = max_ or len(lst)
    if max_ > n:
        max_ = n
    gen = get_random_generator(generator=generator)
    n = gen.randint(min_, max_)
    return gen.sample(lst, n)


def span_pair_overlap(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
    """
    Check if 2 spans overlap
    """
    s1, e1 = span1
    s2, e2 = span2
    return not (e1 <= s2 or e2 <= s1)


def spans_overlap(spans: List[Tuple[int, int]]) -> bool:
    """
    Check if any 2 spans in a list of spans overlap
    """
    spans = sorted(spans)  # sort keywords by appearance order
    return any(span_pair_overlap(s1, s2) for s1, s2 in zip(spans[:-1], spans[1:]))  # thus, can optimize by checking only adjacent spans
