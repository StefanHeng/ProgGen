"""
Includes
1. Category options for all attribute Dimensions on X, and
2. Named entity options (Y)
"""

import os
import json
import copy
import random
from os.path import join as os_join
from typing import List, Dict, Union, Any

from stefutil import get_logger, pl, ca, sic, now, get
from src.util import pu, dataset_name2data_dir
from src.generate.diversify.util import (
    DIVERSE_DNM, DIVERSE_CONTEXT_DNM, DIVERSE_ENTITY_DNM, ENTITY_KEY, ENTITY_KEY_SEEDED, ENTITY_KEY_MIXED
)
from src.generate.diversify.attr2cats_dicts import (
    conll2003_a2c, mit_movie_a2c, job_stack_a2c, wiki_gold_a2c, mit_restaurant_a2c
)


__all__ = ['Attribute2Categories']


logger = get_logger(__name__)


CategoryOptions = Union[List[str], Dict[str, List[str]]]


class Attribute2Categories:
    dataset_name2attribute2categories = {
        'conll2003': conll2003_a2c, 'mit-movie': mit_movie_a2c, 'job-stack': job_stack_a2c, 'wiki-gold': wiki_gold_a2c,
        'mit-restaurant': mit_restaurant_a2c,
    }
    dataset_name2default_x_attributes = {
        # 'conll2003': ['news-category', 'length', 'writing-style', 'location'],
        # 'conll2003': ['news-category', 'writing-style'],
        # location is an entity type in conll2003, adding additional location requirement may be conflicting w/ diverse entity
        'conll2003': ['news-category', 'writing-style'],

        # 'mit-movie': ['query-category', 'user-persona', 'preference', 'query-complexity', 'time-period', 'culture', 'emotion', 'language'],
        # re-do after GPT4 & GPT3.5 update
        # 'mit-movie': ['query-category', 'demographic', 'culture', 'emotion', 'language'],
        'mit-movie': ['query-category', 'demographic', 'emotion', 'language'],  # no good culture options generated

        # `location` dropped for conflict w/ entity type
        'mit-restaurant': ['meal-category', 'demographic', 'ambiance', 'price', 'dietary', 'special', 'service'],

        # 'job-stack': ['job-category', 'language', 'experience-level', 'location', 'culture', 'tone'],
        # drop `culture` for too complex categories
        # 'job-stack': ['job-category', 'language', 'experience-level', 'location', 'tone'],
        # drop `location` for not many categories
        'job-stack': ['job-category', 'language', 'experience-level', 'tone'],

        'wiki-gold': ['topic', 'language'],
    }
    dataset_name2dependent_x_attributes = {
        dnm: [k for k, v in d.items() if (v.get('seed_category') is not None and k != ENTITY_KEY_SEEDED)]
        for dnm, d in dataset_name2attribute2categories.items()
    }
    # duplicate these values for `conll2003-no-misc`, `wiki-gold-no-misc`
    for d in [dataset_name2attribute2categories, dataset_name2default_x_attributes, dataset_name2dependent_x_attributes]:
        d['conll2003-no-misc'] = d['conll2003'].copy()
        d['wiki-gold-no-misc'] = d['wiki-gold'].copy()

    def __init__(
            self, dataset_name: str = 'conll2003', d: Dict[str, Dict[str, Any]] = None, presets: Dict[str, Any] = None,
            diverse_context: bool = None, diverse_entity: Union[bool, str] = None, attributes: List[str] = None,
            diverse_entity_seed_ratio: float = None
    ):
        # static default attributes
        self.dataset_name = dataset_name
        self.default_x_attributes = Attribute2Categories.dataset_name2default_x_attributes[dataset_name]

        self.d = d or copy.deepcopy(Attribute2Categories.dataset_name2attribute2categories[dataset_name])
        self.presets = presets
        if presets:
            sk2k = self.short_key2key
            for attr, preset in presets.items():
                d = self.d.get(attr, self.d.get(sk2k.get(attr)))
                d['categories'] = preset
        logger.info(f'{pl.i(self.__class__.__name__)} initialized w/ presets {pl.i(presets)}')
        self.dependent_attributes = [k for k, v in self.d.items() if v.get('seed_category') is not None]

        # dynamic, inferred attributes
        diverse_context = diverse_context if diverse_context is not None else True
        diverse_entity = diverse_entity if diverse_entity is not None else False
        if not diverse_context and not diverse_entity:
            raise ValueError(f'At least one of {pl.i("diverse_context")}, {pl.i("diverse_entity")} must be True')
        self.diverse_entity_attr_name, self.diverse_entity_seed_attr_name, self.diverse_entity_seed_ratio = None, None, None
        if diverse_entity:
            if isinstance(diverse_entity, str):
                ca(diverse_entity=diverse_entity)
            else:
                assert isinstance(diverse_entity, bool)

            if diverse_context:  # to diversify both context and entity, entity must be seeded
                assert diverse_entity in [True, 'seeded']
                diverse_entity = 'seeded'
            if diverse_entity == 'independent' or diverse_entity is True:
                self.diverse_entity_attr_name = ENTITY_KEY
            elif diverse_entity == 'seeded':
                self.diverse_entity_attr_name = ENTITY_KEY_SEEDED
            else:
                assert diverse_entity == 'mixed'
                self.diverse_entity_attr_name = ENTITY_KEY_MIXED
                self.diverse_entity_seed_ratio = diverse_entity_seed_ratio or 0.5
            if diverse_entity == 'mixed':
                self.diverse_entity_seed_attr_name = tuple(
                    get(self.d, f'{anm}.seed_category') for anm in [ENTITY_KEY, ENTITY_KEY_SEEDED])
            else:
                self.diverse_entity_seed_attr_name = self.d[self.diverse_entity_attr_name].get('seed_category')
        self.diverse_context, self.diverse_entity = diverse_context, diverse_entity

        self.allowed_attributes = attributes or self.implied_attributes()

        # Sanity check preset dependency (e.g., `sub-category` depends on `news-category`)
        for attr in self.allowed_attributes:
            if attr in [ENTITY_KEY, ENTITY_KEY_SEEDED, ENTITY_KEY_MIXED]:
                continue
            d = self.d[attr]
            preset, seed_cat = d['categories'], d.get('seed_category')
            if seed_cat is not None:
                preset_expect, preset_got = d['presets_dependency'][preset], self.d[seed_cat]['categories']
                if preset_expect != preset_got:
                    d_log = dict(
                        attribute=attr, seed_category=seed_cat, preset=preset, seed_preset_expect=preset_expect, seed_preset_got=preset_got)
                    raise ValueError(f'Preset dependency mismatch w/ {pl.i(d_log)}')

        self.cache_attr2options = dict()

    @property
    def attr_name2pretty_attr_name(self) -> Dict[str, str]:
        # for diverse context
        return {attr: self.d[attr]['name'] for attr in self.allowed_attributes if attr != self.diverse_entity_attr_name}

    def to_json(self, save_path: str = None, save_fnm: str = None, meta: Dict[str, Any] = None) -> Dict[str, Any]:
        meta = dict(
            dataset_name=self.dataset_name,
            diverse_context=self.diverse_context, diverse_entity=self.diverse_entity, attributes=self.allowed_attributes,
            **(meta or dict())
        )
        attr2vals = dict()
        for attr in self.allowed_attributes:
            attr2vals[attr] = self.__call__(attribute_name=attr)
        ret = dict(meta=meta, attribute2values=attr2vals)

        if save_path or save_fnm:
            save_path: str = save_path or os_join(pu.generated_data_path, DIVERSE_DNM)
            os.makedirs(save_path, exist_ok=True)
            save_fnm = f'{save_fnm}.json' or f'{now(for_path=True, fmt="short-full")}_Attr2Cats-Config.json'
            with open(os_join(save_path, save_fnm), 'w') as f:
                json.dump(ret, f, indent=4)
        return ret

    @classmethod
    def from_json(
            cls, dataset_name: str = 'conll2003', diverse_context: bool = None, diverse_entity: Union[bool, str] = None,
            diverse_x_config: str = None, diverse_y_config: str = None, diverse_y_latent_attr_dim: str = None, **kwargs
    ) -> 'Attribute2Categories':
        assert diverse_context or diverse_entity  # sanity check
        d = dict()

        def get_single(config: str = None):
            if os.path.exists(config):
                with open(config, 'r') as f:
                    return json.load(f)['attribute2values']
            else:
                return json.loads(config)  # must be a json string
        # if diverse_context:
        if diverse_x_config:
            d_ = get_single(config=diverse_x_config)
            d_ = {k: dict(categories=v) for k, v in d_.items()}  # fit to the internal `d` API
            d.update(d_)
        # if diverse_entity:
        if diverse_y_config:
            d_ = get_single(config=diverse_y_config)
            assert len(d_) == 1  # sanity check
            d_ = {k: dict(categories=v, seed_category=diverse_y_latent_attr_dim) for k, v in d_.items()}
            d.update(d_)
        d.update(kwargs)
        return cls(dataset_name=dataset_name, d=d, diverse_context=diverse_context, diverse_entity=diverse_entity)

    def __getitem__(self, key: str):
        return get(dic=self.d, ks=key)

    def implied_attributes(self) -> List[str]:
        ret = []
        if self.diverse_context:
            ret += Attribute2Categories.dataset_name2default_x_attributes[self.dataset_name]
        if self.diverse_entity:
            ret.append(self.diverse_entity_attr_name)

        if self.presets is not None:
            ks = self.presets.keys()
            sk2k = self.short_key2key
            ret_ = [sk2k.get(k, k) for k in ks]
            ret += [k for k in ret_ if k not in ret]
        return ret

    def get_dependency_map(self, attributes: List[str] = None) -> str:
        k2sk = self.key2short_key

        dep = dict()
        attributes = attributes or []
        for attr in attributes:
            if attr in k2sk and attr != ENTITY_KEY_MIXED:
                seed_cat = self.d[attr].get('seed_category')
                if seed_cat is not None:  # store the dependent preset value
                    cat = self.d[seed_cat]['categories']
                    if not isinstance(cat, str):
                        assert isinstance(cat, list)
                        pref = seed_cat.split('-')[0]
                        cat = f'{pref}-{len(cat)}'
                    dep[k2sk[attr]] = cat
        if dep:
            return pl.i(dep, key_value_sep='<=', pairs_sep=',', with_color=False)

    def get_attr2file_dir(self, attributes: List[str] = None) -> Dict[str, str]:
        ret = dict()
        for attr in attributes:
            if attr != ENTITY_KEY_MIXED:
                d = self.d[attr]
                if d.get('from_file', None):
                    cat = d['categories']
                    dir_nm = d['presets'][cat]
                    ret[attr] = dir_nm
        return ret

    def get_preset_categories(self, attributes: List[str] = None) -> Dict[str, str]:
        return {attr: d['categories'] for attr, d in self.d.items() if attr in attributes and isinstance(d['categories'], str)}

    @property
    def short_key2key(self) -> Dict[str, str]:
        return {v['short']: k for k, v in self.d.items() if 'short' in v}

    @property
    def key2short_key(self) -> Dict[str, str]:
        return {k: v['short'] for k, v in self.d.items() if 'short' in v}

    def __call__(self, attribute_name: str, seed_attribute_name: str = None) -> CategoryOptions:
        if attribute_name == ENTITY_KEY_MIXED:
            attribute_name = ENTITY_KEY_SEEDED if random.random() < self.diverse_entity_seed_ratio else ENTITY_KEY
        options = self.get_options(attribute_name=attribute_name)
        if seed_attribute_name is not None:
            assert get(self.d, f'{attribute_name}.seed_category')
            return options[seed_attribute_name]
        else:
            return options

    def _dir_name2entity_options(self, dir_name: str = None) -> Dict[str, List[str]]:
        path = dataset_name2data_dir(dataset_name=self.dataset_name, sub_dir=DIVERSE_ENTITY_DNM, input_dir=dir_name).path
        with open(os_join(path, 'processed-entities.json'), 'r') as f:
            return json.load(f)['entity-type2entity-names']

    def get_options(self, attribute_name: str = None) -> CategoryOptions:
        if attribute_name in self.cache_attr2options:
            return self.cache_attr2options[attribute_name]

        if attribute_name in [ENTITY_KEY, ENTITY_KEY_SEEDED]:  # named entities
            # always load from file
            d = self.d[attribute_name]  # see `attr2cats`
            cat, presets = d['categories'], d.get('presets')
            if isinstance(cat, str):  # a pre-defined dataset
                assert presets
                dir_nm = presets[cat]
                if isinstance(dir_nm, str):
                    options = self._dir_name2entity_options(dir_name=dir_nm)
                else:
                    assert isinstance(dir_nm, dict)
                    options = dict()
                    for dir_nm_ in dir_nm.values():
                        options.update(self._dir_name2entity_options(dir_name=dir_nm_))
            else:
                assert isinstance(cat, dict)  # already a loaded entity pool
                options = cat
        else:
            d = self.d[attribute_name]
            options, from_file = d['categories'], d.get('from_file')
            if isinstance(options, str):
                options = d['presets'][options]
            if from_file:
                dnm_ = self.dataset_name
                if self.dataset_name.endswith('-no-misc'):  # diverse context were from original dataset directory
                    dnm_ = self.dataset_name[:-len('-no-misc')]
                base_path = os_join(dataset_name2data_dir(dataset_name=dnm_, sub_dir=DIVERSE_CONTEXT_DNM).path, options)
                with open(os_join(base_path, 'processed-categories.json'), 'r') as f:
                    options = json.load(f)['attribute2categories'][attribute_name]
        self.cache_attr2options[attribute_name] = options
        return options


if __name__ == '__main__':
    dnm = 'conll2003-no-misc'
    # dnm = 'wiki-gold-no-misc'
    # dnm = 'mit-movie'
    # dnm = 'mit-restaurant'
    # dnm = 'job-stack'

    def check_x_from_file():
        pst = None
        if dnm == 'conll2003-no-misc':
            pst = dict(cat='30', sub='30=>10')
        a2c = Attribute2Categories(dataset_name=dnm, presets=pst, diverse_context=True)

        if dnm == 'conll2003-no-misc':
            if pst is None:
                sic(a2c(attribute_name='news-category'))
            else:
                sic(a2c(attribute_name='sub-category', seed_attribute_name='Sports'))
        elif dnm == 'mit-restaurants':
            sic(a2c(attribute_name='meal-category'))
    # check_x_from_file()

    def check_y():
        pst = None
        if dnm == 'conll2003-no-misc':
            # pst = None
            # pst = {'ent-s': '30=>50'}
            # pst = {'ent-s': '30=>50-split'}
            pst = {'ent-s': '30=>30'}
        # de = True
        de = 'mixed'
        a2c = Attribute2Categories(dataset_name=dnm, presets=pst, diverse_context=False, diverse_entity=de)
        sic(a2c.allowed_attributes)

        # anm = ENTITY_KEY
        # anm = ENTITY_KEY_SEEDED
        anm = ENTITY_KEY_MIXED
        options = a2c(attribute_name=anm)
        sic(options)
    check_y()
