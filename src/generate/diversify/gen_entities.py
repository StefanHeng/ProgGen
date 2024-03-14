"""
For generating diverse entities (Y)

SimPrompt: Generate list of entities given entity type
    Variation: Given a sub-entity type, e.g. location => street name
AttrPrompt: Generate list of entities given entity type and sentence category
"""

import re
import json
import random
from copy import deepcopy
from os.path import join as os_join
from typing import List, Union, Optional, Iterable

from stefutil import *
from src.util import *
from src.util.ner_example import *
from src.data_util import *
from src.generate.diversify.util import *
from src.generate.diversify import Attribute2Categories


__all__ = ['EntityGenerator']


_logger = get_logger('Entity Gen')


KEY_DEFAULT_N_LIST = '__default_#list__'


dataset_name2entity_type2n_list = {
    # 'conll2003-no-misc': {
    #     'person': 50,
    #     'location': 120,  # if list less locations, many duplicate locations across prompts
    #     # 'location': 150,
    #     # 'organization': 50
    #     'organization': 80
    # },
    'conll2003-no-misc': 50,
    # 'conll2003-no-misc': 75,

    # 'mit-movie': 45,  # 45 for each entity type
    'mit-movie': {
        # 'Title': 60,
        # 'MPAA Rating': 25,

        'Title': 75,
        'Director': 75,
        'Actor': 75,
        'Character': 75,

        'MPAA Rating': 15,
        # 'MPAA Rating': 10,
        # 'MPAA Rating': 8,
        'Plot': 30,
        'Trailer': 15,
        'Song': 30,
        KEY_DEFAULT_N_LIST: 45
    },
    'mit-restaurant': 45,
    # 'job-stack': 50,
    'job-stack': 45,
}
dataset_name2seeded_entity_type2n_list = {
    # if seeded by news category, don't need too many entities
    # 'conll2003-no-misc': 20,
    'conll2003-no-misc': 15,

    # 'mit-movie': 15,
    # 'mit-movie': 20,
    # 'mit-movie': 25,

    'mit-movie': {
        'Title': 45, 'Director': 45, 'Actor': 45, 'Character': 45,

        'MPAA Rating': 8,
        'Plot': 25, 'Trailer': 10, 'Song': 25,
        KEY_DEFAULT_N_LIST: 35
    },

    'mit-restaurant': 15,
    'job-stack': 20,
    # try an even smaller set
    'wiki-gold-no-misc': 8
}
# use the same entity sizes for `wiki-gold-no-misc` as `conll2003-no-misc`
dataset_name2entity_type2n_list['wiki-gold-no-misc'] = dataset_name2entity_type2n_list['conll2003-no-misc']
# dataset_name2seeded_entity_type2n_list['wiki-gold-no-misc'] = dataset_name2seeded_entity_type2n_list['conll2003-no-misc'].copy()


# Additional information that clarify the defn. for each entity type
dataset_name2entity_type_info = {
    'mit-movie': {
        "Viewers' Rating": {
            'element-name': 'term', 'element-name-pl': 'terms',
            'explanations': [
                dict(definition='is a rating score', example='4 out of 5 stars'),
                dict(definition='is a brief recommendation/popularity level', example='must see')
            ]
        },
        'Year': {
            'element-name': 'term', 'element-name-pl': 'terms',
            'explanations': [dict(definition='is a year or a year range')],
            # 'explanations': [dict(
            #     definition='is a year or a year range',
            #     # example=['2004', '90s'],
            #     example=['2004', '90s', 'last year']
            # )]
            # 'explanations': [
            #     dict(definition='is a year'),
            #     dict(definition='is a year range', example='90s'),
            # ]
            # 'explanations': [dict(definition='is a year or a range of years')]
        },
        'Plot': {'explanations': [dict(definition='is a movie theme or plot element', example='bounty hunter', example_prob=0.5)]},
        'Trailer': {
            'element-name': 'term', 'element-name-pl': 'terms',
            'explanations': [dict(definition='indicates a segment or a clip from a movie')],
        },
        'Review': {
            'element-name': 'term', 'element-name-pl': 'terms',
            'explanations': [dict(definition='is a detailed movie comment', example='funniest of all time')]
        }
    }
}


pattern_comma = re.compile(r'^(?P<entity>.*),\s(?P<comma>.*)$')
pattern_paren = re.compile(r'^(?P<entity>.*)\s\((?P<paren>.*)\)$')


class EntityGenerator(OptionGenerator):
    def __init__(
            self, entity_types: Union[str, List[str]] = None, seeded: bool = False, a2c: Attribute2Categories = None,
            n_demo: int = None, include_demo_ratio: float = None,
            drop_after_comma: bool = None, split_paren: bool = None,
            **kwargs
    ):
        """
        :param entity_types: If given, only generate entities of these types
        :param seeded: If true, generate entities based on sentence category
            e.g. For CoNLL-2003, use `news-category` options
        :param a2c: If given, overrides `Attribute2Categories` loading
            Relevant for Diversify Y (seeded) only
        :param n_demo: If given, demo named entities are added to the prompt
        :param include_demo_ratio: Probability of generating entities w/ demo
            Relevant when `n_demo` is given
        :param drop_after_comma: If true, only keep the part before the comma
            Intended for e.g. `Tokyo, Japan`
                For e.g. `Tesla, Inc.`, dropping `Inc` still makes sense
            Intended for entity pair mapping edge case when we split entities by comma,
                and since these are effectively 2 entities anyway
        :param split_paren: If true, split entities with terms in the parenthesis into two entities
            Intended for e.g. `International Monetary Fund (IMF)`
            Intended for entity pair mapping edge case when entity type is in the parenthesis,
                and since the abbreviation is a separate entity anyway
        """
        super().__init__(**kwargs)
        self.n_demo, self.dataset = n_demo, None
        if n_demo:
            self.dataset = DatasetLoader(dataset_name=self.dataset_name, data_format='readable')
            self.include_demo_ratio = include_demo_ratio or 1.0
        else:
            self.include_demo_ratio = None
        self.default_entity_types = sconfig(f'datasets.{self.dataset_name}.readable-entity-types')
        if entity_types:
            if isinstance(entity_types, str):
                entity_types = [entity_types]
            assert all(et in self.default_entity_types for et in entity_types)
            self.entity_types = entity_types
        else:
            self.entity_types = self.default_entity_types.copy()
        d_et_info = dataset_name2entity_type_info.get(self.dataset_name)
        self.entity_type_info = d_et_info.copy() if d_et_info is not None else None

        self.sample_type = 'Entity'
        self.dir_args = dict(dataset_name=self.dataset_name, sub_dir=DIVERSE_ENTITY_DNM)

        # e.g. `8. Berlin`
        # e.g. `2. Javier Rodriguez`
        # e.g. `Paris, France (known for its culinary heritage and renowned restaurants)`
        #    Drop parenthesis if it's not all caps
        # e.g. `Abbey Road Studios, London - An iconic recording studio in London famous for its association with The Beatles.`
        #   Drop after dash
        # e.g. `Memphis, Tennessee: Known for its influence on rock 'n' roll and home to famous musicians such as Elvis Presley and B.B. King.`
        #  Drop after colon
        # e.g. `25. "Single Ladies (Put a Ring on It)" - Beyoncé`
        #  Drop song artist info
        #   e.g. `14. "A Whole New World" from Aladdin`, `15. "Gangnam Style" by PSY`
        # Drop the prefix
        #   e.g. `1. The Making of Jaws`, `2. Behind the Scenes of The Godfather`, `1. 1994 - The year when "The Lion King" was released.`
        # drop starting and trailing quotes
        # Edge case: entity in next line, e.g.
        #   `1.
        #   The American Heart Association
        #   2.
        #   Macy's`
        pattern_enum_prefix = rf'(?P<idx>\d+)\.( )?)'
        pat_d = rf'[--\u2013]'
        self.pattern_entity = [
            # contains both dash and paren
            # dash inside paren, e.g.
            #   `10. PG-14 (Parents strongly cautioned – Some material may be inappropriate for children under 14)`
            re.compile(rf'({pattern_enum_prefix}(?P<entity>.*) \(.+ {pat_d} .+\)\n'),

            # dash first
            re.compile(rf'({pattern_enum_prefix}"(?P<entity>.*)" {pat_d} (?P<dash>.*?) \((?![A-Z]+\)).*\n'),
            re.compile(rf'({pattern_enum_prefix}(?P<entity>.*) {pat_d} (?P<dash>.*?) \((?![A-Z]+\)).*\n'),
            # paren first
            re.compile(rf'({pattern_enum_prefix}"(?P<entity>.*) \(.+\)" {pat_d} (?P<dash>.*)\n'),
            re.compile(rf'({pattern_enum_prefix}(?P<entity>.*) \(.+\) {pat_d} (?P<dash>.*)\n'),

            re.compile(rf'({pattern_enum_prefix}"(?P<entity>.+?)" {pat_d} (?P<dash>.*)\n'),  # match the smallest span before the dash
            re.compile(rf'({pattern_enum_prefix}(?P<entity>.+?) {pat_d} (?P<dash>.*)\n'),

            re.compile(rf'({pattern_enum_prefix}"(?P<entity>.*) \((?![A-Z]+)\)".*\n'),
            re.compile(rf'({pattern_enum_prefix}(?P<entity>.*) \((?![A-Z]+)\).*\n'), re.compile(r'(?P<idx>\d+)\. "(?P<entity>.*) \(.+\)"\n'),
            re.compile(rf'({pattern_enum_prefix}(?P<entity>.*) \(.+\)\n'),

            re.compile(rf'({pattern_enum_prefix}(?P<entity>.*): (?P<dash>.*)\n'), re.compile(r'(?P<idx>\d+)\. "(?P<entity>.*)"\n'),

            re.compile(r'(?P<idx>\d+)\. \n(?P<entity>.{1,30})\n'),  # since so mal-formed cos newline, enforce max length check
            re.compile(rf'({pattern_enum_prefix}(?P<entity>[^\n]*)\n'),

            # e.g. `- American`
            re.compile(rf'- (?P<entity>.*)\n'),

            # re.compile(r'(?P<idx>\d+)\. (?P<entity>.*) [--\u2013] the year when .*\n', re.IGNORECASE),
            # re.compile(r'(?P<idx>\d+)\. "(?P<entity>.*)" (from|by) (?P<artist>.*)', re.IGNORECASE),
            # re.compile(r'(?P<idx>\d+)\. "(the making of|behind the scenes of) (?P<entity>.+)"', re.IGNORECASE),
            # re.compile(r'(?P<idx>\d+)\. (the making of|behind the scenes of) (?P<entity>.+)', re.IGNORECASE),
            #
            # # contains both dash and paren
            # # dash first
            # re.compile(r'(?P<idx>\d+)\. "(?P<entity>.*)" [-\u2013] (?P<dash>.*?) \((?![A-Z]+\)).*\n'),
            # re.compile(r'(?P<idx>\d+)\. (?P<entity>.*) [-\u2013] (?P<dash>.*?) \((?![A-Z]+\)).*\n'),
            # # paren first
            # re.compile(r'(?P<idx>\d+)\. "(?P<entity>.*) \(.+\)" [-\u2013] (?P<dash>.*)\n'),
            # re.compile(r'(?P<idx>\d+)\. (?P<entity>.*) \(.+\) [-\u2013] (?P<dash>.*)\n'),
            #
            # re.compile(r'(?P<idx>\d+)\. "(?P<entity>.*)" [-\u2013] (?P<dash>.*)\n'),
            # re.compile(r'(?P<idx>\d+)\. (?P<entity>.*) [-\u2013] (?P<dash>.*)\n'),
            #
            # re.compile(r'(?P<idx>\d+)\. "(?P<entity>.*) \((?![A-Z]+)\)".*\n'),
            # re.compile(r'(?P<idx>\d+)\. (?P<entity>.*) \((?![A-Z]+)\).*\n'),
            # re.compile(r'(?P<idx>\d+)\. "(?P<entity>.*) \(.+\)"\n'),
            # re.compile(r'(?P<idx>\d+)\. (?P<entity>.*) \(.+\)\n'),
            #
            # re.compile(r'(?P<idx>\d+)\. (?P<entity>.*): (?P<dash>.*)\n'),
            # re.compile(r'(?P<idx>\d+)\. "(?P<entity>.*)"\n'),
            # re.compile(r'(?P<idx>\d+)\. (?P<entity>.*)\n'),
            #
            # # e.g. `- American`
            # re.compile(r'- (?P<entity>.*)\n'),

        ]
        if self.dataset_name == 'mit-movie':
            pat_et = [
                re.compile(rf'({pattern_enum_prefix}(?P<entity>.*) {pat_d} the year when .*\n', re.IGNORECASE),
                re.compile(rf'({pattern_enum_prefix}"(?P<entity>.*)" (from|by) (?P<artist>.*)', re.IGNORECASE),
                re.compile(rf'({pattern_enum_prefix}"(the making of|behind the scenes of) (?P<entity>.+)"', re.IGNORECASE),
                re.compile(rf'({pattern_enum_prefix}(the making of|behind the scenes of) (?P<entity>.+)', re.IGNORECASE),
            ]
            self.pattern_entity = pat_et + self.pattern_entity

        self.seeded, self.seed_attr_name, self.a2c, self.seed_options = seeded, None, None, None
        self.drop_after_comma = drop_after_comma or False
        self.split_paren = split_paren or False
        if seeded:
            self.a2c = a2c or Attribute2Categories(dataset_name=self.dataset_name, diverse_context=False, diverse_entity='seeded')
            self.seed_attr_name = self.a2c.diverse_entity_seed_attr_name
            self.seed_options = self.a2c(attribute_name=self.seed_attr_name)

        # use parent API for iterating setups
        d = dataset_name2seeded_entity_type2n_list if self.seeded else dataset_name2entity_type2n_list
        d_n_list = d[self.dataset_name]
        if isinstance(d_n_list, int):
            d_n_list = {et: d_n_list for et in self.entity_types}
        else:
            assert isinstance(d_n_list, dict)
            if not all(et in self.entity_types for et in d_n_list.keys()):
                assert KEY_DEFAULT_N_LIST in d_n_list  # a default value for all entity types must be given
                default = d_n_list[KEY_DEFAULT_N_LIST]
                d_n_list = {et: d_n_list.get(et, default) for et in self.entity_types}
            d_n_list = d_n_list.copy()
        self.keyword2n_list = d_n_list
        self.keywords = self.entity_types

        self.logger = _logger

    def get_prompt(self, n_list: int = None, entity_type: str = None, seed_category: str = None, generator: Union[random.Random, int] = None) -> str:
        # note: by default, generate entities zero-shot
        #   from prior study on CoNLL-03, showing demo seems to limit diversity
        assert entity_type is not None and entity_type in self.entity_types
        if self.seeded:
            assert seed_category is not None
        gen = get_random_generator(generator=generator)

        enms = []
        if self.n_demo and gen.random() < self.include_demo_ratio:
            samples: List[NerReadableExample] = self.dataset.get_few_demo_samples(n_demo=self.n_demo, demo_type='n-shot', shuffle=True)
            enms = [nm for eg in samples for nm, tp in zip(eg.entity_names, eg.entity_types) if tp == entity_type]
            if len(enms) > 1:
                gen.shuffle(enms)

        n_list = n_list or dataset_name2entity_type2n_list[self.dataset_name][entity_type]
        if self.dataset_name == 'conll2003-no-misc':
            if self.seeded:
                # return (f'Suppose you are a news writer. '
                #         f'Please generate {n_list} diverse named entities in news about {seed_category} '
                #         f'that can be categorized as {entity_type}.')
                # ret = (f'Suppose you are a news writer for {seed_category}. '
                #        f'Please generate {n_list} diverse named entities that can be categorized as {entity_type}.')

                # mention the domain twice
                ret = (f'Suppose you are a news writer for {seed_category}. '
                       f'Please generate {n_list} diverse named entities in news articles that can be categorized as {entity_type}.')
            else:
                # ret = (f'Suppose you are a news writer. '
                #        f'Please generate {n_list} diverse named entities that can be categorized as {entity_type}.')

                # mention the domain twice
                ret = (f'Suppose you are a news writer. '
                       f'Please generate {n_list} diverse named entities in news articles that can be categorized as {entity_type}.')
        elif self.dataset_name == 'mit-movie':
            # if self.seeded:
            #     # ret = (f'Suppose you are a user of a dialog system or conversational agent for {seed_category} movie queries. '
            #     #        f'Please generate {n_list} diverse named entities that can be categorized as {entity_type}.')
            #     # ret = (f'Suppose you are a user of a dialog system or conversational agent for movie queries about {seed_category}. '
            #     #        f'Please generate {n_list} diverse named entities that can be categorized as {entity_type}.')
            #
            #     # mention the domain twice; sample [`named entities`. `keywords`]
            #     kind = gen.choice(['named entities', 'keywords'])
            #     ret = (f'Suppose you are a user of a dialog system or conversational agent for movie queries about {seed_category}. '
            #            f'Please generate {n_list} diverse {kind} in movie queries that can be categorized as {entity_type}.')
            # else:

            # ret = (f'Suppose you are a user of a dialog system or conversational agent about movies. '
            #        f'Please generate {n_list} diverse named entities that can be categorized as {entity_type}.')

            # mention the domain twice
            # ret = (f'Suppose you are a user of a dialog system or conversational agent about movies. '
            #        f'Please generate {n_list} diverse named entities in movie queries that can be categorized as {entity_type}.')
            # swap `named entities` to `keywords`
            # ret = (f'Suppose you are a user of a dialog system or conversational agent about movies. '
            #        f'Please generate {n_list} diverse keywords in movie queries that can be categorized as {entity_type}.')

            # sample [`named entities`. `keywords`]
            # kind = random.choice(['named entities', 'keywords'])
            # ret = (f'Suppose you are a user of a dialog system or conversational agent about movies. '
            #        f'Please generate {n_list} diverse {kind} in movie queries that can be categorized as {entity_type}.')
            # mention domain once
            # ret = (f'Suppose you are a user of a dialog system or conversational agent about movies. '
            #        f'Please generate {n_list} diverse {kind} that can be categorized as {entity_type}.')

            # after manual prompt search for each entity type, e.g.
            #   `Suppose you are a user of a dialog system or conversational agent about movies.
            #   Please generate 45 diverse terms in movie queries that can be categorized as Review.
            #   A Review term is a detailed movie comment, e.g. "funniest of all time".`
            pref = 'Suppose you are a user of a movie dialog system.'

            enm, enm_pl = 'named entity', 'named entities'
            d = self.entity_type_info.get(entity_type)
            if d is not None:
                enm = d.get('element-name', enm)
                enm_pl = d.get('element-name-pl', enm_pl)
            # ret = f'{ret} Please generate {n_list} diverse {enm_pl} that can be categorized as {entity_type}.'
            instr = f'Please generate {n_list} diverse {enm_pl} that can be categorized as {entity_type}'
            if self.seeded:
                instr = f'{instr} in movie queries about {seed_category}'
            ret = f'{pref} {instr}.'

            if d is not None and 'explanations' in d:
                expls = d['explanations']
                if len(expls) > 1:
                    expl = gen.choice(expls)
                else:
                    expl = expls[0]
                defn, eg, eg_prob = expl['definition'], expl.get('example'), expl.get('example_prob')
                assert defn is not None
                if eg is not None and eg_prob is not None and gen.random() > eg_prob:  # keep the example at the probability
                    eg = None

                expl_str = f'A {entity_type} {enm} {defn}'
                if eg is not None:
                    if isinstance(eg, list):
                        et = deepcopy(eg)
                        gen.shuffle(et)
                        et = [f'"{e}"' for e in eg]
                        eg = pl.nc(et)
                    else:
                        assert isinstance(eg, str)
                        eg = f'"{eg}"'
                    expl_str = f'{expl_str}, e.g. {eg}.'
                else:
                    expl_str = f'{expl_str}.'
                ret = f'{ret}\n{expl_str}'
            return ret
        elif self.dataset_name == 'mit-restaurant':
            if self.seeded:
                # ret = (f'Suppose you are a user of a dialog system or conversational agent for restaurant queries about {seed_category}. '
                #        f'Please generate {n_list} diverse named entities that can be categorized as {entity_type}.')

                # mention the domain twice; sample [`named entities`. `keywords`]
                kind = gen.choice(['named entities', 'keywords'])
                ret = (f'Suppose you are a user of a dialog system or conversational agent for restaurant queries about {seed_category}. '
                       f'Please generate {n_list} diverse {kind} in restaurant queries that can be categorized as {entity_type}.')
            else:
                # ret = (f'Suppose you are a user of a dialog system or conversational agent about restaurants. '
                #        f'Please generate {n_list} diverse named entities that can be categorized as {entity_type}.')

                # mention the domain twice; sample [`named entities`. `keywords`]
                kind = gen.choice(['named entities', 'keywords'])
                ret = (f'Suppose you are a user of a dialog system or conversational agent about restaurants. '
                       f'Please generate {n_list} diverse {kind} in restaurant queries that can be categorized as {entity_type}.')
        elif self.dataset_name == 'job-stack':
            if self.seeded:
                # ret = (f'Suppose you are a recruiter creating job postings on StackOverflow for {seed_category} positions. '
                #        f'Please generate {n_list} diverse named entities that can be categorized as {entity_type}.')

                # mention the domain twice
                ret = (f'Suppose you are a recruiter creating job postings on StackOverflow for {seed_category} positions. '
                       f'Please generate {n_list} diverse named entities in job postings that can be categorized as {entity_type}.')
            else:
                # ret = (f'Suppose you are a recruiter creating job postings on StackOverflow. '
                #        f'Please generate {n_list} diverse named entities that can be categorized as {entity_type}.')

                # mention the domain twice
                ret = (f'Suppose you are a recruiter creating job postings on StackOverflow. '
                       f'Please generate {n_list} diverse named entities in job postings that can be categorized as {entity_type}.')

                # sample [`named entities`. `keywords`]
                # kind = gen.choice(['named entities', 'keywords'])
                # ret = (f'Suppose you are a recruiter creating job postings on StackOverflow. '
                #        f'Please generate {n_list} diverse {kind} in job postings that can be categorized as {entity_type}.')
        elif self.dataset_name == 'wiki-gold-no-misc':
            if self.seeded:
                ret = (f'Suppose you are a Wikipedia editor. '
                       f'Please generate {n_list} diverse named entities in Wikipedia articles about {seed_category} '
                       f'that can be categorized as {entity_type}.')
            else:
                ret = (f'Suppose you are a Wikipedia editor. '
                       f'Please generate {n_list} diverse named entities in Wikipedia articles that can be categorized as {entity_type}.')
        else:
            raise NotImplementedError
        if enms:
            if len(enms) == 1:
                ret = f'{ret} For example, {enms[0]}.'
            else:
                raise NotImplementedError
                # ret = f'{ret}. Examples: {pl.nc(enms)}.'
        return ret

    def iter_setups(self, n_list: N_List = None) -> Iterable[GenSetup]:
        knm = 'entity_type'
        if 'wiki-gold' in self.dataset_name:
            mult = 40  # raise max_tokens for wiki-gold since it may generate descriptions
        elif self.dataset_name == 'mit-movie':
            mult = 30
        else:
            mult = None
        if self.seeded:
            return self._iter_setups(n_list=n_list, seeded_keywords=self.seed_options, key_name=knm, max_tokens_multiplier=mult)
        else:
            return self._iter_setups(n_list=n_list, key_name=knm, max_tokens_multiplier=mult)

    def write_completions(self, n_prompt: int = 5, n_list: N_List = None, output_dir_nm: str = None, **kwargs):
        return self._write_completions(n_prompt=n_prompt, n_list=n_list, output_dir_nm=output_dir_nm, **kwargs)

    @property
    def meta(self) -> Optional[str]:
        ret = dict()
        if self.entity_types != self.default_entity_types:
            ret['ets'] = self.entity_types
        if self.seeded:
            ret['seed'] = self.seed_attr_name.split('-')[0]
        if self.n_demo:
            ret['#demo'] = self.n_demo
            if self.include_demo_ratio:
                ret['demo%'] = self.include_demo_ratio
        if self.drop_after_comma:
            ret['drop-[,]'] = True
        if self.split_paren:
            ret['split-()'] = True
        return pl.pa(ret) if ret != dict() else None

    def process_completions(
            self, completions_dir_name: str = None, expected_samples_per_completion: N_List = None, output_dir_name: str = None
    ):
        drop_dup = None
        # if not self.seeded and 'conll2003' in self.dataset_name:  # keep all values to ensure a fair distribution
        #     drop_dup = dict(location=False, organization=False)

        out = self._process_completions(
            completions_dir_name=completions_dir_name, expected_samples_per_completion=expected_samples_per_completion,
            output_dir_name=output_dir_name, drop_duplicates=drop_dup
        )
        if self.ec and self.ec.have_edge_case:
            self.logger.info(f'Extraction Edge cases: {pl.i(self.ec.summary())}')

        output_path, et2enms, d_sz = out.output_path, out.key2value, out.key2size
        self.logger.info(f'Entity sizes: {pl.i(d_sz, indent=1)}')

        meta = {
            'dataset-name': self.dataset_name, 'entity-types': self.entity_types,
            'seeded': self.seeded, 'seed-attribute-name': self.seed_attr_name, 'seed-options': self.seed_options,
            'entity-sizes': d_sz
        }
        out = {'entity-type2entity-names': et2enms, 'meta': meta}
        with open(os_join(output_path, 'processed-entities.json'), 'w') as f:
            json.dump(out, f, indent=4)
        self.logger.info(f'Processed entities written to {pl.i(stem(output_path))}')
        return et2enms

    def extract_values(self, text: str = None, n_expect: int = None) -> List[str]:
        ms = self._match_values(text=text, pattern=self.pattern_entity, n_expect=n_expect, union_patterns=True)
        ret = [m.group('entity').strip() for m in ms]
        ret = [edit.drop_enclosing_quotes(e) for e in ret]

        if self.drop_after_comma:
            ms = [re.match(pattern_comma, e) for e in ret]
            if any(m is not None for m in ms):
                ret_ = []
                et2et = dict()
                for et, m in zip(ret, ms):
                    if m is None:
                        ret_.append(et)
                    else:
                        e1, e2 = m.group('entity').strip(), m.group('comma').strip()
                        ret_.append(e1)  # drop the part after comma
                        et2et[et] = e1
                self.logger.info(f'Dropping {pl.i(len(et2et))} entities with comma: {pl.i(et2et)}')
                ret = ret_
        if self.split_paren:
            ms = [re.match(pattern_paren, e) for e in ret]
            if any(m is not None for m in ms):
                ret_ = []
                et2split = dict()
                for et, m in zip(ret, ms):
                    if m is None:
                        ret_.append(et)
                    else:
                        e1, e2 = m.group('entity').strip(), m.group('paren').strip()
                        ret_.extend([e1, e2])
                        et2split[et] = (e1, e2)
                self.logger.info(f'Split {pl.i(len(et2split))} entities with parenthesis: {pl.fmt(et2split)}')
                ret = ret_
        # sanity check no empty string
        if not all(e != '' for e in ret):
            sic(ret)
        assert all(e != '' for e in ret)
        return ret


if __name__ == '__main__':
    dnm = 'conll2003-no-misc'
    # dnm = 'wiki-gold-no-misc'
    # dnm = 'mit-movie'
    # dnm = 'mit-restaurant'
    # dnm = 'job-stack'
    ets = None
    if dnm == 'mit-movie':
        # ets = ['Year']
        ets = None
    elif dnm == 'mit-restaurant':
        # ets = 'Price'
        ets = None
    kw_in_prompt = None
    # kw_in_prompt = dnm in ['mit-movie', 'mit-restaurant']  # `keywords` is sampled in prompt
    # kw_in_prompt = dnm in ['mit-movie', 'mit-restaurant', 'job-stack']
    seed = 42

    n_demo_ = None
    # n_demo_ = 1
    # dr = None
    dr = 0.4
    # sd = False
    sd = True
    pc = EntityGenerator(dataset_name=dnm, entity_types=ets, n_demo=n_demo_, include_demo_ratio=dr, seeded=sd)

    def check_prompt():
        gen = get_random_generator(generator=seed)
        dup = 1
        # dup = 5
        prompts = []
        for s in pc.iter_setups():
            prompts += [pc.get_prompt(**s.get_prompt_args, generator=gen) for _ in range(dup)]
        prettier.print_prompts(prompt=prompts)

    def write_completion():
        if not sd:
            # n_call = 8 if kw_in_prompt else 5
            # n_call = 5
            n_call = 10
            # n_call = 15
        else:
            # n_call = 5 if kw_in_prompt else 3
            n_call = 5
            # n_call = 3
        # md_nm = 'gpt-3.5-turbo'
        md_nm = 'gpt-3.5-turbo-1106'  # better instruction-following, cheaper
        pc.write_completions(n_prompt=n_call, prompt_seed=seed, model_name=md_nm, temperature=1, seed=seed, timeout=30)

    def process():
        if dnm == 'conll2003-no-misc':
            if not sd:
                # dir_nm = '23-11-05_17-27-40_Entity-Res_{#l=50,de=T}_debug'

                # after location #list change
                # dir_nm = '23-11-05_18-58-10_Entity-Res_{#l=50,de=T}_debug'
                # dir_nm = '23-11-05_19-09-32_Entity-Res_{de=T}'

                # after org #list change
                # dir_nm = '23-11-05_19-17-18_Entity-Res_{de=T}'

                # from GPT3.5-1106; smaller entity pool
                # dir_nm = '23-11-19_20-37-28_Entity-Res'

                # mention the domain twice
                # dir_nm = '23-12-28_22-55-34_Entity-Res'

                # re-do w/ larger entity pool
                # dir_nm = '24-02-07_15-45-42_Entity-Res'
                #
                dir_nm = '24-02-07_20-27-21_Entity-Res'
            else:
                # dir_nm = '23-11-09_21-51-13_Entity-Res_seeded'
                # less requested entities per setup
                # dir_nm = '23-11-10_13-06-23_Entity-Res_seeded'
                # less requested entities per setup again
                # dir_nm = '23-11-10_15-38-46_Entity-Res_seeded'

                # from GPT3.5-1106; from updated seed categories
                # dir_nm = '23-11-19_21-03-13_Entity-Res_{seed=news}'

                # mention the domain twice
                # dir_nm = '23-12-28_23-57-40_Entity-Res_{seed=news}'

                dir_nm = '24-03-12_00-19-38_Entity-Res_{seed=news}'  # bug during script refactor
        elif dnm == 'mit-movie':
            if not sd:
                if ets is None:
                    # dir_nm = '23-11-12_17-16-47_Entity-Res'
                    # dir_nm = '23-11-12_17-33-00_Entity-Res'

                    # re-do after GPT4 & GPT3.5 update
                    # dir_nm = '23-11-21_17-49-26_Entity-Res'

                    # mention domain twice in prompt
                    # dir_nm = '23-12-27_14-23-43_Entity-Res'
                    # many generated entities are actually Titles, move them
                    # dir_nm = '23-12-27_14-23-43_Entity-Res_moved'

                    # `named entities` => `keywords` in prompt
                    # dir_nm = '23-12-27_15-18-08_Entity-Res'

                    # sample [`named entities`. `keywords`] in prompt
                    # dir_nm = '23-12-27_16-00-36_Entity-Res'
                    # mention domain once in prompt
                    # dir_nm = '23-12-27_16-26-12_Entity-Res'

                    # after manual prompt search for each entity type, e.g.
                    # dir_nm = '24-01-16_22-36-17_Entity-Res'
                    # above except #prompt for each entity type 5 => 10
                    # dir_nm = '24-01-17_10-04-12_Entity-Res'

                    # above except larger #list for Title & person classes
                    # dir_nm = '24-02-06_16-04-20_Entity-Res'
                    # above except #prompt for each entity type 10 => 15
                    # dir_nm = '24-02-06_17-04-31_Entity-Res'
                    # dir_nm = '24-02-06_17-27-10_Entity-Res'  # fix: different seeding on same prompt to ensure diversity
                    # dir_nm = '24-02-06_17-39-20_Entity-Res'
                    # above except #list for MPAA rating 15 => 10
                    # dir_nm = '24-02-06_17-57-19_Entity-Res'
                    # above except #list for MPAA rating 10 => 8
                    # dir_nm = '24-02-06_18-03-57_Entity-Res'

                    # larger #list for Title & person classes; #list for MPAA rating 10 => 8
                    # dir_nm = '24-02-06_19-00-30_Entity-Res'
                    # dir_nm = '24-02-06_19-13-45_Entity-Res'  # w/o OpenAI completion seed

                    dir_nm = '24-02-06_20-05-41_Entity-Res'  # Year defn. wording
                else:
                    assert ets == ['Plot', 'Trailer', 'Review']
                    dir_nm = '23-11-21_18-53-19_Entity-Res_{ets=[Plot,Trailer,Review],#demo=1,demo%=0.4}'
            else:
                # dir_nm = '23-11-12_18-48-05_Entity-Res_seeded'

                # re-do after GPT4 & GPT3.5 update
                # dir_nm = '23-11-21_21-38-37_Entity-Res_{seed=query}'
                # dir_nm = '23-11-21_21-51-17_Entity-Res_{seed=query}'  # accidentally ran
                # raise max-token
                # dir_nm = '23-11-21_22-03-22_Entity-Res_{seed=query}'

                # larger entity pool
                # dir_nm = '23-11-21_22-49-49_Entity-Res_{seed=query}'

                # mention the domain twice; sample [`named entities`. `keywords`]
                # dir_nm = '23-12-27_17-14-34_Entity-Res_{seed=query}'

                # prompt template update following non-seeded version
                dir_nm = '24-02-06_22-15-33_Entity-Res_{seed=query}'
        elif dnm == 'mit-restaurant':
            if not sd:
                if ets is None:
                    # dir_nm = '23-11-18_21-03-53_Entity-Res'

                    # mention the domain twice; sample [`named entities`. `keywords`]
                    dir_nm = '23-12-28_01-57-05_Entity-Res'
                else:
                    assert ets == 'Price'
                    dir_nm = '23-11-18_21-21-32_Entity-Res_{#demo=1,demo%=1.0}'
            else:
                # dir_nm = '23-11-18_22-39-52_Entity-Res_{seed=meal}'

                # mention the domain twice; sample [`named entities`. `keywords`]
                dir_nm = '23-12-28_02-19-08_Entity-Res_{seed=meal}'
        elif dnm == 'job-stack':
            if not sd:
                if n_demo_ is None:
                    # dir_nm = '23-11-14_22-10-25_Entity-Res'

                    # mention the domain twice
                    # dir_nm = '23-12-28_14-08-12_Entity-Res'

                    # sample [`named entities`. `keywords`]
                    dir_nm = '23-12-28_14-25-06_Entity-Res'
                else:
                    assert n_demo_ == 1 and dr == 0.4
                    dir_nm = '23-11-14_23-04-36_Entity-Res_{#demo=1,demo%=0.4}'
            else:
                if n_demo_ is None:
                    # dir_nm = '23-11-14_23-28-38_Entity-Res_seeded'

                    # after GPT3.5 update
                    # dir_nm = '23-11-26_18-34-23_Entity-Res_{seed=job}'

                    # mention the domain twice
                    dir_nm = '23-12-28_18-14-55_Entity-Res_{seed=job}'
                else:
                    assert n_demo_ == 1 and dr == 0.4
                    # dir_nm = '23-11-14_23-53-40_Entity-Res_{seed=job,#demo=1,demo%=0.4}'

                    # after GPT3.5 update
                    dir_nm = '23-11-26_18-36-17_Entity-Res_{seed=job,#demo=1,demo%=0.4}'
        elif dnm == 'wiki-gold-no-misc':
            if not sd:
                dir_nm = '23-11-15_04-03-20_Entity-Res'
            else:
                # dir_nm = '23-11-15_04-31-59_Entity-Res_{seed=topic}_debug'
                # dir_nm = '23-11-15_05-05-41_Entity-Res_{seed=topic}_debug'
                # dir_nm = '23-11-15_05-14-24_Entity-Res_{seed=topic}'

                # smaller set of entities per setup
                # dir_nm = '23-11-17_21-23-39_Entity-Res_{seed=topic}'

                # after GPT3.5 update
                dir_nm = '23-11-26_23-02-25_Entity-Res_{seed=topic}'
        else:
            raise NotImplementedError
        pc.process_completions(completions_dir_name=dir_nm)
    # check_prompt()
    # write_completion()
    process()
