"""
Broken into a few steps
1. For a given attribute dimension, construct prompt
2. Sent prompt to GPT API to get completions
3. Process completions to get list of category values
"""


import re
import json
import random
from os.path import join as os_join
from typing import List, Dict, Union, Iterable, Optional

from stefutil import *
from src.util import *
from src.data_util import *
from src.generate.diversify.util import *
from src.generate.diversify import Attribute2Categories


__all__ = ['CategoryGenerator']


_logger = get_logger('Attr Cat Gen')


class CategoryGenerator(OptionGenerator):
    """
    Generate categories for each attribute dimension
    """
    dataset_name2default_gen_attributes = {
        dnm: Attribute2Categories.dataset_name2default_x_attributes[dnm].copy()
        for dnm in ['conll2003', 'mit-movie', 'job-stack', 'wiki-gold']
    }
    attrs_rest = Attribute2Categories.dataset_name2default_x_attributes['mit-restaurant'].copy()
    attrs_rest.remove('service')  # drop `service` for the default 4 categories looks good
    dataset_name2default_gen_attributes['mit-restaurant'] = attrs_rest

    dataset_name2attribute2n_list = {
        'conll2003': {'news-category': 30, 'writing-style': 8, 'sub-category': 10, 'subtopic': 10, 'perspective': 10, 'culture': 10},
        'mit-movie': {
            # 'query-category': 15, 'user-persona': 20, 'preference': 10, 'query-complexity': 10, 'time-period': 15, 'culture': 10,
            # 'emotion': 15, 'language': 8

            # re-do after GPT4 & GPT3.5 update
            # 'query-category': 10, 'demographic': 10, 'culture': 10, 'language': 8, 'emotion': 10
            'query-category': 10, 'demographic': 6, 'culture': 6, 'language': 8, 'emotion': 10
        },
        'mit-restaurant': {'meal-category': 10, 'demographic': 10, 'ambiance': 8, 'price': 5, 'dietary': 8, 'special': 10},
        'job-stack': {'job-category': 8, 'language': 10, 'experience-level': 5, 'location': 5, 'culture': 5, 'tone': 5},
        'wiki-gold': {'topic': 30, 'language': 10}}
    dataset_name2attribute2n_list['conll2003-no-misc'] = dataset_name2attribute2n_list['conll2003'].copy()
    dataset_name2attribute2n_list['wiki-gold-no-misc'] = dataset_name2attribute2n_list['wiki-gold'].copy()

    def __init__(
            self, dataset_name: str = 'conll2003', attributes: Union[str, List[str]] = None,
            a2c: Attribute2Categories = None, presets: Dict[str, str] = None,  **kwargs
    ):
        super().__init__(**kwargs)
        if dataset_name not in [
            'conll2003', 'conll2003-no-misc', 'wiki-gold', 'wiki-gold-no-misc', 'mit-movie', 'mit-restaurant', 'job-stack']:
            raise NotImplementedError
        if dataset_name.endswith('-no-misc'):
            # drop it, since w/ or w/o misc, same logic
            dataset_name = dataset_name[:-8]
        self.dataset_name = dataset_name
        self.sample_type = 'Category'
        self.dir_args = dict(dataset_name=self.dataset_name, sub_dir=DIVERSE_CONTEXT_DNM)

        if isinstance(attributes, str):
            attributes = [attributes]
        self.attributes = attributes or CategoryGenerator.dataset_name2default_gen_attributes[dataset_name].copy()
        self.a2c = a2c or Attribute2Categories(dataset_name=dataset_name, presets=presets)

        # use parent API for iterating setups
        self.keyword2n_list = CategoryGenerator.dataset_name2attribute2n_list[self.dataset_name].copy()
        self.keywords = self.attributes

        # Many different formats since I didn't specify anything, e.g.
        #   1. `15. **category name**: description`
        #   2. `15. category name: description`
        #   3. `15) category name: description`
        #   4. `15. category name - description`
        #   5. `15. category name. description`
        #   6. `15. category name (description)`
        #   7. `15) category name:`  # description is on the next line
        #   8. `15. category name`
        #   9. `category name`

        # for edge case where each bullet is an entire passage, keep only the first sentence, e.g.
        #   The perspective of social justice advocates who scrutinize the social impact and ethical considerations
        #   surrounding technology companies and startups.
        #   This viewpoint may address issues like algorithmic bias, discrimination in AI technologies,
        #   and how companies handle controversial content moderation on their platforms.
        self.pattern_option = [
            re.compile(r'(?P<idx>\d+)([.)])\s(\*\*)?(?P<value>.+)(\*\*)?: (?P<description>.+)'),
            re.compile(r'(?P<idx>\d+)\.\s(?P<value>.+) - (?P<description>.+)'),
            re.compile(r'(?P<idx>\d+)\.\s(?P<value>.+)\s*\((?P<description>.+)\)'),
            re.compile(r'(?P<idx>\d+)\.\s(?P<value>.+)\. (?P<description>.+)'),
            re.compile(r'(?P<idx>\d+)\.\s(?P<value>.+):?'),
            re.compile(r'(?P<value>.+)')
        ]

        self.logger = _logger

    @property
    def meta(self) -> Optional[str]:
        d_meta = dict()
        if self.attributes:
            if self.attributes != CategoryGenerator.dataset_name2default_gen_attributes[self.dataset_name]:
                d_meta['attr'] = self.attributes
            # omit if same as default
        dep = self.a2c.get_dependency_map(attributes=self.attributes)
        if dep:
            d_meta['dep'] = dep
        return pl.pa(d_meta) if d_meta != dict() else None

    def get_prompt(
            self, n_list: int = None, attribute_name: str = 'news-category', seed_category: str = None,
            generator: random.Random = None
    ) -> str:
        """
        :param attribute_name: Attribute name
        :param n_list: Number of categories to list
        :param seed_category: Seed category to generate categories for
        :param generator: Random generator
        """
        gen = get_random_generator(generator=generator)
        if self.dataset_name == 'conll2003':
            prefix = 'Suppose you are a news writer for Reuters.'
            d = self.a2c[attribute_name]
            desc, egs = d.get('desc'), d.get('examples')

            if attribute_name == 'news-category':
                ret = f'List {n_list} diverse news categories'
            elif attribute_name == 'writing-style':
                ret = f'List {n_list} different writing styles'
            else:
                raise NotImplementedError

            if desc is not None:
                ret = f'{ret}, i.e. {desc},'
            ret = f'{prefix} {ret} for news articles.'
            if egs is not None:
                if isinstance(egs[0], list):
                    egs = gen.sample(egs, 1)[0]
                    assert isinstance(egs, list) and all(isinstance(e, str) for e in egs)  # sanity check

                gen.shuffle(egs)
                ret = f'{ret} Some examples are {pl.nc(egs)}.'
            return ret
        elif self.dataset_name == 'mit-movie':
            assert seed_category is None
            # prefix = 'Suppose you are the user of a dialog system or conversational agent. '
            prefix = 'Suppose you are a user of a dialog system or conversational agent.'
            d = self.a2c[attribute_name]
            desc, egs = d.get('desc'), d.get('examples')

            if attribute_name == 'query-category':
                ret = f'List {n_list} diverse movie query categories'
            elif attribute_name == 'user-persona':
                ret = f'List {n_list} diverse user personas'
            elif attribute_name == 'demographic':
                ret = f'List {n_list} different user demographics'
            elif attribute_name == 'preference':
                ret = f'List {n_list} different user preferences'
            elif attribute_name == 'query-complexity':
                ret = f'List {n_list} different query complexities'
            elif attribute_name == 'time-period':
                ret = f'List {n_list} different time periods'
            elif attribute_name == 'culture':
                # doesn't seem to integrate well by manual inspection, especially topic
                # instr_ = f'List {n_list} diverse cultural references or trending topics and events'
                # responses are mostly topics, not events
                # instr_ = f'List {n_list} diverse trends and events'
                # ret = f'List {n_list} diverse events'

                # re-do after GPT4 & GPT3.5 update
                # ret = f'List {n_list} different cultural references'
                ret = f'List {n_list} different cultural or regional references'
            elif attribute_name == 'emotion':
                # instr_ = f'List {n_list} diverse emotions or moods'

                # re-do after GPT4 & GPT3.5 update
                ret = f'List {n_list} different user moods'
            else:
                assert attribute_name == 'language'
                # ret = f'List {n_list} different language styles'

                # re-do after GPT4 & GPT3.5 update
                ret = f'List {n_list} different language variations'
            if desc is not None:
                ret = f'{ret}, i.e. {desc.lower()},'
            ret = f'{prefix} {ret} for movie queries to the dialog system.'
            if egs is not None:
                if isinstance(egs[0], list):
                    egs = gen.sample(egs, 1)[0]
                    assert isinstance(egs, list) and all(isinstance(e, str) for e in egs)  # sanity check
                gen.shuffle(egs)
                ret = f'{ret} Some examples are {pl.nc(egs)}.'
            return ret
        elif self.dataset_name == 'mit-restaurant':
            assert seed_category is None
            prefix = 'Suppose you are a user of a dialog system or conversational agent.'

            d = self.a2c[attribute_name]
            desc, egs = d.get('desc'), d.get('examples')

            if attribute_name == 'meal-category':
                ret = f'List {n_list} diverse meal types'
            elif attribute_name == 'demographic':
                ret = f'List {n_list} different customer demographics'
            elif attribute_name == 'ambiance':
                ret = f'List {n_list} different restaurant ambiances or settings'
            elif attribute_name == 'price':
                ret = f'List {n_list} different price ranges'
            elif attribute_name == 'dietary':
                ret = f'List {n_list} different dietary restrictions'
            elif attribute_name == 'special':
                ret = f'List {n_list} different special features or offers'
            else:
                raise NotImplementedError

            if desc is not None:
                ret = f'{ret}, i.e. {desc},'
            ret = f'{prefix} {ret} for restaurant queries to the dialog system.'
            if egs is not None:
                gen.shuffle(egs)
                ret = f'{ret} Some examples are {pl.nc(egs)}.'
            return ret
        elif self.dataset_name == 'job-stack':
            assert seed_category is None
            ret = 'Suppose you are a recruiter. '

            d = self.a2c[attribute_name]
            desc, egs = d.get('desc'), d.get('examples')
            if attribute_name == 'job-category':
                ret += f'List {n_list} diverse job categories'
            elif attribute_name == 'language':
                ret += f'List {n_list} different language styles'
            elif attribute_name == 'experience-level':
                ret += f'List {n_list} different experience levels'
            elif attribute_name == 'location':
                ret += f'List {n_list} different locations'
            elif attribute_name == 'culture':
                ret += f'List {n_list} different company cultures and values'
            elif attribute_name == 'tone':
                ret += f'List {n_list} different tones'
            else:
                raise NotImplementedError
            ret = f'{ret} for job descriptions on Stack Overflow.'

            if desc is not None:
                raise NotImplementedError
            if egs is not None:
                ret = f'{ret} Some examples are {pl.nc(egs)}.'
            return ret
        elif self.dataset_name == 'wiki-gold':
            assert seed_category is None
            ret = 'Suppose you are a Wikipedia editor. '

            d = self.a2c[attribute_name]
            desc, egs = d.get('desc'), d.get('examples')
            if attribute_name == 'topic':
                # ret += f'List {n_list} diverse topics'  # generates too-detailed topics for new GPT3.5
                ret += f'List {n_list} diverse categories'
            elif attribute_name == 'language':
                ret += f'List {n_list} different language styles'
            else:
                raise NotImplementedError
            ret = f'{ret} for Wikipedia articles.'

            if desc is not None:
                raise NotImplementedError
            if egs is not None:
                gen.shuffle(egs)
                ret = f'{ret} Some examples are {pl.nc(egs)}.'
            return ret
        else:
            raise NotImplementedError

    def iter_setups(self, n_list: N_List = None) -> Iterable[GenSetup]:
        # TODO: seeded categories
        # larger multiplier since when #list is small, GPT may generate additioanl descriptions
        return self._iter_setups(n_list=n_list, key_name='attribute_name', max_tokens_multiplier=100)

    def write_completions(
            self, n_prompt: int = 5, n_list: int = None, output_dir_nm: str = None, **kwargs
    ) -> completions.WriteCompletionsOutput:
        d_log = dict(attributes=self.attributes)
        return self._write_completions(n_prompt=n_prompt, n_list=n_list, output_dir_nm=output_dir_nm, init_log=d_log, **kwargs)

    def process_completions(
            self, completions_dir_name: str = None, expected_samples_per_completion: N_List = None, output_dir_name: str = None
    ):
        drop_dup = None
        if self.dataset_name in ['conll2003', 'wiki-gold']:
            drop_dup = False  # keep all values to ensure a fair distribution
        if self.dataset_name == 'job-stack':
            # after manual inspection, ensure a uniform distribution of categories
            drop_dup = {'location': False, 'experience-level': False}

        out = self._process_completions(
            completions_dir_name=completions_dir_name, expected_samples_per_completion=expected_samples_per_completion,
            output_dir_name=output_dir_name, drop_duplicates=drop_dup
        )
        output_path, attr2cats, d_sz = out.output_path, out.key2value, out.key2size
        self.logger.info(f'Entity sizes: {pl.i(d_sz)}')

        # filter out some invalid categories
        if self.dataset_name == 'job-stack' and 'location' in attr2cats:
            cats = attr2cats['location']
            cats = [c for c in cats if c != 'Freelance/Contract']  # an invalid category
            attr2cats['location'] = cats
        elif self.dataset_name == 'mit-restaurant' and 'price' in attr2cats:
            cats = attr2cats['price']
            cats = [c for c in cats if c != 'Fine dining']
            attr2cats['price'] = cats

        meta = {'dataset-name': self.dataset_name, 'attributes': self.attributes, 'entity-sizes': d_sz}
        out = {'attribute2categories': attr2cats, 'meta': meta}
        with open(os_join(output_path, 'processed-categories.json'), 'w') as f:
            json.dump(out, f, indent=4)
        _logger.info(f'Processed categories saved to {pl.i(stem(output_path, top_n=3))}')
        return attr2cats

    def extract_values(self, text: str = None, n_expect: int = None) -> List[str]:
        # `min_match` for edge case where only 1 of e.g. 10 rows haa a colon, resulting in 1 matches instead of 10
        ms = self._match_values(text=text, pattern=self.pattern_option, n_expect=n_expect, union_patterns=True)

        # for edge case, e.g.
        #   1. "Women Breaking Barriers in Finance: The Rise of Female CEOs in Fortune 500 Companies" -
        #   This news article could highlight the increasing representation of women in leadership roles
        #   within the finance industry, exploring how it is challenging traditional gender norms
        #   and contributing to broader gender equality movements.
        # the matched group becomes `"Women Breaking Barriers in Finance` with a beginning quote
        return [edit.drop_enclosing_quotes(m.group('value').strip()) for m in ms]


if __name__ == '__main__':
    dnm = 'conll2003'
    # dnm = 'mit-movie'
    # dnm = 'mit-restaurant'
    # dnm = 'job-stack'
    # dnm = 'wiki-gold'

    pst, attrs = None, None
    if dnm == 'conll2003':
        # pst = dict(cat='9')
        # pst = dict(cat='30')
        # attrs = ['subtopic']
        # attrs = ['sub-category']
        # attrs = 'perspective'
        # attrs = 'culture'
        attrs = None
    elif dnm == 'wiki-gold':
        # attrs = None
        attrs = 'topic'
    elif dnm == 'mit-movie':
        # attrs = None
        # attrs = 'user-persona'
        # attrs = ['query-category', 'culture', 'language', 'emotion']
        # attrs = 'culture'
        # attrs = ['query-category', 'culture']
        attrs = 'query-category'
        # attrs = 'demographic'
    elif dnm == 'mit-restaurant':
        # attrs = None
        # attrs = ['meal-category', 'special']
        attrs = 'price'
    cg = CategoryGenerator(dataset_name=dnm, presets=pst, attributes=attrs)

    def check_prompt():
        prompts = [cg.get_prompt(**s.get_prompt_args) for s in cg.iter_setups()]
        prettier.print_prompts(prompt=prompts)

    def write_completion():
        # n_call = 1
        n_call = 3
        # md_nm = 'gpt-3.5-turbo'
        md_nm = 'gpt-3.5-turbo-1106'  # better instruction-following, cheaper
        cg.write_completions(n_prompt=n_call, model_name=md_nm, timeout=30)

    def process():
        if dnm == 'conll2003':
            # dir_nm = '2023-10-10_17-34-01_LLM-Attr-Completion_{attrs=[sub-category]}'
            # dir_nm = '2023-10-10_20-53-27_LLM-Attr-Completion_{attrs=[sub-category]}'  # updated prompt
            # dir_nm = '2023-10-10_23-12-24_LLM-Attr-Completion_{attrs=[sub-category],#list=10,dep={sub<=9}}'
            # dir_nm = '2023-10-10_23-48-27_LLM-Attr-Completion_{attrs=[sub-category],#list=20,dep={sub<=9}}'
            #
            # dir_nm = '2023-10-13_00-02-33_LLM-Attr-Completion_{attrs=[perspective],#list=5,dep={per<=30}}'
            # dir_nm = '2023-10-13_01-22-43_LLM-Attr-Completion_{attrs=[culture],#list=5,dep={cul<=30}}'

            dir_nm = '23-11-19_19-01-31_Category-Res'
        elif dnm == 'mit-movie':
            if attrs is None:
                # dir_nm = '23-11-21_14-20-33_Category-Res'
                dir_nm = '23-11-21_14-43-33_Category-Res'
            elif attrs == ['query-category', 'culture']:
                # dir_nm = '23-11-21_14-25-19_Category-Res_{attr=[query-category,culture]}'
                # dir_nm = '23-11-21_15-59-23_Category-Res_{attr=[query-category,culture]}'
                dir_nm = '23-11-21_16-07-40_Category-Res_{attr=[query-category,culture]}'
            elif attrs == 'query-category':
                # dir_nm = '23-11-21_16-11-14_Category-Res_{attr=[query-category]}'
                dir_nm = '23-11-21_16-14-45_Category-Res_{attr=[query-category]}'
            elif attrs == 'demographic':
                # dir_nm = '23-11-21_16-30-39_Category-Res_{attr=[demographic]}'
                dir_nm = '23-11-21_16-35-44_Category-Res_{attr=[demographic]}'
            else:
                raise NotImplementedError
        elif dnm == 'mit-restaurant':
            if attrs is None:
                # dir_nm = '23-11-18_16-06-46_Category-Res'

                # GPT3.5 update
                # dir_nm = '23-11-18_16-19-22_Category-Res'
                dir_nm = '23-11-21_14-43-33_Category-Res'
            elif attrs == ['meal-category', 'special']:
                dir_nm = '23-11-18_16-33-45_Category-Res_{attr=[meal-category,special]}'
            else:
                assert attrs == 'price'
                dir_nm = '23-11-18_19-13-27_Category-Res_{attr=[price]}'
        elif dnm == 'job-stack':
            # dir_nm = '23-11-14_18-53-15_Category-Res'

            # GPT3.5 update
            dir_nm = '23-11-26_18-02-18_Category-Res'
        elif dnm == 'wiki-gold':
            # dir_nm = '23-11-15_03-07-23_Category-Res'

            # GPT3.5 update
            if attrs is None:
                dir_nm = '23-11-26_22-30-08_Category-Res'
            else:
                assert attrs == 'topic'
                dir_nm = '23-11-26_22-40-34_Category-Res_{attr=[topic]}'
        else:
            raise NotImplementedError
        cg.process_completions(completions_dir_name=dir_nm)
    check_prompt()
    # write_completion()
    # process()

    def check_match():
        txt = "7. Basic information (e.g. genre, language, runtime)"
        pattern = [
            re.compile(r'(?P<idx>\d+)([.)])\s(\*\*)?(?P<value>.+)(\*\*)?: (?P<description>.+)'),
            re.compile(r'(?P<idx>\d+)\.\s(?P<value>.+) - (?P<description>.+)'),
            re.compile(r'(?P<idx>\d+)\.\s(?P<value>.+)\s*\((?P<description>.+)\)'),
            re.compile(r'(?P<idx>\d+)\.\s(?P<value>.+)\. (?P<description>.+)'),
            re.compile(r'(?P<idx>\d+)\.\s(?P<value>.+):?'),
            re.compile(r'(?P<value>.+)')
        ]
        m = patterns.match_row(pattern=pattern, text=txt, verbose=True)
        sic(m)
        sic(m.group('value'))
    # check_match()
