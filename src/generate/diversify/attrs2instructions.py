import random
import logging
from typing import List, Tuple, Dict, Union, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from stefutil import *
from src.util import *
from src.util.sample_formats import *
from src.generate.diversify.util import *
from src.generate.diversify import Attribute2Categories


__all__ = ['attr_d2meta', 'DiversityRequirementConstructor']


_logger = get_logger(__name__)


def attr_d2meta(attr2cats_d: Dict[str, Dict[str, Any]], abbreviate: bool = True, attributes: List[str] = None) -> Dict[str, Any]:
    attrs = attributes or attr2cats_d.keys()
    attrs = [attr for attr in attrs if 'presets' in attr2cats_d[attr]]
    d = {attr: attr2cats_d[attr] for attr in attrs}
    return {(v['short'] if abbreviate else k): v['categories'] for k, v in d.items()}


@dataclass
class Instruction:
    attribute_name: Union[str, List[str]] = None
    category: Union[str, List[str], List[Tuple[str, str]]] = None
    sentence: str = None


@dataclass
class EntityInstructionOutput:
    entities: Union[List[Tuple[str, str]], List[str]] = None
    encoded_entities: str = None


def _get_attr2prob(**kwargs) -> Dict[str, float]:
    ret = defaultdict(lambda: 1.)
    ret.update(**kwargs)
    return ret


def lower_1st(sentence: str = None) -> str:
    """
    Lowercase the first character of a sentence
    """
    return sentence[0].lower() + sentence[1:]


class DiversityRequirementConstructor:
    """
    Generates requirement configuration by randomly sampling from category options
    """
    # instruction sampling probability for each attribute

    dataset_name2attr2prob = {
        'conll2003': _get_attr2prob(**{
            # 'length': 0.9, 'perspective': 0.4, 'culture': 0.4,
            'news-category': 1, 'writing-style': 0.4  # reduce average #requirement per prompt
        }),
        'mit-movie': _get_attr2prob(**{
            # 'user-persona': 0.8, 'query-complexity': 0.8, 'language': 0.8,
            # 'preference': 0.4, 'time-period': 0.4, 'culture': 0.4, 'emotion': 0.4

            # reduce sampling prob, try to improve performance, this config has roughly 3, 4 attributes per prompt
            # 'user-persona': 0.4, 'query-complexity': 0.4, 'language': 0.4,
            # 'preference': 0.2, 'time-period': 0.2, 'culture': 0.2, 'emotion': 0.2

            # further reducing sampling prob
            # 'query-category': 0.4,
            # 'user-persona': 0.3, 'query-complexity': 0.3, 'language': 0.3,
            # 'preference': 0.1, 'time-period': 0.1, 'culture': 0.1, 'emotion': 0.1

            # try category is always included
            # 'query-category': 1,
            # 'user-persona': 0.1, 'query-complexity': 0.1, 'language': 0.1,
            # 'preference': 0.05, 'time-period': 0.05, 'culture': 0.05, 'emotion': 0.05

            # re-do after GPT4 & GPT3.5 update
            # 'query-category': 1, 'demographic': 0.1, 'emotion': 0.1, 'language': 0.1,
            # 'query-category': 1, 'demographic': 0.2, 'emotion': 0.2, 'language': 0.2,  # score was worse
            'query-category': 1, 'demographic': 0.4, 'emotion': 0.4, 'language': 0.4,
        }),
        'mit-restaurant': _get_attr2prob(**{
            'meal-category': 1, 'demographic': 0.1, 'ambiance': 0.2, 'price': 0.2, 'dietary': 0.1, 'special': 0.1, 'service': 0.1

            # try larger prob for token diversity
            # 'meal-category': 1, 'demographic': 0.2, 'ambiance': 0.2, 'price': 0.4, 'dietary': 0.2, 'special': 0.2, 'service': 0.2
        }),
        'job-stack': _get_attr2prob(**{
            # 'job-category': 1, 'language': 0.2, 'tone': 0.2, 'experience-level': 0.2
            'job-category': 1, 'language': 0.3, 'tone': 0.3, 'experience-level': 0.3
        }),
        'wiki-gold': _get_attr2prob(topic=1.0, language=0.4),
    }
    dataset_name2attr2prob['conll2003-no-misc'] = dataset_name2attr2prob['conll2003'].copy()
    dataset_name2attr2prob['wiki-gold-no-misc'] = dataset_name2attr2prob['wiki-gold'].copy()
    # dataset_name2attr2prob['conll2003-no-misc'].update({  # try reducing instruction complexity by reducing the sampling prob
    #     'news-category': 0.5, 'location': 0.5, 'writing-style': 0.5
    # })
    # mutually exclusive instructions
    dataset_name2conflict_group = {
        'mit-movie': [['demographic', 'emotion', 'language']],
        'mit-restaurant': [['demographic', 'dietary', 'special'], ['ambiance', 'price', 'service']],
        'job-stack': [['language', 'tone', 'experience-level']],
    }

    def __init__(
            self, dataset_name: str = 'conll2003', a2c: Attribute2Categories = None,
            diverse_context: bool = None, diverse_entity: Union[bool, str] = None, front_matter: Optional[Union[bool, str]] = None,
            dataset_front_matter: bool = None,
            presets: Dict[str, Any] = None, attributes: List[str] = None, group_attributes: bool = None,
            attr2prob: Dict[str, float] = None,
            sample_format: str = 'natural-pair-v2',
            entity_sep: str = None,  entity_pair_map: EntityPairTemplate = None, expected_n_entity: Union[int, float] = 1.5,
            annotate_type: bool = None, logger: logging.Logger = None, drop_prob: float = None, shuffle_instructions: bool = False,
            diverse_entity_seed_ratio: float = None,
    ):
        """
        :param dataset_name: Dataset name
        :param a2c: Attribute2Categories instance
        :param diverse_context: If true, instructions diversify the context
        :param diverse_entity: If true, instructions diversify the entities
        :param front_matter: If true, a prefix front matter description on instruction is included
            If a string is given, override the default description
        :param presets: Preset configuration for Attribute2Categories
        :param attributes: List of attributes to include in the requirement configuration
        :param group_attributes: If true, group relevant
        :param attr2prob: Sampling probability for each attribute
        :param sample_format: Sample format
            Relevant for diverse entity only
        :param entity_sep: Entity separator
            Relevant for diverse entity only
        :param entity_pair_map: Entity pair map
            Relevant for diverse entity only
        :param expected_n_entity: Expected number of entities in the instruction
        :param annotate_type: If true, annotate the type of each entity in the instruction
        :param logger: Logger
        :param drop_prob: Drop probability
            If given, no AttrPrompt instruction at the given probability
        :param shuffle_instructions: If true, instructions are shuffled
        :param diverse_entity_seed_ratio: Ratio of seeded diverse entity
            Relevant for mixed diverse entity
        """
        ca(dataset_name=dataset_name)
        self.dataset_name = dataset_name

        self.presets = presets
        a2c_args = dict(
            diverse_context=diverse_context, diverse_entity=diverse_entity, attributes=attributes,
            diverse_entity_seed_ratio=diverse_entity_seed_ratio)
        self.a2c = a2c or Attribute2Categories(dataset_name=dataset_name, presets=presets, **a2c_args)
        self.diverse_context, self.diverse_entity = self.a2c.diverse_context, self.a2c.diverse_entity
        # self.diverse_entity_seed_ratio = self.a2c.diverse_entity_seed_ratio
        self.attributes = self.a2c.allowed_attributes
        self.diverse_entity_attr_name = self.a2c.diverse_entity_attr_name
        self.front_matter = front_matter or True
        self.dataset_front_matter = dataset_front_matter if dataset_front_matter is not None else True

        self.group_attributes = group_attributes
        self.attribute_groups = []
        if group_attributes:
            for attr in self.attributes:
                if attr in ['news-category', 'writing-style', 'location']:
                    self.attribute_groups.append([attr])
                elif attr == 'sub-category':
                    # insert into the same group as `news-category`
                    last_grp = self.attribute_groups[-1]
                    assert 'news-category' in last_grp
                    last_grp.append(attr)
                else:
                    raise NotImplementedError
        else:
            self.attribute_groups = [[attr] for attr in self.attributes]

        self.attr2prob = DiversityRequirementConstructor.dataset_name2attr2prob[dataset_name].copy()
        if attr2prob is not None:
            for attr, prob in attr2prob.items():
                self.attr2prob[attr] = prob
        if self.dataset_name == 'conll2003':
            assert self.attr2prob['news-category'] == 1.  # sanity check
        self.conflict_group = DiversityRequirementConstructor.dataset_name2conflict_group.get(dataset_name, None)

        self.entity_types = sconfig(f'datasets.{dataset_name}.readable-entity-types')
        self.sample_format, self.entity_sep, self.entity_pair_map = sample_format, None, None
        if self.diverse_entity:
            if sample_format != 'natural-pair-v2':
                raise NotImplementedError
            self.entity_sep = entity_sep or get_default_entity_sep(sample_format=sample_format)
            self.entity_pair_map = entity_pair_map or get_default_entity_pair_map(sample_format=sample_format)
        self.expected_n_entity = expected_n_entity or 1.5
        self.annotate_type = annotate_type or False
        self.drop_prob = drop_prob
        self.shuffle_instructions = shuffle_instructions

        self.logger = logger or _logger
        self.init_log()

    def init_log(self, logger: logging.Logger = None):
        attrs = self.attributes.copy()
        if ENTITY_KEY_MIXED in attrs:
            attrs.remove(ENTITY_KEY_MIXED)
            attrs += [ENTITY_KEY, ENTITY_KEY_SEEDED]
        d_log = {
            'dataset-name': self.dataset_name, 'diverse-context': self.diverse_context, 'diverse-entity': self.diverse_entity,
            'attributes': self.attributes, 'group-attributes': self.group_attributes,
            'attribute-sampling-prob': self.attr2prob, 'conflict-group': self.conflict_group,
            'presets': self.presets, 'preset-categories': self.a2c.get_preset_categories(attributes=attrs),
            'dependency-map': self.a2c.get_dependency_map(attributes=attrs),
            'attr2file-dir-name': self.a2c.get_attr2file_dir(attributes=attrs),
            'sample-format': self.sample_format, 'entity-types': self.entity_types, 'entity-sep': self.entity_sep,
            'entity-pair-map': self.entity_pair_map.__class__.__qualname__, 'expected-#entity': self.expected_n_entity,
            'annotate-type': self.annotate_type, 'drop-probability': self.drop_prob, 'shuffle-instructions': self.shuffle_instructions
        }
        (logger or self.logger).info(f'{pl.i(self.__class__.__qualname__)} initialized w/ {pl.i(d_log, indent=1)}')

    def __call__(self, prefix: str = None, generator: Union[random.Random, int] = None) -> Optional[str]:
        """
        :return: A string of instructions
            In case no instructions are generated, None is returned
        """
        gen = get_random_generator(generator=generator)
        if self.drop_prob is not None and gen.random() < self.drop_prob:
            return None

        instrs = []
        seed = None if self.group_attributes else dict()
        for attr in self.attribute_groups:
            include = gen.random() < self.attr2prob[attr[0]]
            if not include:
                continue

            # if drop_prob is specified and got here, need to enforce at least 1 instruction to ensure the ratio
            out = self._get_single(attribute_name=attr, seed_category=seed, enforce=self.drop_prob is not None, generator=gen)
            if out is None:
                continue

            if not self.group_attributes:  # TODO: always return seed
                seed[out.attribute_name] = out.category
            instrs.append(out)

        assert seed is not None
        if self.conflict_group:
            for attrs in self.conflict_group:
                instrs_ = [i for i in instrs if i.attribute_name in attrs]
                if len(instrs_) > 1:
                    # randomly select one, drop the rest
                    instr_keep = gen.choice(instrs_)
                    instrs_drop = [i for i in instrs_ if i != instr_keep]
                    instrs = [i for i in instrs if i not in instrs_drop]
        instrs = [i.sentence for i in instrs]

        instrs_ = []  # add index and punctuation
        n = len(instrs)
        single_instr = n == 1
        if single_instr:  # no indexing
            instrs_ = [f'{instrs[0]}.']
        else:
            for i, instr in enumerate(instrs, start=1):
                punc = '.' if i == n else ';'
                instrs_.append(f'{i}. {instr}{punc}')
        if len(instrs_) > 0:  # otherwise, no instructions
            if self.shuffle_instructions:
                gen.shuffle(instrs_)

            strt = f'{self._get_starter(prefix=prefix, n_instr=len(instrs_))}'
            if single_instr and strt[-1] != '\n':  # if continue on the same line, lowercase 1st char
                return f'{strt}{lower_1st(instrs_[0])}'
            else:
                instrs = '\n'.join(instrs_)
                return f'{strt}{instrs}'

    def _get_starter(self, prefix: str = None, n_instr: int = None) -> str:
        if self.dataset_front_matter is True:
            dc, de = self.diverse_context, self.diverse_entity
            if self.dataset_name == 'conll2003-no-misc':
                if dc and not de:
                    return 'Additionally, the generated news article sentences should follow the requirements below:\n'
                elif not dc and de:
                    return 'Additionally, in the generated sentences, '
                else:
                    assert dc and de
                    return 'Additionally, the generated news article sentences should follow the requirements below:\n'
            elif self.dataset_name == 'mit-movie':
                if dc:
                    return 'Additionally, the generated spoken movie queries should follow the requirements below:\n'
                else:  # diverse entity only
                    return 'Additionally, in the generated queries, '
            elif self.dataset_name == 'mit-restaurant':
                if dc:
                    return 'Additionally, the generated restaurant queries should follow the requirements below:\n'
                else:
                    return 'Additionally, in the generated queries, '
            elif self.dataset_name == 'job-stack':
                if dc:
                    return 'Additionally, the generated job description sentences should follow the requirements below:\n'
                else:
                    return 'Additionally, in the generated sentences, '
            elif self.dataset_name in ['wiki-gold', 'wiki-gold-no-misc']:
                if dc:
                    return 'Additionally, the generated Wikipedia article sentences should follow the requirements below:\n'
                else:
                    return 'Additionally, in the generated sentences, '
            else:
                raise NotImplementedError
        else:
            if self.front_matter is True:
                rq = 'requirements' if n_instr > 1 else 'requirement'
                if prefix:
                    return f'{prefix}please follow the {rq} below:\n'
                else:
                    return f'Please follow the {rq} below:\n'
            elif isinstance(self.front_matter, str):
                return self.front_matter
            else:
                return ''

    def meta(self, abbreviate: bool = True) -> Dict[str, Any]:
        ret = dict()
        if self.presets is not None:
            ret['presets'] = attr_d2meta(attr2cats_d=self.a2c.d, abbreviate=abbreviate, attributes=self.attributes)
        if self.group_attributes is not None:
            ret['group_attributes'] = self.group_attributes
        return ret

    def _get_single(
            self, attribute_name: Union[str, List[str]] = None, seed_category: Dict[str, Any] = None,
            generator: Union[random.Random, int] = None, **kwargs
    ) -> Optional[Instruction]:
        if len(attribute_name) == 1 and attribute_name[0] in [ENTITY_KEY, ENTITY_KEY_SEEDED, ENTITY_KEY_MIXED]:
            f = self._get_single_y
        else:
            f = self._get_single_x
        gen = get_random_generator(generator=generator)
        return f(attribute_name=attribute_name, seed_category=seed_category, generator=gen, **kwargs)

    def _get_single_y(
            self, attribute_name: Union[str, List[str]] = None, seed_category: Dict[str, Any] = None, enforce: bool = False,
            generator: Union[random.Random, int] = None
    ) -> Optional[Instruction]:
        gen = get_random_generator(generator=generator)
        assert len(attribute_name) == 1
        attr = attribute_name[0]
        assert attr in [ENTITY_KEY, ENTITY_KEY_SEEDED, ENTITY_KEY_MIXED]
        options = self.a2c(attribute_name=attr)

        assert isinstance(options, dict)
        is_seeded = all(isinstance(v, dict) for v in options.values())
        if is_seeded:
            assert attr in [ENTITY_KEY_SEEDED, ENTITY_KEY_MIXED]
            if not self.diverse_context:  # entity requirement is the only requirement
                assert seed_category == dict()  # sample a news category
                seed_category = {self.a2c.diverse_entity_seed_attr_name: sample_single(list(options.keys()), generator=gen)}
            else:  # diverse context should have already sampled the seed category
                assert seed_category != dict()
            seed_cat = seed_category[self.a2c.diverse_entity_seed_attr_name]
            options = options[seed_cat]

        out = self._get_entities(options=options, at_least_1=enforce, generator=gen)
        if out is None:
            return None
        n_et = len(out.entities)
        if self.annotate_type:
            # sent = f'Try to include some of the following named entities if you can: [{entities}]'
            tm, tm_pl = 'named entity', 'named entities'
            # try to make it less complex
            # sent = f'Try to include the following named entities: [{entities}]'
            # sent = f'Include the following named entities: [{out.encoded_entities}]'
        else:  # just the entity span, no entity type
            tm, tm_pl = 'term', 'terms'
        if n_et == 1:
            sent = f'Include the {tm} [{out.encoded_entities}]'
            # sent = f'Include the term {out.encoded}'  # drop brackets if just 1 entity
            # sent = f'focus on {seed_cat} and include the term: [{out.encoded}]'
        else:
            sent = f'Include the following {tm_pl}: [{out.encoded_entities}]'
            # sent = f'focus on {seed_cat} and include the following terms: [{out.encoded}]'
        if self.diverse_context:
            if self.dataset_name in ['conll2003', 'conll2003-no-misc']:
                # sent = f'The news story sentences should {lower_1st(sent)}'
                sent = f'The generated news article sentences should {lower_1st(sent)}'
                # sent = f'The generated news article sentences should be about {seed_cat} and {lower_1st(sent)}'
                # sent = f'{lower_1st(sent)}'
            elif self.dataset_name == 'mit-movie':
                sent = f'The generated spoken movie queries should {lower_1st(sent)}'
            elif self.dataset_name == 'mit-restaurant':
                sent = f'The generated restaurant queries should {lower_1st(sent)}'
            elif self.dataset_name == 'job-stack':
                sent = f'The generated job description sentences should {lower_1st(sent)}'
            elif self.dataset_name in ['wiki-gold', 'wiki-gold-no-misc']:
                sent = f'The generated Wikipedia article sentences should {lower_1st(sent)}'
            else:
                raise NotImplementedError
        return Instruction(attribute_name=attr, category=out.entities, sentence=sent)

    def _get_single_x(
            self, attribute_name: Union[str, List[str]] = None, seed_category: Dict[str, Any] = None, enforce: bool = False,
            generator: Union[random.Random, int] = None
    ) -> Optional[Instruction]:
        gen = get_random_generator(generator=generator)
        if enforce:
            sic(enforce)
            raise NotImplementedError
        if self.dataset_name in ['conll2003', 'conll2003-no-misc']:
            assert attribute_name in [
                ['news-category'], ['length'], ['writing-style'], ['location'], ['perspective'], ['culture'],
                ['sub-category'], ['news-category', 'sub-category'],
                [ENTITY_KEY], [ENTITY_KEY_SEEDED]
            ]
            attr = attribute_name[0]
            if not self.group_attributes:
                assert len(attribute_name) == 1  # sanity check
            options = self.a2c(attribute_name=attr)
            cat = None
            if self.diverse_context and attr not in [self.a2c.default_x_attributes] + [self.diverse_entity_attr_name]:
                # dictionary of lists based on seed category
                assert isinstance(options, list)  # sanity check
                cat = sample_single(options, generator=gen)

            if attribute_name == ['news-category', 'sub-category']:
                assert self.group_attributes and seed_category is None
                cat_sub = sample_single(self.a2c(attribute_name='sub-category', seed_attribute_name=cat), generator=gen)
                sent = f'The news should be about {cat}, specifically about {cat_sub}'
                return Instruction(attribute_name=attribute_name, category=[cat, cat_sub], sentence=sent)
            elif len(attribute_name) == 1:
                if attr in ['news-category', 'length', 'writing-style', 'location']:
                    cat = sample_single(options, generator=gen)
                    if attr == 'news-category':
                        if self.diverse_entity:
                            # sent = f'The news story sentences should be about {cat}'
                            sent = f'The generated news article sentences should be about {cat}'
                            # sent = f'The generated news article sentences should focus on {cat}'
                            # sent = f'be about {cat}'
                        else:
                            sent = f'The news should be about {cat}'
                        # sent = f'should be about {cat}'
                    elif attr == 'length':
                        # GPT doesn't seem to follow this, maybe cos it assumes instruction for the whole passage
                        # sent = f'The length of the news should be {cat}'
                        # sent = f'The length of the sentences should be {cat}'
                        sent = f'The length of each sentence should be {cat}'
                    elif attr == 'writing-style':
                        if self.diverse_entity:
                            # sent = f'The writing style of the news story sentences should be {cat}'
                            sent = f'The writing style of the generated news article sentences should be {cat.lower()}'
                            # sent = f'represent the writing style of {cat}'
                        else:
                            sent = f'The writing style of the news should be {cat.lower()}'
                        # sent = f'the writing style should be {cat}'
                    else:
                        assert attr == 'location'
                        sent = f'The location of the news should be in {cat}'
                        # sent = f'should be in {cat}'
                        # sent = f'should involve events in {cat}'
                    return Instruction(attribute_name=attr, category=cat, sentence=sent)
                elif attr in Attribute2Categories.dataset_name2dependent_x_attributes[self.dataset_name]:
                    assert not self.group_attributes and seed_category is not None
                    cat = sample_single(options[seed_category['news-category']], generator=gen)
                    if attr == 'sub-category':
                        sent = f'The news should focus on {cat}'
                    elif attr == 'perspective':
                        # the categories themselves may start with terms `perspective` or `viewpoint`
                        # sent = f'The news should include the perspective of {cat}'
                        # sent = f'The news should include source or commentary from {cat}'
                        terms = ['perspective', 'viewpoint', 'opinion', 'analysis', 'voice', 'stories', 'reaction']
                        has_term = any(t in cat.lower() for t in terms)
                        if has_term:
                            sent = f'The news should include {cat}'
                        else:
                            sent = f'The news should include the perspective of {cat}'
                    else:
                        assert attr == 'culture'
                        sent = f'The news should consider the aspect of {cat}'
                    return Instruction(attribute_name=attr, category=cat, sentence=sent)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        elif self.dataset_name == 'mit-movie':
            assert len(attribute_name) == 1
            attr = attribute_name[0]
            options = self.a2c(attribute_name=attr)
            cat = sample_single(options, generator=gen)
            if attr == 'query-category':
                # sent = f'The query category should be {cat}'
                # sent = f'The spoken movie queries should be about {cat}'

                # re-do after GPT4 & GPT3.5 update
                sent = f'The query should inquire about {cat}'
            elif attr == 'user-persona':
                sent = f'The user persona should be {cat}'
                # sent = f'The user persona of the spoken movie queries should be {cat}'
            elif attr == 'preference':
                # heuristics, change wording if `preference` already appears in `cat`
                terms = ['preference', 'interest']
                has_term = any(t in cat.lower() for t in terms)
                if has_term:
                    assert any(cat.lower().startswith(t) for t in terms)  # sanity check
                    sent = f'The user should have {cat}'
                else:
                    sent = f'The user preference should be {cat}'
            elif attr == 'demographic':
                sent = f'The user demographic should be {cat.lower()}'
            elif attr == 'query-complexity':
                sent = f'The query should be {cat}'
                # sent = f'The spoken movie queries should be {cat}'
            elif attr == 'time-period':
                sent = f'The time period of the query should be {cat}'
                # sent = f'The time period of the spoken movie queries should be {cat}'
            elif attr == 'culture':
                sent = f'The query should include the event of {cat}'
                # sent = f'The spoken movie queries should include the event of {cat}'
            elif attr == 'emotion':
                # sent = f'The emotion or mood of the user should be {cat}'

                # re-do after GPT4 & GPT3.5 update
                # sent = f'The mood of the user should be {cat.lower()}'
                sent = f'The user should be {cat.lower()}'
            else:
                assert attr == 'language'
                sent = f'The query language should be {cat.lower()}'
                # sent = f'The language of the spoken movie queries should be {cat}'
            return Instruction(attribute_name=attr, category=cat, sentence=sent)
        elif self.dataset_name == 'mit-restaurant':
            assert len(attribute_name) == 1
            attr = attribute_name[0]
            options = self.a2c(attribute_name=attr)
            cat = sample_single(options, generator=gen)
            if attr == 'meal-category':
                sent = f'The meal category should be {cat}'  # already in lowercase
            elif attr == 'demographic':
                sent = f'The user demographic should be {cat.lower()}'
            elif attr == 'ambiance':
                sent = f"The restaurant's ambience should be {cat.lower()}"
            elif attr == 'price':
                sent = f"The restaurant's price range should be {cat.lower()}"
            elif attr == 'dietary':
                sent = f"The user's dietary restriction should be {cat.lower()}"
            elif attr == 'special':
                sent = f'The restaurant should offer {cat.lower()}'
            else:
                assert attr == 'service'
                sent = f'The restaurant should support {cat}'
            return Instruction(attribute_name=attr, category=cat, sentence=sent)
        elif self.dataset_name == 'job-stack':
            assert len(attribute_name) == 1
            attr = attribute_name[0]
            options = self.a2c(attribute_name=attr)
            cat = sample_single(options, generator=gen)
            if attr == 'job-category':
                sent = f'The job category should be {cat}'
            elif attr == 'language':
                sent = f'The language of the job description should be {cat.lower()}'
            elif attr == 'experience-level':
                sent = f'The experience level of the job should be {cat.lower()}'
            elif attr == 'location':
                sent = f'The location of the job should be {cat.lower()}'
            elif attr == 'culture':
                raise NotImplementedError
            else:
                assert attr == 'tone'
                sent = f'The tone of the job description should be {cat.lower()}'
            return Instruction(attribute_name=attr, category=cat, sentence=sent)
        elif self.dataset_name in ['wiki-gold', 'wiki-gold-no-misc']:
            assert len(attribute_name) == 1
            attr = attribute_name[0]
            options = self.a2c(attribute_name=attr)
            cat = sample_single(options, generator=gen)
            if attr == 'topic':
                sent = f'The topic of the Wikipedia article sentences should be {cat}'
            else:
                assert attr == 'language'
                sent = f'The writing style of the Wikipedia article sentences should be {cat.lower()}'
            return Instruction(attribute_name=attr, category=cat, sentence=sent)
        else:
            raise NotImplementedError

    def _get_entities(
            self, options: Dict[str, List[str]] = None, at_least_1: bool = False, generator: Union[random.Random, int] = None
    ) -> Optional[EntityInstructionOutput]:
        gen = get_random_generator(generator=generator)
        pairs = sum([[(enm, et) for enm in sample_few(options[et], max_=3, generator=gen)] for et in self.entity_types], start=[])
        if self.annotate_type:
            elms = [self.entity_pair_map(nm, tp) for nm, tp in pairs]
        else:
            elms = [nm for nm, tp in pairs]
            # drop potential duplicates in case the same entity fails into different entity types
            # e.g. WikiGold: Medical advancements and breakthroughs
            #   => National Institutes of Health (NIH) for both ORG and LOC
            elms = list(dict.fromkeys(elms))  # preserve order for deterministic sampling

        if len(elms) == 0:
            if at_least_1:
                et = gen.choice(self.entity_types)
                elms = [gen.choice(options[et])]
                enc = self.entity_pair_map(elms[0], et) if self.annotate_type else elms[0]
                return EntityInstructionOutput(entities=[(elms[0], et)], encoded_entities=enc)
            else:
                return None
        # Now expected length is 1.5 x #entity => reduce expected length back to 1.5 by independent sampling
        ratio = self.expected_n_entity / len(elms)
        elms = [pair for pair in elms if gen.random() < ratio]
        if len(elms) == 0:
            if at_least_1:
                elms = [gen.choice(elms)]
            else:
                return None
        # # this is effectively a poisson distribution
        # n = np.random.poisson(lam=self.expected_n_entity)
        # if n == 0:
        #     if at_least_1:
        #         n = 1
        #     else:
        #         return None
        # if n < len(elms):
        #     elms = random.sample(elms, k=n)  # not in the same order as the original list already
        gen.shuffle(elms)
        return EntityInstructionOutput(entities=elms, encoded_entities=f'{self.entity_sep} '.join(elms))


if __name__ == '__main__':
    from src.data_util import print_prompts

    # dnm = 'conll2003'
    # dnm = 'conll2003-no-misc'
    dnm = 'mit-movie'
    # dnm = 'mit-restaurant'
    # dnm = 'job-stack'
    # dnm = 'wiki-gold'
    # dnm = 'wiki-gold-no-misc'

    sd = 42
    gen_ = get_random_generator(generator=sd)

    def check_x():
        attrs, pst, ga = None, None, None
        dp = None
        # dp = 0.3

        if dnm in ['conll2003', 'conll2003-no-misc']:
            # attrs = ['news-category', 'writing-style', 'location']
            # attrs = ['news-category', 'sub-category', 'writing-style', 'location']
            # attrs = ['news-category', 'perspective', 'writing-style', 'location']
            attrs = None

            # pst = dict(cat='9')
            # pst = dict(cat='30', sub='30=>10')
            # pst = dict(cat='30')
            # pst = dict(cat='30', per='30=>5')
            # pst = dict(cat='30', per='30=>5', cul='30=>5')
            # pst = dict(cat='30', cul='30=>5')
            # ga = True
            # ga = False
            ga = None
        rc = DiversityRequirementConstructor(
            dataset_name=dnm, diverse_context=True, presets=pst, attributes=attrs, group_attributes=ga, drop_prob=dp)
        sic(rc.meta(), rc.attributes)

        # n = 10
        n = 5
        prompts = [rc(generator=gen_) for _ in range(n)]
        print_prompts(prompt=prompts)
    # check_x()

    def check_y():
        fm = None
        # fm = 'Additionally, in your generated sentences, '
        # de = True
        de = 'seeded'
        # de = 'mixed'
        a_t = True
        # a_t = False
        dp = None
        # dp = 0.1
        rc = DiversityRequirementConstructor(
            dataset_name=dnm, diverse_context=False, diverse_entity=de, front_matter=fm, annotate_type=a_t, drop_prob=dp)
        sic(rc.meta(), rc.attributes)
        n = 20
        # n = 5
        prompts = [rc(generator=gen_) for _ in range(n)]
        prompts = [p for p in prompts if p is not None]  # drop if not sampled
        print_prompts(prompt=prompts)
    check_y()

    def check_x_y():
        # the best-performing one after GPT3.5 update
        # fm = 'Additionally, the generated news article sentences should follow the requirements below:\n'
        # fm = 'Additionally, the generated news article sentences should:\n'
        # fm = 'Additionally, in your generated sentences, '
        fm = None
        dc = True
        # dc = False
        de = 'seeded'
        rc = DiversityRequirementConstructor(dataset_name=dnm, diverse_context=dc, diverse_entity=de, front_matter=fm)
        sic(rc.meta(), rc.attributes)
        print_prompts(prompt=rc, n=10)
    # check_x_y()
