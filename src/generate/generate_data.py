"""
Explore 4 different prompting formats for NER
    1. natural-pair: List of (sentence, label) pairs, each represent a sentence
    2. natural-inline: List of sentences, where each entity is enclosed w/ the entity type
    3. bio-list: List of (list of tokens, list of labels) pairs, each represent a sentence
    4. bio-line: List of tokens & labels, separated by space for each new sentence

Try on `CoNLL-2003` dataset first
"""

import random
import string
from typing import List, Dict, Union, Any

from stefutil import get_logger, pl, ca, get_random_generator
from src.util import sconfig, sample_fmt2original_data_fmt, dataset_meta, dataset_name2data_dir
from src.util.ner_example import DatasetLoader, NerExample, NerReadableExample, NerSpanExample, NerBioExample
from src.util.sample_formats import (
    EntityPairTemplate, TokenMapEnclose, get_default_token_map, get_default_entity_pair_map, get_default_entity_sep
)
from src.data_util import completions, edit
from src.generate import schemas
from src.generate.diversify import DiversityRequirementConstructor


__all__ = [
    'sub_sample_demo', 'DataGenerator'
]


_logger = get_logger(__name__)


def sub_sample_demo(samples: List, scheme: str = 'uniform', generator: Union[random.Random, int] = None) -> List:
    """
    :param samples: A list of samples to be sub-sampled
    :param scheme: subsample size weighting scheme, one of `uniform`, `bernoulli`
    :param generator: Random generator
    """
    gen = get_random_generator(generator=generator)
    ca.assert_options(display_name='Subsample Scheme', val=scheme, options=['uniform', 'bernoulli'])
    if scheme == 'uniform':
        sz = gen.randint(1, len(samples))
        return gen.sample(samples, sz)
    else:
        assert scheme == 'bernoulli'  # number of samples returned centered at half of the original size
        return [s for s in samples if gen.random() < 0.5]


class InstructionConstructor:
    def __init__(
            self, dataset_name: str = 'conll2003', sample_format: str = 'natural-pair',
            as_passage: bool = False, insert: str = None, lowercase_entity_type: bool = None
    ):
        ca(dataset_name=dataset_name, sample_format=sample_format, insert=insert)
        self.dataset_name = dataset_name
        self.sample_format = sample_format
        self.insert = insert
        if insert and sample_format != 'natural-pair-v2':
            raise NotImplementedError
        self.lowercase_entity_type = lowercase_entity_type or False

        self.as_passage = as_passage

    def __call__(self, n_list: int = 20, n_demo: int = None, context_requirement: str = None) -> str:
        """

        :param n_list: Number of sentences that LLM is asked to list
        :param n_demo: # demo examples that will be shown
        :param context_requirement: If given, diversity instructions will be inserted
        :return:
        """
        format_desc, demo_prefix = self._get_format_desc(n_list=n_list), self._get_demo_prefix(n_demo=n_demo)
        if context_requirement is not None:
            # ret = f'{format_desc}\n\n{context_requirement}\n\n{demo_prefix}'
            ret = f'{format_desc}\n\n{context_requirement}\n\n---\n\n{demo_prefix}'  # try add separator
        else:
            ret = f'{format_desc}\n{demo_prefix}'
        return ret

    def _get_format_desc(self, n_list: int = 20):
        if self.as_passage:
            if self.dataset_name == 'conll2003':
                if self.sample_format == 'natural-pair-v2':
                    # with_entity_list = False
                    with_entity_list = True
                    if with_entity_list:
                        ret = ("Suppose you are a news writer. "
                               f"Please generate a synthetic news story/article with around {n_list} sentences.\n"
                               "Then, for each sentence, please write each sentence line by line "
                               "and identify all entity names occurred that belong to one of the following entity types:\n"
                               "[person, location, organization, miscellaneous].\n"
                               "Please list such entities with the corresponding entity types on the following line, "
                               "in the order of occurrence.\n"
                               "If no entity is found in the sentence, list 'None'.")
                    else:
                        ret = (f"Suppose you are a news writer. Please generate a synthetic news story/article with {n_list} sentences.\n"
                               "Then, for each sentence, please write them line by line, "
                               "identify all entity names occurred in the sentences "
                               "and list them with their corresponding entity types on the following line.\n"
                               "If no entity is found in the sentence, list 'None'.")
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:  # standard, one sentence at a time
            if self.sample_format != 'natural-pair-v2' and n_list == 1:
                raise NotImplementedError
            if self.dataset_name == 'conll2003':
                if self.sample_format == 'natural-pair':
                    ret = "Suppose you are a news writer. "\
                          f"Please list {n_list} sentences or phrases from news stories "\
                          "and identify all entity names occurred in the sentences and the types of these entities. "\
                          "If no entity is found in the sentence, list 'None'. "
                elif self.sample_format == 'natural-pair-v2':
                    # zero_shot = False
                    # # zero_shot = True
                    # if zero_shot:
                    #     ret = ("Suppose you are a news writer. "
                    #            f"Please generate {n_list} synthetic sentences or phrases from news stories. "
                    #            f"Please identify all entity names occurred in the sentences "
                    #            f"and list them with their corresponding entity types on the following line.")
                    #     return ret
                    # else:
                    # with_entity_list = False
                    with_entity_list = True
                    if with_entity_list:
                        ret = "Suppose you are a news writer. "
                        if n_list == 1:
                            ret += f"Please generate a synthetic sentence or phrase from news stories. "
                        else:
                            ret += f"Please generate {n_list} synthetic sentences or phrases from news stories. "
                            # try swap `synthetic` for `diverse`
                            # ret += f"Please generate {n_list} diverse sentences or phrases from news stories. "
                        ret += ("Please identify all entity names occurred that belong to one of the following entity types:\n"
                                "[person, location, organization, miscellaneous].\n"
                                "Please list such entities with the corresponding entity types on the following line, "
                                "in the order of occurrence.\n"
                                "If no entity is found in the sentence, list 'None'.")
                    else:
                        ret = ("Suppose you are a news writer. "
                               f"Please generate {n_list} synthetic sentences or phrases from news stories. "
                               "Please identify all entity names occurred in the sentences "
                               "and list them with their corresponding entity types on the following line. "
                               "If no entity is found in the sentence, list 'None'. ")
                elif self.sample_format == 'natural-inline':
                    # ret = ("Suppose you are a news writer. "
                    #        "Please list 20 sentences or phrases from news stories. "
                    #        "Please identify all entity names occurred in the sentences and annotate the types of these entities inline. "
                    #        "Please list them in the following format.")
                    # replace last sentence, instruct LLM to generate *new* samples
                    # ret = ("Suppose you are a news writer. "
                    #        f"Please list {n_list} sentences or phrases from news stories. "
                    #        "Please identify all entity names occurred in the sentences and annotate the types of these entities inline. "
                    #        "Here are some examples.")
                    # add both these are examples, and ask LLM to follow this format
                    # ret = ("Suppose you are a news writer. "
                    #        f"Please list {n_list} sentences or phrases from news stories. "
                    #        "Please identify all entity names occurred in the sentences and annotate the types of these entities inline. "
                    #        "Here are some examples. Please follow this format.")
                    ret = ("Suppose you are a news writer. "
                           f"Please list {n_list} sentences or phrases from news stories. "
                           "Please identify all entity names that occurred in the sentences and annotate the types of these entities inline. ")
                elif self.sample_format == 'natural-inline-v2':
                    ret = ("Suppose you are a news writer. "
                           f"Please generate {n_list} synthetic sentences or phrases from news stories. "
                           "On a new line, please identify all entity names that occurred in the sentences "
                           "and annotate their corresponding entity types inline.")
                elif self.sample_format == 'bio-list':
                    # ret = ("Suppose you are a news writer and a named entity tag annotator. "
                    #        "Please list 20 sentences or phrases from news stories, break them into tokens, "
                    #        "and annotate each token with the NER tag using the Inside–outside–beginning tagging format. "
                    #        "Please list them in the following format.")
                    # drop annotator role
                    # ret = ("Suppose you are a news writer. "
                    #        "Please list 20 sentences or phrases from news stories, token by token, "
                    #        "and annotate each token with the NER tag using the Inside–outside–beginning tagging format. "
                    #        "Please list them in the following format.")
                    # abbreviate as just `BIO`
                    # ret = ("Suppose you are a news writer. "
                    #        "Please list 20 sentences or phrases from news stories, token by token, "
                    #        "and annotate each token with the NER tag using the BIO tagging format. "
                    #        "Here are some examples.")
                    # `token` => `word`, doesn't seem to help
                    # ret = ("Suppose you are a news writer. "
                    #        "Please list 20 sentences or phrases from news stories, word followed by word, "
                    #        "and annotate each token with the NER tag using the BIO tagging format. "
                    #        "Here are some examples.")
                    # instruct LLM to add a tag for trailing period
                    # ret = ("Suppose you are a news writer. "
                    #        "Please list 20 sentences or phrases from news stories, token by token, "
                    #        "and annotate each token, including punctuations, with the NER tag using the BIO tagging format. "
                    #        "Here are some examples.")
                    # ret = ("Suppose you are a news writer. Please list 20 sentences or phrases from news stories, token by token, "
                    #        "and annotate each token, including the trailing punctuation, with the NER tag using the BIO tagging format. "
                    #        "Here are some examples.")
                    ret = ("Suppose you are a news writer. "
                           f"Please list {n_list} sentences or phrases from news stories, token by token, "
                           "and annotate each token with the NER tag using the BIO tagging format. "
                           "Please escape double quotes inside each token. "
                           "Please annotate the NER tag for the ending punctuations too. ")
                elif self.sample_format == 'bio-list-v2':
                    ret = ("Suppose you are a news writer. "
                           f"Please list {n_list} sentences or phrases from news stories, token by token, "
                           "and annotate each token with the NER tag using the BIO tagging format. ")
                else:
                    assert self.sample_format == 'bio-line'
                    # ret = ("Suppose you are a news writer. Please list 20 sentences or phrases from news stories, token by token. "
                    #        "Please annotate each token with the named entity tag using the BIO tagging format. "
                    #        "Put each token-tag pair on a separate line, and add an empty line after each sentence. "
                    #        "Here are some examples.")
                    ret = (f"Suppose you are a news writer. Please list {n_list} sentences or phrases from news stories, token by token. "
                           "Please annotate each token with the named entity tag using the BIO tagging format. "
                           "Join the token and NER tag with a comma. "
                           "Put each token-tag pair on a separate line, and add an empty line after each sentence. ")
            elif self.dataset_name == 'conll2003-no-misc':
                assert self.sample_format == 'natural-pair-v2'
                gpt_suggest = False
                # gpt_suggest = True
                if gpt_suggest:
                    # insert_diverse = False
                    # insert_diverse = True

                    # ret = (f"Generate {n_list} short news article snippets that could be found in Reuters. "
                    #        "Identify named entities that can fall into one the following categories: "
                    #        "[person, location, organization].\n")
                    # if insert_diverse:
                    #     ret += ("Ensure the sentences are on different topics, from different regions, "
                    #             "and mentioning different entities. "
                    #             "Vary the complexity of the snippets. "
                    #             "Some should be simple and short, while others can be more detailed with multiple entities. "
                    #             "Ensure each snippet is unique and not repetitive. Avoid using the same entities frequently.\n")
                    # ret += ("After each snippet, list the identified named entities with their categories in square brackets. "
                    #         "Here's the format:\n\n"
                    #         "\"This is a news snippet.\"\n"
                    #         "Entity Names: [entity 1 (type of entity 1), ...]\n")  # additional newline for isolating the format
                    ret = ("Suppose you are a news writer. "
                           f"Please generate {n_list} synthetic sentences or phrases from news stories. "
                           "Identify all entity names occurred that belong to one of the following entity types:\n"
                           "[person, location, organization].\n"
                           "After each sentence, list such identified entities with their categories, "
                           "in the order of occurrence.\n"
                           "Here's the format:\n\n"
                           "\"This is a news sentence or phrase.\"\n"
                           "Entity Names: [entity 1 (type of entity 1), ...]\n\n"
                           "If no entity is found in the sentence, list 'None'.")
                else:
                    # ret = ("Suppose you are a news writer. "
                    #        f"Please generate {n_list} synthetic sentences or phrases from news stories. "
                    #        "Please identify all entity names occurred that belong to one of the following entity types:\n"
                    #        "[person, location, organization].\n")

                    # stress on **named** entities; add instruction for multiple-occurrence
                    # ret = ("Suppose you are a news writer. "
                    #        f"Please generate {n_list} synthetic sentences or phrases from news stories. "
                    #        "Please identify all named entities occurred that belong to one of the following entity types:\n"
                    #        "[person, location, organization].\n")

                    v = 0
                    # v = 1
                    # v = 2
                    # v = 3
                    if v == 0:
                        ret = ("Suppose you are a news writer. "
                               f"Please generate {n_list} synthetic sentences or phrases from news stories. "
                               "Please identify all named entities occurred that belong to one of the following entity types:\n"
                               "[person, location, organization].\n")
                    elif v in [1, 3]:
                        # after model change, try to fix locations not found in sentence
                        ret = ("Suppose you are a news writer. "
                               f"Please generate {n_list} synthetic sentences or phrases from news stories.\n")
                        if v == 1:
                            ret += ("After each sentence, please identify the named entities occurred "
                                    "that belong to one of the following entity types:\n")
                        else:
                            assert v == 3
                            ret += ("After each sentence, please identify specific named entities that exactly appeared and "
                                    "belong to one of the following entity types:\n")
                        ret += "[person, location, organization].\n"
                    else:
                        assert v == 2
                        # try to reduce entities not found as exact match edge case, stress on exact match
                        ret = ("Suppose you are a news writer. "
                               f"Please generate {n_list} synthetic sentences or phrases from news stories.\n"
                               "In the generated sentences, identify and label specific named entities "
                               "that belong to one of the following entity types:\n"
                               "[person, location, organization].\n\n")  # additional new line
                    if self.insert is not None:
                        ret += "\nPlease use the definitions below to identify the named entities.\n"
                        if self.insert == 'defn':
                            # defn = schemas.conll2003_no_misc_defn

                            # after demo update
                            defn = schemas.conll2003_no_misc_defn2
                        else:
                            assert self.insert == 'schema'
                            # defn = schemas.conll2003_no_misc_schema  # best-performing
                            # sch = schemas.conll2003_no_misc_schema2
                            # sch = schemas.conll2003_no_misc_schema3

                            # after demo update
                            defn = schemas.conll2003_no_misc_schema4
                            # defn = schemas.conll2003_no_misc_schema5  # another version; performance was worse
                        ret += f'{defn}\n\n---\n\n'
                    # ret += ("Please list such entities with the corresponding entity types on the following line, "
                    #         "in the order of occurrence.\n"
                    #         "If no entity is found in the sentence, list 'None'.")

                    if v in [0, 1, 3]:
                        ret += ("Please list such named entities with the corresponding entity types on the following line, "
                                "in the order of occurrence.\n"
                                # doesn't seem to work anyway
                                # "If a named entity occurs multiple times in a sentence, list it as many times as it occurs.\n"
                                # "If no entity is found in the sentence, list 'None'.")
                                # seems to work better after model change: `gpt-3.5-turbo-0613` => `gpt-3.5-turbo-1106`
                                "If no entity is found in the generated sentence, leave the brackets empty.")
                    else:
                        assert v == 2
                        # try to reduce entities not found as exact match edge case, stress on exact match
                        ret += ("After each generated sentence, please list the identified the named entities in the order of occurrence "
                                "exactly as they appear in the sentence.\n"
                                "After each named entity listed, label the corresponding entity type in parenthesis.\n"
                                "If no entity is found in the generated sentence, leave the brackets empty.\n")
            elif self.dataset_name == 'job-desc':
                if self.sample_format == 'natural-pair-v2':
                    ret = ("Suppose you are a human resource staff/recruiter. "
                           f"Please generate {n_list} synthetic sentences from job descriptions on a recruitment website. "
                           "Please identify all entities occurred in the sentences that belong to one of the entity types below: \n"
                           "[Skill, Qualification, Experience, Domain, Occupation]. \n")
                    schema_desc = False
                    # schema_desc = True
                    if schema_desc:
                        ret += "\nPlease use our definitions, described below, to identify the entities. \n"
                        ret += f'{schemas.job_desc_schema}\n'

                    ret += ("Please list such entities with their corresponding entity types on the following line. "
                            "If no entity is found in the sentence, list 'None'. ")
                else:
                    raise NotImplementedError
            elif self.dataset_name == 'mit-movie':
                assert self.sample_format == 'natural-pair-v2'
                # ret = ('Suppose you are the user of a dialog system or conversational agent. '
                # ret = ('Suppose you are a user of a dialog system or conversational agent. '
                #        # f'Please generate {n_list} queries related to movies. '
                #        # try to emphasize `spoken`
                #        # f'Please generate {n_list} queries related to movies as if you are speaking to the agent. '
                #        # emphasize trial 2
                #        f'Please generate {n_list} spoken queries related to movies to the dialog system. '
                #        'Each query should inquire about one or more aspects of movies or related entities. '
                #        'On a newline, identify and tag the entities within the queries using the format: '
                #        # many entities annotated are out of order, try to enforce ordering => made things worse
                #        # 'On a newline, identify and tag the entities within the queries in the order of occurrence using the format: '
                #        '[entity 1 (type of entity 1), ...].\n'
                #        'Ensure to include one or more of the following entity types in the queries:\n'
                #        "[Title, Viewers' Rating, Year, Genre, Director, MPAA Rating, Plot, Actor, Trailer, Song, Review, Character]")

                # use a consistent template as `conll2003`
                # return ("Suppose you are a user of a dialog system or conversational agent. "
                #         f"Please generate {n_list} synthetic spoken queries related to movies to the dialog system. "
                #         f"Please identify all keywords occurred that belong to one of the following categories:\n"
                #         f"[Title, Viewers' Rating, Year, Genre, Director, MPAA Rating, Plot, Actor, Trailer, Song, Review, Character].\n"
                #         f"Please list such keywords with the corresponding categories on the following line, in the order of occurrence.")

                # switch back to `Named Entities`
                ret = ("Suppose you are a user of a dialog system or conversational agent. "
                       f"Please generate {n_list} synthetic spoken queries related to movies to the dialog system. "
                       "Please identify all named entities occurred that belong to one of the following entity types:\n")
                # "[Title, Viewers' Rating, Year, Genre, Director, MPAA Rating, Plot, Actor, Trailer, Song, Review, Character].\n"
                ets = sconfig(f'datasets.{self.dataset_name}.readable-entity-types')
                # random.shuffle(ets)
                ets = ', '.join(ets)
                ret += f'[{ets}].\n'
                ret += ("Please list such named entities with the corresponding entity types on the following line, "
                        "in the order of occurrence.\n")
            elif self.dataset_name == 'mit-restaurant':
                ret = ("Suppose you are a user of a dialog system or conversational agent. "
                       f"Please generate {n_list} synthetic spoken queries related to restaurants to the dialog system. "
                       "Please identify all named entities occurred that belong to one of the following entity types:\n")
                ets = sconfig(f'datasets.{self.dataset_name}.readable-entity-types')
                # random.shuffle(ets)
                if self.lowercase_entity_type:
                    ets = [et.lower() for et in ets]
                ets = ', '.join(ets)
                ret += f'[{ets}].\n'
                ret += ("Please list such named entities with the corresponding entity types on the following line, "
                        "in the order of occurrence.\n")
            elif self.dataset_name == 'job-stack':
                # consistent template w/ `conll2003`
                # return ("Suppose you are a human resource recruiter. "
                #         f"Please generate {n_list} synthetic sentences from job postings on StackOverflow. "
                #         "Please identify all named entities occurred that belong to one of the following entity types:\n"
                #         "[Organization, Location, Profession, Contact, Name].\n"
                #         "Please list such named entities with their corresponding entity types on the following line, "
                #         "in the order of occurrence.\n"
                #         "If no entity is found in the generated sentence, leave the brackets empty.")

                # drop `human resource`
                return ("Suppose you are a recruiter. "
                        f"Please generate {n_list} synthetic sentences from job postings on StackOverflow. "
                        "Please identify all named entities occurred that belong to one of the following entity types:\n"
                        "[Organization, Location, Profession, Contact, Name].\n"
                        "Please list such named entities with their corresponding entity types on the following line, "
                        "in the order of occurrence.\n"
                        "If no entity is found in the generated sentence, leave the brackets empty.")
            elif self.dataset_name == 'wiki-gold-no-misc':
                #  Suppose you are a Wikipedia editor. Please generate 20 synthetic sentences from Wikipedia articles.
                ret = ("Suppose you are a Wikipedia editor. "
                       f"Please generate {n_list} synthetic sentences from Wikipedia articles. "
                       "Please identify all named entities occurred that belong to one of the following entity types:\n")

                # insert `after each sentence`, doesn't help
                # ret = ("Suppose you are a Wikipedia editor. "
                #        f"Please generate {n_list} synthetic sentences from Wikipedia articles. "
                #        "After each sentence, please identify all named entities occurred that "
                #        "belong to one of the following entity types:\n")

                ets = sconfig(f'datasets.{self.dataset_name}.readable-entity-types')
                # random.shuffle(ets)  # doesn't seem to help
                ets = ', '.join(ets)
                ret += f'[{ets}].\n'

                ret += ("Please list such named entities with their corresponding entity types on the following line, "
                        "in the order of occurrence.\n"
                        "If no entity is found in the generated sentence, leave the brackets empty.")
            elif self.dataset_name == 'ncbi-disease':
                ret = ("Suppose you are a researcher. "
                       f"Please generate {n_list} synthetic sentences from PubMed paper abstracts. "
                       "Please identify all named entities occurred that belong to the entity type Disease.\n"
                       "Please list such named entities with the corresponding entity type (disease) on the following line, "
                       "in the order of occurrence.\n"
                       "If no entity is found in the generated sentence, leave the brackets empty.")
            else:
                raise NotImplementedError
        return ret

    def _get_demo_prefix(self, n_demo: int = None):
        if self.as_passage:
            if self.dataset_name == 'conll2003':
                if self.sample_format == 'natural-pair-v2':
                    return "Here is an example. Please follow this format."
                else:
                    raise NotImplementedError
        else:  # standard, one sentence at a time
            if self.dataset_name == 'conll2003':
                if n_demo == 1 and self.sample_format != 'natural-pair-v2':
                    raise NotImplementedError
                if self.sample_format == 'natural-pair':
                    return "Please list them in the following format:"
                elif self.sample_format == 'natural-pair-v2':
                    if n_demo == 1:
                        return "Here is an example. Please follow this format."
                    else:
                        return "Here are some examples. Please follow this format."
                elif self.sample_format == 'natural-inline':
                    return "Here are some examples. Please list them in the following format."
                elif self.sample_format == 'natural-inline-v2':
                    return "Here are some examples. Please follow this format."
                elif self.sample_format == 'bio-list':
                    return "Here are some examples."
                elif self.sample_format == 'bio-list-v2':
                    return "Here are some examples. Please list them in this format."
                else:
                    assert self.sample_format == 'bio-line'
                    return "Here are some examples."
            elif self.dataset_name == 'conll2003-no-misc':
                assert self.sample_format == 'natural-pair-v2'
                # return "Here are some examples. Please follow this format.\n\nExamples:"
                # return "Here are some examples:"

                # taking inspiration from 2-stage prompt
                # return ("Here are some example sentences from news articles and annotations for your reference. "
                #         "Please follow this format.\n\nExamples:")
                return "Here are some example sentences and annotations for your reference. Please follow this format.\n\nExamples:"
            elif self.dataset_name == 'job-desc':
                if self.sample_format == 'natural-pair-v2':
                    return "Here are some examples. Please follow this format."
                else:
                    raise NotImplementedError
            elif self.dataset_name in ['mit-movie', 'mit-restaurant']:
                assert self.sample_format == 'natural-pair-v2'
                # return "Here are some examples. Please follow this format.\n\nExamples:"

                # use a consistent template as `conll2003`
                return "Here are some example queries and annotations for your reference. Please follow this format.\n\nExamples:"
            elif self.dataset_name in ['job-stack', 'wiki-gold-no-misc', 'ncbi-disease']:
                assert self.sample_format == 'natural-pair-v2'
                return "Here are some example sentences and annotations for your reference. Please follow this format.\n\nExamples:"
            else:
                raise NotImplementedError


class DataGenerator:
    def __init__(
            self, dataset_name: str = 'conll2003', dataset_loader: DatasetLoader = None,
            diverse_context: Union[bool, Dict[str, Any]] = False, diverse_entity: Union[bool, str] = False,
            diversity_instr_at_end: bool = False, diversity_args: Dict[str, Any] = None,
            as_passage: bool = False,
            token_sep: str = ',', entity_sep: str = None, token_tag_sep: str = ',',
            entity_pair_map: EntityPairTemplate = None, token_map: TokenMapEnclose = None,
            sample_format: str = 'natural-pair', subsample_demo: bool = False, insert: str = None,
            lowercase_entity_type: bool = None
    ):
        """
        :param dataset_name: Dataset name to retrieve demo examples
        :param dataset_loader: Data loader
        :param diverse_context: If true, instruction specifies diverse attributes for generating contexts/sentences
        :param diverse_entity: If true, instruction specifies diverse attributes for generating entities
        :param diversity_instr_at_end: If true, instruction is inserted at the end of the prompt
        :param diversity_args: additional arguments for `DiversityRequirementConstructor`
        :param as_passage: If true, each sample contains multiple sentences from a single passage
        :param token_sep: token separator symbol, intended for `list-bio` format
        :param entity_sep: entity separator symbol, intended for [`natural`, `list-bio`] format
        :param token_tag_sep: token-tag separator symbol, intended for `line-bio` format
        :param entity_pair_map: for encoding & decoding entities
        :param sample_format: Prompt format, one of ['natural-pair', 'natural-inline', 'bio-list', 'bio-line']
        :param subsample_demo: If true, use a subset of the demo examples dynamically
        :param lowercase_entity_type: If true, lowercase entity types
        """
        ca(dataset_name=dataset_name, sample_format=sample_format)
        self.dataset_name = dataset_name
        self.sample_format = sample_format

        if dataset_loader:
            d_fmt = sample_fmt2original_data_fmt(sample_format)
            assert dataset_loader.data_format == d_fmt
        self._loader = dataset_loader

        self.diverse_context, self.diverse_entity = diverse_context, diverse_entity
        self.diverse_instr_at_end = diversity_instr_at_end
        self.drc, self.attr_prompt_args = None, diversity_args

        self.have_diversity = diverse_context or diverse_entity
        if self.have_diversity:
            args = dict(diverse_context=bool(diverse_context), diverse_entity=diverse_entity)
            if diverse_context:
                if isinstance(diverse_context, dict):
                    args.update(diverse_context)
            self.drc = DiversityRequirementConstructor(dataset_name=dataset_name, **args, **(diversity_args or dict()), logger=_logger)
        self.as_passage = as_passage
        self.insert = insert
        if lowercase_entity_type and dataset_name not in ['mit-restaurant']:
            raise NotImplementedError
        self.lowercase_entity_type = lowercase_entity_type or False
        self.ic = InstructionConstructor(
            dataset_name=dataset_name, sample_format=sample_format, as_passage=as_passage, insert=insert,
            lowercase_entity_type=lowercase_entity_type
        )

        self.token_sep = token_sep
        self.entity_sep = entity_sep or get_default_entity_sep(sample_format=sample_format)
        self.token_tag_sep = token_tag_sep

        self.entity_pair_map = entity_pair_map or get_default_entity_pair_map(sample_format=sample_format)
        self.token_map = token_map or get_default_token_map(sample_format=sample_format)

        self.subsample_demo = subsample_demo
        if subsample_demo and as_passage:
            raise ValueError(f'{pl.i("subsample_demo")} intended for single-sentence samples only, '
                             f'but {pl.i("as_passage")} = {pl.i(as_passage)}')

        self.logger = _logger

    @property
    def loader(self) -> DatasetLoader:
        if self._loader is None:
            d_fmt = sample_fmt2original_data_fmt(self.sample_format)
            self._loader = DatasetLoader(dataset_name=self.dataset_name, data_format=d_fmt)
        return self._loader

    def meta(self, n_list: int = None, postfix: str = None, **kwargs) -> str:
        abb = kwargs.pop('abbreviate', True)
        if isinstance(self.diverse_context, dict):
            dc_ = self.drc.meta(abbreviate=abb)
        else:
            assert isinstance(self.diverse_context, bool)
            dc_ = self.diverse_context
        return dataset_meta(
            sample_format=self.sample_format, n_list=n_list, diverse_context=dc_, diverse_entity=self.diverse_entity,
            as_passage=self.as_passage, postfix=postfix, **kwargs
        )

    def example2demo_str(self, sample: NerExample) -> str:
        if self.sample_format in ['natural-pair', 'natural-pair-v2']:
            assert isinstance(sample, NerReadableExample)
            sent, nms, types = sample.sentence, sample.entity_names, sample.entity_types
            assert len(nms) == len(types)
            if self.lowercase_entity_type:
                types = [t.lower() for t in types]

            if len(nms) == 0:
                # entities = 'None'
                entities = '[]'
            else:
                entities = f'{self.entity_sep} '.join(self.entity_pair_map(nm, tp) for nm, tp in zip(nms, types))
                entities = f'[{entities}]'
            if self.sample_format == 'natural-pair':
                return f'sentence: {sent}\nentities: {entities}.'
            else:  # `natural-pair-v2`
                # return f'"{sent}"\nEntity Names: [{entities}]'
                # pref = 'Entity Names'
                # pref = 'Named Entities'
                # return f'Sentence: {enclose_in_quote(sample.sentence)}\n{pref}: {entities}'
                d_dset = sconfig(f'datasets.{self.dataset_name}')
                pref_x, pref_y = d_dset['x-name'], d_dset['y-name']
                return f'{pref_x}: {edit.enclose_in_quote(sample.sentence)}\n{pref_y}: {entities}'
        elif self.sample_format in ['natural-inline', 'natural-inline-v2']:
            assert isinstance(sample, NerSpanExample)

            ret = ''
            for i, s in enumerate(sample.spans):
                cont = s.content

                # edge case: merge without space if current token is a `'s`?
                # e.g. NerSpan(content='Germany', entity_type='LOC'), NerSpan(content="'s representative to the", entity_type=None)
                sep = ' '
                if i == 0 or cont.startswith("'s") and len(ret) > 0:
                    sep = ''

                if s.entity_type is None:
                    ret = f'{ret}{sep}{s.content}'
                else:
                    ret = f'{ret}{sep}{self.entity_pair_map(s.content, s.entity_type)}'
            # sic(sample, ret)
            if self.sample_format == 'natural-inline':
                return f'sentence: {ret.strip()}'
            else:  # `natural-inline-v2`
                # return f'"{sample.sentence}"\nAnnotated: {ret.strip()}'
                # LLM doesn't seem to follow the `Sentence` prefix
                # return f'Sentence: "{sample.sentence}"\nAnnotated Sentence: "{ret.strip()}"'
                return f'"{sample.sentence}"\nAnnotated Sentence: "{ret.strip()}"'
        else:  # `list-bio` or `line-bio`
            assert isinstance(sample, NerBioExample)
            tokens, tags = sample.tokens, sample.ner_tags
            assert len(tokens) == len(tags)

            if 'bio-list' in self.sample_format:
                # tokens = f'{self.token_sep} '.join(f'`{t}`' for t in tokens)  # the [`] enclosing seems hard to follow?
                tokens = f'{self.token_sep} '.join(self.token_map(t) for t in tokens)
                entities = f'{self.entity_sep} '.join(tags)
                if self.sample_format == 'bio-list':
                    return f'tokens: {tokens}\nentity tags: {entities}'
                else:
                    assert self.sample_format == 'bio-list-v2'
                    sent = sample.sentence
                    return f'Sentence: "{sent}"\nTokens: [{tokens}]\nNER tags: [{entities}]'
            else:
                assert self.sample_format == 'bio-line'

                pairs = [f'{tok}{self.token_tag_sep} {tag}' for tok, tag in zip(tokens, tags)]
                pairs = '\n'.join(pairs)
                return f'sentence:\n{pairs}'

    def get_prompt(
            self, n_list: int = 20, n_demo: int = 5, demo_type: str = 'n-shot', demo_args: Dict[str, Any] = None,
            generator: Union[random.Random, int] = None
    ) -> str:
        """
        :param n_list: Number of samples LLM is asked to list
        :param n_demo: Number of samples in the demo
        :param demo_type: Scheme for getting demo, one of [`n-shot`, `n`]
        :param demo_args: Arguments for getting demo
        :param generator: Random number generator
        """
        gen = get_random_generator(generator=generator)

        subsample, n_demo_ = False, None
        if self.subsample_demo:
            n_demo_ = gen.randint(1, n_demo)
            if n_demo_ < n_demo:
                subsample = True

        diverse_instr, instr_args = None, dict()
        if self.have_diversity:
            diverse_instr = self.drc(generator=gen)  # random sampling may result in no instructions
        if diverse_instr and not self.diverse_instr_at_end:
            instr_args['context_requirement'] = diverse_instr
        ret = self.ic(n_list=n_list, n_demo=n_demo_, **instr_args)
        # shouldn't have access to the entire training set for few-shot setup

        samples: List[NerExample] = self.loader.get_few_demo_samples(n_demo=n_demo, demo_type=demo_type, shuffle=False, **(demo_args or dict()))

        if self.as_passage:
            if self.dataset_name == 'conll2003':
                # use a formatting for news articles
                sents = [eg.sentence for eg in samples]
                title, sents = sents[0], sents[1:]
                sep = ' '
                article = sents[0]
                for s in sents[1:]:
                    # if last sentence don't end in punctuation, start a new line
                    # Kinda ad-hoc cos I know the 2nd, 3rd samples are [`Peter Blackburn`, `BRUSSELS 1996-08-22`]
                    last_char = article[-1]
                    if last_char not in string.punctuation:
                        sep = '\n'
                    article = f'{article}{sep}{s}'

                ret = f'{ret}\n\n**Title**: {title}\n\n**Article**:\n{article}'
            else:
                raise NotImplementedError

        if subsample:
            samples = gen.sample(samples, n_demo_)
        gen.shuffle(samples)

        add_enum_prefix = False
        # add_enum_prefix = True  # this seems to result in worse f1 (~1%) on across 3 1-stage runs for CoNLL-03, w/ #list=3
        samples_str = [self.example2demo_str(s) for s in samples]
        if add_enum_prefix:
            samples_str = [f'{i}. {s}' for i, s in enumerate(samples_str, start=1)]
        samples: str = '\n\n'.join(samples_str)
        # separate the samples more from the instructions via additional newlines
        if self.as_passage:
            # ret = f'{ret}\n\n**Sentences**: \n{samples}'
            ret = f'{ret}\n\n**Sentences**: \n{samples}\n\n[End of Sentences]'  # for easy re parsing
        else:
            ret = f'{ret}\n\n{samples}'
        # if self.sample_format in ['natural-pair-v2', 'natural-inline', 'natural-inline-v2', 'bio-line']:
        #     ret = f'{ret}\n\n'  # add newline so that it starts generating a new sample
        ret = f'{ret}\n\n\n---'  # signal end of demo examples
        if diverse_instr and self.diverse_instr_at_end:
            # ret = f'{ret}\n\n{attr_instr}'
            ret = f'{ret}\n\n{diverse_instr}'  # separate examples w/ instructions
        # ret = f'{ret}Please generate more examples in this format.'
        return ret
        # return f'{ret}\n\n'  # ~~signals end of instructions~~ this made things worse

    def write_completions(
            self, n_prompt: int = 10, n_list: int = 50, output_dir_nm: str = None, prompt_args: Dict[str, Any] = None,
            generator: Union[random.Random, int] = None, **kwargs
    ):
        """
        :param n_prompt: #prompts, corresponding to #API calls to make
        :param n_list: # of demo examples to include in the prompt
        :param output_dir_nm: output directory name
        :param prompt_args: Arguments for getting prompt
        :param generator: Random number generator
        """
        output_dir = dataset_name2data_dir(dataset_name=self.dataset_name, output_dir='Sample-Res', output_postfix=output_dir_nm).path
        # run multiple times: each call shuffles the demo examples, see `PromptConstructor.__call__`
        # for diverse context, more randomization from sampling instructions & categories
        gen = get_random_generator(generator=generator)
        prompts = [self.get_prompt(n_list=n_list, generator=gen, **(prompt_args or dict())) for _ in range(n_prompt)]
        # sic(prompts[:5], len(prompts))
        # raise NotImplementedError

        args = dict(logger=_logger, init_log={'dataset-name': self.dataset_name, '#example requested': n_list})
        if self.have_diversity:
            args['log_callback'] = lambda log: self.drc.init_log(logger=self.logger)
        return completions.write_completions(prompts=prompts, completion_type='sample', output_path=output_dir, **args, **kwargs)


if __name__ == '__main__':
    pass
