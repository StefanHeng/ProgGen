"""
1st step in step-wise data generation: generate sentences

Similar functionalities of
    `src.generate_dataset.demo2prompt::PromptConstructor`,
    `src.generate_dataset.prompt2completion::CompletionsWriter`,
    `src.generate_dataset.completion2train_data::NerDatasetWriter`
for just sentence generation
"""

import re
import json
import random
from os.path import join as os_join
from typing import Dict, List, Union, Any, Optional

from stefutil import *
from src.util import *
from src.util.ner_example import *
from src.data_util import *
from src.generate import *
from src.generate.diversify import DiversityRequirementConstructor
from src.generate.step_wise.util import *


__all__ = ['SentenceGenerator']


_logger = get_logger('Gen X')


class SentenceGenerator(StepWiseGenerator):
    def __init__(
            self, attr_prompt_args: Union[bool, Dict] = None, attr_prompt_instr_at_end: Optional[bool] = None, as_passage: bool = False,
            subsample_demo: bool = False, drop_with_inline_type: bool = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.completion_type = 'sentence'

        d_c, d_e = None, None
        if attr_prompt_args:
            d_c, d_e = attr_prompt_args.get('diverse_context', None), attr_prompt_args.get('diverse_entity', None)
        if d_c or d_e:
            self.attr_prompt = True
            if isinstance(attr_prompt_args, bool):
                self.crc = DiversityRequirementConstructor(dataset_name=self.dataset_name)
            else:
                assert isinstance(attr_prompt_args, dict)
                self.crc = DiversityRequirementConstructor(dataset_name=self.dataset_name, **attr_prompt_args)
            self.diverse_context, self.diverse_entity = self.crc.diverse_context, self.crc.diverse_entity
        else:
            self.attr_prompt, self.crc = False, None
            self.diverse_context, self.diverse_entity = None, None
        self.diverse = self.diverse_context or self.diverse_entity
        self.attr_prompt_instr_at_end = attr_prompt_instr_at_end or True
        self.as_passage = as_passage
        self.subsample_demo = subsample_demo

        self.entity_types = sconfig(f'datasets.{self.dataset_name}.readable-entity-types').copy()

        if self.as_passage:
            if self.dataset_name == 'conll2003-no-misc':
                # news articles generated tend to contain a title, extract it separately
                # e.g. Title: "Revolutionary Electric Car Sets New Speed Record"
                self.pattern_title = re.compile(r'^(Title|Headline|Breaking News|BREAKING|Opinion): (")?(?P<title>.+)(")?', re.IGNORECASE)

            # LLM generations contains other stuff, remove them
            # e.g. Disclaimer: This news article is purely fictional and does not represent real events or entities.
            # e.g. (person: Peter Blackburn, location: BRUSSELS, organization: European Union)
            # e.g. ---
            # e.g. 1. United Nations Headquarters, New York City
            #   TODO: on 2nd thought, these can be considered valid sentences? at least entity type not present
            # e.g. 1. Cupertino, California (location) // drop if below 30 characters
            self.pattern_skip = [
                re.compile(r'^(\()?(Disclaimer|Note):.*$', re.IGNORECASE),
                re.compile(r'^[(\[]?(- )?(person|location|organization):.*[)\]]?$', re.IGNORECASE),
                re.compile(r'---'),
                re.compile(r'^\d+\.$'),
                # re.compile(r'^\d+\. .{0,50}$'),
                re.compile(r'^(\d+\. )?.{0,30} \((person|location|organization|date|product)\)$', re.IGNORECASE),
                re.compile(r'^(BREAKING NEWS|Named Entities|Sources):$', re.IGNORECASE),
            ]

            # LLM generated sentences have some entities bolded since the instructions asked to cover entity names,
            #   drop these boldface chars
            # e.g. **New York City**, known for its ... => New York City, known for its ...
            self.pattern_bold = re.compile(r'\*\*(?P<entity>.+?)\*\*')
        else:
            # edge case: sentences have entity type prefix, e.g.
            #   2. Organization: "Apple announces new iPhone release date."
            #   2. ORGANIZATION - Google announces plans to launch a new satellite internet service.
            #   "Location: Los Angeles - The annual music festival in Coachella Valley attracts thousands of attendees."
            #   [Location] The Great Barrier Reef experiences a 50% decline in coral coverage due to rising sea temperatures.
            #   "Location Los Angeles to host the 2023 Grammy Awards at the Staples Center."
            # edge case: sentences have entity type postfix, e.g.
            #   2. "The humanitarian organization Red Cross announced today that they are mobilizing resources to provide aid to the victims of the devastating earthquake in Indonesia." - organization
            # edge case: sentences have entity name prefix, e.g.
            #   6. (NASA) NASA's Perseverance rover successfully landed on Mars, marking a major milestone in space exploration.
            # edge case, trailing entity annotations, e.g.
            #   "Notorious local gang leader arrested in police raid on east side neighborhood." [person, location]
            #   `"Our team at CyberSafe Ltd in London is seeking a talented Cybersecurity Engineer to design and implement security solutions to protect the organization from cyber threats." [Organization: CyberSafe Ltd, Location: London, Profession: Cybersecurity]`
            # edge case: annotations on unseen entity types, e.g.
            #   1. "Tropical Storm Karen expected to hit the Gulf Coast by the end of the week." (location, Weather)
            ets = self.entity_types.copy()
            if self.dataset_name == 'conll2003-no-misc':
                ets += ['Weather', 'event']
            elif self.dataset_name == 'mit-movie':
                ets += ['Movie Showtimes', 'Ticket Availability', 'Inheritance', 'Motive', 'Respect', 'E', 'Judgement']
            elif self.dataset_name == 'mit-restaurant':
                ets += ['Meal Category', 'Working hours']
            elif self.dataset_name == 'job-stack':
                ets += ['Connection']
            d_dset = sconfig(f'datasets.{self.dataset_name}')
            pref_x, pref_y = d_dset['x-decoded-name'], d_dset['y-decoded-name']
            pref_x_ = d_dset['x-name']

            pattern_ets = options2re_options(options=ets)
            # match e.g. `[Location, Organization]`, `(person, organization)`, `(Amenity and Cuisine)`
            pattern_et_lst = rf'[\[(]{pattern_ets}((,| and) {pattern_ets})*[)\]]'
            # match e.g. `[person, location, organization]`
            pattern_et_lst2 = rf'\[{pattern_ets}( {pattern_ets})*\]'
            # match e.g. `[person] [location] [organization]`
            pattern_et_lst3 = rf'\[{pattern_ets}\]( \[{pattern_ets}\])*'
            # match a trailing tag in brackets
            #   e.g. 3. "I'm looking for a movie directed by a female filmmaker that falls under the science fiction genre." [Discovering hidden talent]
            pattern_tag1 = r'\[.{1,30}\]'
            pattern_tag2 = r'\(.{1,30}\)'
            # e.g. `1. `, `Query 1: `
            pattern_enum_prefix = [rf'((?P<idx>\d+)\. )', rf'({pref_x_} (\d+): )', r'\s*-\s*']
            pattern_enum_prefix = options2re_options(options=pattern_enum_prefix)

            # match annotation row of only type-name pairs
            #   e.g. `   - [Organization: Nature's Beauty, Location: San Francisco]`
            #   e.g. `(person: Maria Ramirez, location: Los Angeles, organization: food truck)`
            #   e.g. `   - [Organization: Acme Inc.], [Location: Dublin], [Profession: Senior Front-End Developer]`
            pattern_e_pair2 = rf'{pattern_ets}: .{{1,30}}'
            pattern_e_pair3 = rf'\[{pattern_ets}: .{{1,30}}]'
            pattern_e_pair4 = rf'\[{pattern_ets}:( )?.{{1,30}}]'
            pattern_e_pair_lst = rf'[\[\(]({pattern_e_pair2})(, ({pattern_e_pair2}))*[])]'
            # e.g. `[Organization: Acme Corporation] [Location: anywhere] [Connection: remotely]`
            pattern_e_pair_lst2 = rf'{pattern_e_pair4}( {pattern_e_pair4})*'

            # needs 2 versions, w/ and w/o double quotes, cos need to drop both (start & end) at the same time, for e.g.
            #   Actress Emma Stone is set to star in a new film adaptation of the classic novel "Little Women."
            self.pattern_sent = [
                re.compile(rf'^{pattern_enum_prefix}?"(?P<et>{pattern_ets}): (?P<sent>.*?)"$', re.IGNORECASE),
                re.compile(rf'^{pattern_enum_prefix}?(?P<et>{pattern_ets}): "(?P<sent>.*?)"$', re.IGNORECASE),
                re.compile(rf'^{pattern_enum_prefix}?(?P<et>{pattern_ets}): (?P<sent>.*?)$', re.IGNORECASE),

                re.compile(rf'^{pattern_enum_prefix}?"(?P<sent>.*?)" - {pattern_ets}$', re.IGNORECASE),
                re.compile(rf'^{pattern_enum_prefix}(?P<sent>.*?) - {pattern_ets}$', re.IGNORECASE),

                re.compile(rf'^{pattern_enum_prefix}?"?(?P<et>{pattern_ets})( )?- "(?P<sent>.*?)"$', re.IGNORECASE),
                re.compile(rf'^{pattern_enum_prefix}?"?(?P<et>"{pattern_ets})( )?- (?P<sent>.*?)"$', re.IGNORECASE),
                re.compile(rf'^{pattern_enum_prefix}?"?(?P<et>{pattern_ets})( )?- (?P<sent>.*?)$', re.IGNORECASE),

                re.compile(rf'^{pattern_enum_prefix}?{pattern_et_lst} "(?P<sent>.*)"$', re.IGNORECASE),
                re.compile(rf'^{pattern_enum_prefix}?{pattern_et_lst} (?P<sent>.*)$', re.IGNORECASE),

                # for edge case: e.g.
                #   "I'm craving Italian food, can you recommend a highly-rated Italian restaurant in the downtown area [Cuisine, Location, Rating] [Restaurant Name]?"
                re.compile(rf'^{pattern_enum_prefix}?"(?P<sent>.*?)( )?{pattern_et_lst}( {pattern_tag1})?[?.]?"$', re.IGNORECASE),

                re.compile(rf'^{pattern_enum_prefix}?"(?P<sent>.*)" {pattern_tag1}$', re.IGNORECASE),
                re.compile(rf'^{pattern_enum_prefix}?"(?P<sent>.*?) {pattern_et_lst3}[?.]?"$', re.IGNORECASE),
                re.compile(rf'^{pattern_enum_prefix}(?P<sent>.*) {pattern_tag1}$', re.IGNORECASE),

                re.compile(rf'^{pattern_enum_prefix}?{pattern_tag2} "(?P<sent>.*)"$', re.IGNORECASE),
                re.compile(rf'^{pattern_enum_prefix}?{pattern_tag2} (?P<sent>.*)$', re.IGNORECASE),

                # match the smallest sentence
                re.compile(rf'^{pattern_enum_prefix}?"(?P<sent>.*?)"( )?{pattern_et_lst}\.?$', re.IGNORECASE),
                re.compile(rf'^{pattern_enum_prefix}?(?P<sent>.*?)( )?{pattern_et_lst}\.?$', re.IGNORECASE),

                re.compile(rf'^{pattern_enum_prefix}?"(?P<sent>.*?)" {pattern_e_pair_lst}$', re.IGNORECASE),
                re.compile(rf'^{pattern_enum_prefix}?(?P<sent>.*?) {pattern_e_pair_lst}$', re.IGNORECASE),

                re.compile(rf'^{pattern_enum_prefix}?"(?P<sent>.*?)"$', re.IGNORECASE),
                re.compile(rf'^{pattern_enum_prefix}(?P<sent>.*?)$', re.IGNORECASE),

                re.compile(r'^"(?P<sent>.+?)"$', re.IGNORECASE),  # no enum prefix, at least 1 char
                re.compile(r'^(?P<sent>.+?)$', re.IGNORECASE),
            ]
            self.pattern_sent = re_pattern2regex_pattern(pattern=self.pattern_sent)  # 20X faster since too many patterns

            # edge case: assistant prefix, e.g.
            #   Here are the 3 synthetic sentences for news stories:
            # edge case: entity type-name pair annotations before sentence, e.g.
            #   2. Location: New York City – Organization: The United Nations You announced that Meghan Markle will be a guest speaker at the UN Climate Action Summit.
            # edge case: single entity type-name pair, inside sentence (instead of at start of sentence), e.g.
            #   "Two suspects arrested in connection with the robbery at a jewelry store in LOCATION - Los Angeles."
            # match 2 or more type-name pairs where name is below 30 chars
            #   e.g. `Location: New York City – Organization: The United Nations`, `Person: Prince Harry - Location: Canada - Organization: Royal Family`
            pattern_e_pair = rf'{pattern_ets} [-–] [^?]{{1,30}}'
            pattern_e_pairs = rf'{pattern_ets}: .{{1,30}} [-–] {pattern_ets}: .{{1,30}}'

            pref_x, pref_y = options2re_options(options=pref_x), options2re_options(options=pref_y)
            self.pattern_skip = [
                re.compile(rf'^.*(synthetic|generated|example).*{pref_x}.*$', re.IGNORECASE),
                re.compile(rf'^.*{pref_x}.*synthetic.*$', re.IGNORECASE),
                re.compile(rf'^.*{pref_x}.*{pref_y}.*$', re.IGNORECASE),
                re.compile(rf'^.*here.*are.*{pref_x}.*$', re.IGNORECASE),
                re.compile(r'^Examples:$', re.IGNORECASE),
                re.compile(r'^(Spoken )?Queries:$', re.IGNORECASE),
                re.compile(rf'^"?$', re.IGNORECASE),  # empty string or just a double quote
                re.compile(rf'^{pattern_tag2}$', re.IGNORECASE),
                re.compile(rf'^"?{pattern_ets}"$', re.IGNORECASE),

                re.compile(rf'^\s*-?\s*{pattern_e_pair_lst2}\s*$', re.IGNORECASE),
                re.compile(rf'^\s*{pattern_et_lst3}\s*$', re.IGNORECASE),
                re.compile(rf'^\s*{pattern_et_lst2}\s*$', re.IGNORECASE),
                re.compile(rf'^(\d+\. )?{pattern_e_pairs}.*$', re.IGNORECASE),
                re.compile(rf'^(\d+\. )?(.*)([a-zA-Z]+)(.*){pattern_e_pair}$', re.IGNORECASE),
                re.compile(rf'^{pattern_et_lst}$', re.IGNORECASE),
                re.compile(rf'^\s*-?\s*{pattern_e_pair_lst}$', re.IGNORECASE),
                re.compile(rf'^\s*-?\s*({pattern_e_pair3})(, ({pattern_e_pair3}))*$', re.IGNORECASE),

                re.compile(rf'^(?P<idx>\d+)\.( )?$', re.IGNORECASE)
            ]
            self.pattern_skip = re_pattern2regex_pattern(pattern=self.pattern_skip)

            # named entities are highlighted in brackets
            #   e.g. [Lionel Messi] wins the prestigious ...
            #   e.g. Person [Lionel Messi] wins the prestigious ...
            # consider anything below 30 chars as named entities; exclude the closing `]`
            # named entities are also highlighted w/ parentheses, match substrings that is not all-caps
            #   e.g. 4. (Apple) reported record-breaking sales for its latest (iPhone) model.
            # edge case: missing whitespace before emphasized term, e.g.
            #   `1. "Can you recommend a Title[R-25] with a high Viewers' Rating[R-25] and directed by a female Director[R-25]?"`
            self.pattern_emph = [
                # initial sos/whitespace/boundary ensures matching only a word as the entity type,
                #   not matching the trailing `e` in `movies` in `movie [Breakfast at Tiffany's]`
                #   this is needed cos `E` is a generated entity type for `mit-movie`
                # re.compile(rf'((^| ){pattern_ets} )?\[(?P<et>[^]]{{1,45}})]', re.IGNORECASE),
                re.compile(rf'(\b{pattern_ets} )?\[(?P<et>[^]]{{1,45}})]', re.IGNORECASE),
                re.compile(r'\((?P<et>(?![A-Z0-9\s]+\))[^)]+)\)')
            ]
            # self.pattern_emph = re_pattern2regex_pattern(pattern=self.pattern_emph)
            re.compile(rf'({pattern_ets} )?[\[(](?P<et>[^](]{{1,45}})[])]', re.IGNORECASE)
            # named entities are annotated inline, drop these
            #   e.g. Flooding in [location] Bangladesh displaces thousands of residents, prompting humanitarian aid efforts.
            ets_annot, rmv = ets.copy(), None
            if self.dataset_name == 'mit-movie':
                rmv = ["Viewers' Rating", 'MPAA Rating', 'Genre', 'Trailer', 'Year', 'Plot', 'Review']
            if rmv:  # ignore classes in filtering out inline annotations
                for et in rmv:
                    ets_annot.remove(et)
            pattern_ets_annot = options2re_options(options=ets_annot)
            self.pattern_inline_annot = re.compile(rf'([\[(])(?P<et>{pattern_ets_annot})[])]', re.IGNORECASE)

            if drop_with_inline_type is None:
                drop_with_inline_type = True if self.dataset_name in ['mit-movie'] else False
            self.drop_with_inline_type = drop_with_inline_type
        self.logger = _logger
        self.ec = None

    def meta(self, n_list: int = None, postfix: str = None, **kwargs) -> str:
        return dataset_meta(
            sample_format=self.sample_format, n_list=n_list, diverse_context=self.diverse_context, diverse_entity=self.diverse_entity,
            as_passage=self.as_passage, postfix=postfix, **kwargs
        )

    def get_instruction(self, n_list: int = 20):
        assert self.sample_format == 'natural-pair-v2'
        if self.dataset_name == 'conll2003-no-misc':
            ret = 'Suppose you are a news writer.'
            if self.as_passage:
                # return f'{ret} Please generate a synthetic news story/article with around {n_list} sentences.'
                ret += (f"Please generate a synthetic news article with around {n_list} sentences. "
                        "In your generated news story, ")
            else:
                # ret = f'{ret} Please generate {n_list} synthetic sentences or phrases from news stories.'
                # drop `or phrases`
                ret = (f'{ret} Please generate {n_list} synthetic sentences from news stories. '
                       f'In your generated sentences, ')
            ret += ("try to cover diverse named entities that belong to the following entity types:\n"
                    "[person, location, organization].")
            # drop `try to`
            # ret += ("cover diverse named entities that belong to the following entity types:\n"
            #         "[person, location, organization].")
            return ret
        elif self.dataset_name == 'wiki-gold-no-misc':
            return ('Suppose you are a Wikipedia editor. '
                    f'Please generate {n_list} synthetic sentences from Wikipedia articles. '
                    'In your generated sentences, cover diverse named entities that belong to the following entity types:\n'
                    '[person, location, organization].')
        elif self.dataset_name in ['mit-movie', 'mit-restaurant']:
            assert not self.as_passage
            kd = 'movies' if self.dataset_name == 'mit-movie' else 'restaurants'
            # ret = ('Suppose you are a user of a dialog system or conversational agent. '
            #        f'Please generate {n_list} synthetic spoken queries related to {kd} to the dialog system. '
            #        'In your generated queries, try to cover diverse named entities that belong to the following entity types:\n')

            # drop `try to`
            ret = ('Suppose you are a user of a dialog system or conversational agent. '
                   f'Please generate {n_list} synthetic spoken queries related to {kd} to the dialog system.')
            include_ets = True
            # include_ets = False
            if include_ets:
                # ret = f'{ret} In your generated queries, cover diverse named entities that belong to the following entity types:\n'
                # add back `try to`
                ret = f'{ret} In your generated queries, try to cover diverse named entities that belong to the following entity types:\n'
                ets = self.entity_types.copy()
                ret += pl.nc(ets)
            return ret
        elif self.dataset_name == 'job-stack':
            assert not self.as_passage
            # ret = ('Suppose you are a recruiter.'
            #        f'Please generate {n_list} synthetic sentences from job postings on StackOverflow. '
            #        'In your generated sentences, cover diverse named entities that belong to the following entity types:\n')

            # add `try to`
            ret = ('Suppose you are a recruiter. '
                   f'Please generate {n_list} synthetic sentences from job postings on StackOverflow. '
                   'In your generated sentences, try to cover diverse named entities that belong to the following entity types:\n')
            ets = self.entity_types.copy()
            ret += pl.nc(ets)
            return ret
        else:
            raise NotImplementedError

    @staticmethod
    def example2demo_str(sample: NerExample) -> str:
        return enclose_in_quote(sample.sentence)

    def get_demo_instruction(self, n_demo: int = None):
        assert n_demo is not None and n_demo > 0
        if self.as_passage:  # even though model asked to write complete articles, we can only provide sentences
            if self.dataset_name != 'conll2003-no-misc':
                raise NotImplementedError
            if n_demo == 1:
                return 'Here is an example sentence from a news article for your reference.\n\nExample:'
            else:
                # return 'Here are some example sentences from news articles.\n\nExamples:'
                return 'Here are some example sentences from news articles for your reference.\n\nExamples:'
        else:
            assert n_demo != 1
            if self.dataset_name == 'conll2003-no-misc':
                return 'Here are some example sentences from news articles for your reference.\n\nExamples:'
            elif self.dataset_name == 'wiki-gold-no-misc':
                return 'Here are some example sentences from Wikipedia articles for your reference.\n\nExamples:'
            elif self.dataset_name in ['mit-movie', 'mit-restaurant']:
                kd = 'movies' if self.dataset_name == 'mit-movie' else 'restaurants'
                return f'Here are some example queries related to {kd} for your reference.\n\nExamples:'
            elif self.dataset_name == 'job-stack':
                return f'Here are some example sentences from job postings on StackOverflow for your reference.\n\nExamples:'
            else:
                raise NotImplementedError

    def get_prompt(
            self, n_list: int = 20, n_demo: int = None, demo_type: str = 'n-shot', demo_args: Dict[str, Any] = None,
            with_unlabeled: Optional[Union[bool, int]] = False
    ) -> str:
        ret = self.get_instruction(n_list=n_list)
        attr_prompt_instr = None
        if self.attr_prompt:
            prefix = None
            # if self.dataset_name == 'conll2003-no-misc':
            #     if self.as_passage:
            #         prefix = 'For your generated news story, '
            #     else:
            #         prefix = 'For your generated sentences, '
            # else:
            #     raise NotImplementedError
            attr_prompt_instr = self.crc(prefix=prefix)  # may be empty at probability
            if attr_prompt_instr and not self.attr_prompt_instr_at_end:
                ret = f'{ret}\n\n{attr_prompt_instr}'

        # zero-shot seems more diverse
        if n_demo is not None:
            # demo seems to restrict the diversity of generations, e.g. providing sentences => generate only news headlines

            demo_args_ = dict(n_demo=n_demo, demo_type=demo_type, shuffle=True, **(demo_args or dict()))
            if with_unlabeled:
                n = with_unlabeled if isinstance(with_unlabeled, int) else 100
                out = self.loader.get_few_demo_and_n_samples(n_shot_args=demo_args_, n_args=dict(n=n, shuffle=True))
                samples = out.n_shot_samples + out.n_samples
                samples = sample_few(lst=samples, min_=3, max_=7)
            else:
                samples = self.loader.get_few_demo_samples(**demo_args_)

            random.shuffle(samples)
            if self.subsample_demo:
                samples = sub_sample_demo(samples=samples, scheme='bernoulli')

            if len(samples) > 0:  # it may happen that there are no samples if subsample demo
                ret = f'{ret}\n\n{self.get_demo_instruction(n_demo=len(samples))}'
                samples = '\n\n'.join([self.example2demo_str(sample) for sample in samples])
                ret = f'{ret}\n\n{samples}'

                ret = f'{ret}\n\n\n---'  # signal end of demo examples
        if self.attr_prompt and attr_prompt_instr and self.attr_prompt_instr_at_end:
            ret = f'{ret}\n\n{attr_prompt_instr}'
        return ret

    def write_completions(
            self, n_prompt: int = 10, n_list: int = 50, n_demo: int = None, output_dir_nm: str = None,
            get_prompt_args: Dict[str, Any] = None, **kwargs
    ):
        out_dir = f'{self.completion_type.capitalize()}-Res'
        output_dir = dataset_name2data_dir(**self.dir_args, output_dir=out_dir, output_postfix=output_dir_nm).path
        prompts = [self.get_prompt(n_list=n_list, n_demo=n_demo, **(get_prompt_args or dict())) for _ in range(n_prompt)]

        debug_f1_drop = False
        # debug_f1_drop = True
        if debug_f1_drop:
            dir_nm = '23-11-25_{fmt=n-p2,de=T}_completions/23-11-26_00-56-42_Sentence-Res_{fmt=n-p2,#l=3,de=T}'
            with open(os_join(pu.generated_data_path, 'mit_movie', STEP_WISE_DNM, dir_nm, 'prompts.json'), 'r') as f:
                prompts = json.load(f)['prompts']
            # sic(prompts[:10])
            # raise NotImplementedError

        args = dict(logger=self.logger, init_log={'dataset-name': self.dataset_name, '#example requested': n_list})
        if self.diverse:
            args['log_callback'] = lambda log: self.crc.init_log(logger=self.logger)
        write_completions(
            output_path=output_dir, completion_type=self.completion_type, prompts=prompts, **args, **kwargs
        )

    def _filter_sentences(self, sentences: List[str]) -> List[str]:
        if self.pattern_skip:
            ret = []
            for sent in sentences:
                if match(text=sent.strip(), pattern=self.pattern_skip, verbose=True):  # drop sentences that match skip patterns
                    self.ec(msg=f'Skipped sentence: [{pl.i(sent)}]', kind='filtered', args=dict(sentence=sent))
                    continue
                ret.append(sent)
            return ret
        else:
            return sentences

    def process_completions(
            self, completions_dir_name: Union[str, List[str]], expected_samples_per_completion: int = None,
            output_dir_name: str = None
    ):
        """
        Process completions into individual sentences, for step 2: extract entities
        """
        out = dataset_name2data_dir(**self.dir_args, output_dir='Sentence-Dataset', output_postfix=output_dir_name, timestamp='short-date')
        base_path, output_path = out.base_path, out.path
        n_expect = expected_samples_per_completion
        d_log = {
            'class-name': self.__class__.__qualname__, 'metadata': self.meta(), 'output-path': output_path,
            'completions-dir-name': completions_dir_name, 'expected-samples-per-completion': n_expect
        }
        out = process_completions_init(
            completion_base_path=base_path, completions_dir_name=completions_dir_name, output_path=output_path, completion_type='Sentence',
            logger=self.logger, init_log=d_log
        )
        log_prompt_eg(dir_name=completions_dir_name, base_path=base_path, logger=_logger)
        self.ec = self.ec or EdgeCases(logger=self.logger)

        sents = []
        t = Timer()
        for c in out.iter:
            completion, fnm, p_fnm = c.content, c.filename, c.pretty_filename
            if self.as_passage:
                if len(find_match(text=completion, pattern=self.pattern_bold, match_whole_words=False)) > 0:
                    ori = completion
                    # drop the bold-facing by replacing regex matches with the entity name
                    new = self.pattern_bold.sub(r'\g<entity>', completion)
                    d_log = dict(filename=c.pretty_filename, original=ori, modified=new)
                    self.ec(msg=f'Boldface dropped from completion w/ {pl.i(d_log)}', kind='drop-bold')
                    completion = new
            _sents = []

            if self.as_passage:
                if self.dataset_name == 'conll2003-no-misc':
                    m = self.pattern_title.match(completion)  # extract title separately
                    if m is not None:  # add the title separately
                        _sents.append(m.group('title').strip())
                        assert m.start() == 0  # by construction
                        completion = completion[m.end():]
                lines = completion2lines(completion=completion)
                for ln in lines:
                    from nltk.tokenize import sent_tokenize  # lazy import to save time
                    _sents += self._filter_sentences(sent_tokenize(ln))
            else:
                lines = completion2lines(completion=completion)
                lines = self._filter_sentences(sentences=lines)

                for ln in lines:
                    m = match(text=ln, pattern=self.pattern_sent, verbose=True)
                    assert m is not None
                    sent = m.group('sent').strip()
                    # sic('before', sent)

                    if len(find_match(text=sent, pattern=self.pattern_inline_annot)) > 0:
                        ori = sent
                        # drop the inline annotation by replacing regex matches w/ empty string
                        new = self.pattern_inline_annot.sub(r'', sent)
                        new = drop_consecutive_space(text=new)
                        d_log = dict(filename=c.pretty_filename, original=ori, modified=new)
                        if self.drop_with_inline_type:
                            msg = f'Sentence w/ inline entity type annotation dropped w/ {pl.i(d_log)}'
                            self.ec(msg=msg, kind='drop-inline-type', args=dict(sentence=sent))
                            continue
                        else:
                            self.ec(msg=f'Inline entity type annotation dropped from sentence w/ {pl.i(d_log)}', kind='drop-inline-type')
                            sent = new
                    sent = drop_brackets_in_sentence(sentence=sent, pattern_emph=self.pattern_emph, ec=self.ec, filename=fnm)

                    # drop starting double quote if it's the only one
                    if sent[0] == '"' and sent.find('"', 1) == -1:
                        self.ec(msg=f'Starting double quote dropped from sentence: [{pl.i(sent)}]', kind='drop-start-quote')
                        sent = sent[1:]
                    _sents.append(sent)
                _sents = self._filter_sentences(sentences=_sents)

            if n_expect is not None and len(_sents) != n_expect:
                d_log = {'filename': c.pretty_filename, '#expect': n_expect, '#got': len(_sents)}
                if not self.as_passage and self.diverse_context or self.diverse_entity:
                    d_log['sentences'] = _sents
                msg = f'Expected {pl.i(n_expect)} samples, but decoded {pl.i(len(_sents))} w/ {pl.i(d_log)}.'
                self.ec(msg=msg, kind='wrong-sentence-count')

            sents += [drop_enclosing_quotes(s) for s in _sents]

        d_log_count = {'#sentence-extracted': len(sents)}
        out = de_duplicate_samples(samples=sents, logger=self.logger)
        sents, n_drop = out.samples, out.n_dup_drop
        d_log_count.update({'#duplicate-dropped': n_drop, '#sentence-kept': len(sents)})
        out_fnm = os_join(output_path, 'sentences.json')
        with open(out_fnm, 'w') as f:
            json.dump(sents, f, indent=4)
        self.logger.info(self.ec.summary())
        self.logger.info(f'Processed sentences w/ {pl.i(d_log_count)} in {pl.i(t.end())}')


if __name__ == '__main__':
    from chore.setup2generate_sentence_dir_name import *

    # dnm = 'conll2003-no-misc'
    # dnm = 'wiki-gold-no-misc'
    dnm = 'mit-movie'
    # dnm = 'mit-restaurant'
    # dnm = 'job-stack'

    s_fmt = 'natural-pair-v2'
    # n_demo_ = None
    # n_demo_ = 5
    n_demo_ = 1
    unlabeled = None
    # unlabeled = 100

    psg = False
    # psg = True
    dc = False
    # dc = True
    # de = False
    de = True
    # de = 'seeded'
    # de = 'mixed'
    ap = dict(diverse_context=dc, diverse_entity=de)
    da = None if dnm in ['mit-movie', 'mit-restaurant'] else dict(include_none_samples=True)

    # debug = True
    debug = False
    if not psg:
        if not dc and not de:
            if unlabeled is None:
                n_list_ = 20 if debug else 50
            else:
                n_list_ = 10
        else:
            n_list_ = 3
    else:
        n_list_ = 5
    sic(dnm, s_fmt, n_demo_, da, unlabeled)
    sic(dc, de, psg, n_list_)
    sg = SentenceGenerator(dataset_name=dnm, sample_format=s_fmt, attr_prompt_args=ap, as_passage=psg)

    def check_prompt():
        def get_prompt():
            return sg.get_prompt(n_demo=n_demo_, n_list=n_list_, demo_args=da, with_unlabeled=unlabeled)
        n = 50 if (dc or de) else 5
        print_prompts(prompt=lambda: get_prompt(), n=n)

    out_dnm = sg.meta(n_list=n_list_, with_unlabeled=unlabeled, postfix='debug' if debug else None)
    sic(out_dnm)

    def write_completion():
        if not psg:
            # taken from 1-stage gen
            if not dc and not de:
                if debug:
                    n_call = 10
                else:
                    if unlabeled is None:
                        n_call = 36
                        # n_call = 5
                    else:
                        n_call = 165
                max_tok = 2048
            else:
                if debug:
                    n_call = 20
                else:
                    # n_call = 550  # LLM seems to almost always generate 3 sentences as requested
                    n_call = 580
                    # n_call = 720  # for ~2K NER samples in the end
                max_tok = 256
        else:
            if debug:
                n_call = 5
            else:
                n_call = 60
            max_tok = 512
        # md_nm = 'gpt-3.5-turbo'
        md_nm = 'gpt-3.5-turbo-1106'

        sg.write_completions(
            n_prompt=n_call, n_list=n_list_, n_demo=n_demo_, output_dir_nm=out_dnm,
            get_prompt_args=dict(demo_args=da, with_unlabeled=unlabeled),
            model_name=md_nm, max_tokens=max_tok, timeout=20 if (dc or de) else 40
        )

    def process():
        if dnm == 'conll2003-no-misc':
            dir_nm = conll2003_no_misc(dc=dc, de=de, psg=psg, n_demo=n_demo_, unlabeled=unlabeled)
        elif dnm == 'wiki-gold-no-misc':
            dir_nm = wiki_gold_no_misc(dc=dc, de=de, psg=psg, n_demo=n_demo_)
        elif dnm == 'mit-movie':
            dir_nm = mit_movie(dc=dc, de=de, psg=psg, n_demo=n_demo_)
        elif dnm == 'mit-restaurant':
            dir_nm = mit_restaurant(dc=dc, de=de, psg=psg, n_demo=n_demo_)
        elif dnm == 'job-stack':
            dir_nm = job_stack(dc=dc, de=de, psg=psg, n_demo=n_demo_)
        else:
            raise NotImplementedError
        sic(dir_nm)
        sg.process_completions(completions_dir_name=dir_nm, output_dir_name=out_dnm, expected_samples_per_completion=n_list_)
    # check_prompt()
    # write_completion()
    process()
