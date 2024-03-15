"""
3rd step in step-wise data generation: generate entity types given sentences & entity names
"""

import re
import json
import math
import random
from os.path import join as os_join
from typing import Dict, Tuple, List, Union, Iterable, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field, asdict

from stefutil import get_logger, pl, add_file_handler, group_n, Timer
from src.util import sconfig, dataset_name2data_dir, patterns, span_pair_overlap
from src.util.ner_example import NerReadableExample
from src.util import sample_check as check
from src.data_util import prettier, completions, logprob, dataset, split, edit
from src.generate import schemas
from src.generate.step_wise.util import AnnotationGenerator, load_processed, ProcessedSampleOutput
from src.generate.step_wise.util_3_stage import MANUAL_DEMO_SAMPLES, NOT_ENTITY_TYPE, OTHER_ENTITY_TYPE
from src.generate.step_wise.generate_span import SentenceNSpanSample


__all__ = ['TypeGenerator']


_logger = get_logger(__name__)


@dataclass
class LoadSamplesOutput:
    processed_samples: ProcessedSampleOutput = None
    d_log: Dict[str, Any] = None


@dataclass
class LabeledSpan:
    span: str = None

    # dict of (entity type => binary label)
    # labels: Dict[str, bool] = defaultdict(lambda: None)
    is_type: Dict[str, bool] = field(default_factory=lambda: defaultdict(lambda: None))
    # list of correct entity types
    # types: List[str] = field(default_factory=list)


@dataclass
class SentenceSpanLabelsSample:
    """
    A structured sample for a sentence with multiple spans and their binary entity type labels
    """
    sentence: str = None
    spans: List[LabeledSpan] = field(default_factory=list)


@dataclass
class Diff1InsertOutput:
    is_diff_1_insert: bool = None
    index: int = None


def diff_1_insert(lst1: List[Any] = None, lst2: List[Any] = None) -> Diff1InsertOutput:
    """
    :return: Whether the only difference between lst1 and lst2 is 1 insertion
    """
    lst_long, lst_short = (lst1, lst2) if len(lst1) > len(lst2) else (lst2, lst1)
    assert len(lst_long) == len(lst_short) + 1  # sanity check there's only 1 insertion
    for i, (e1, e2) in enumerate(zip(lst_long, lst_short)):
        if e1 != e2:
            return Diff1InsertOutput(is_diff_1_insert=lst_long[i + 1:] == lst_short[i:], index=i)
    # the only difference is the last element
    assert lst_long[:-1] == lst_short  # sanity check
    return Diff1InsertOutput(is_diff_1_insert=True, index=len(lst_short))


@dataclass
class InsertDiffOutput:
    found: bool = None
    indices: List[int] = None


def find_inserted_diff(lst1: List[Any] = None, lst2: List[Any] = None) -> InsertDiffOutput:
    """
    :return: whether lst1 and lst2 differs in insertion only
        If so, return the index of the longer list elements that are not in the shorter list
    """
    lst_long, lst_short = (lst1, lst2) if len(lst1) > len(lst2) else (lst2, lst1)
    n_insert = len(lst_long) - len(lst_short)
    assert n_insert > 0  # sanity check

    idxs = []
    i_long, i_short = 0, 0
    while i_long < len(lst_long):
        e_long, e_short = lst_long[i_long], lst_short[i_short]
        if e_long != e_short:  # add an insertion
            idxs.append(i_long)
            i_long += 1
        else:
            i_long += 1
            i_short += 1
    assert len(idxs) == n_insert  # sanity check
    long_matched = [elm for i, elm in enumerate(lst_long) if i not in idxs]
    if long_matched != lst_short:
        from stefutil import sic
        sic(long_matched, lst_short)
    assert long_matched == lst_short  # sanity check
    return InsertDiffOutput(found=True, indices=idxs)


class TypeGenerator(AnnotationGenerator):
    def __init__(self, **kwargs):
        super().__init__(generate_type='entity-type', **kwargs)
        self.manual_demos = MANUAL_DEMO_SAMPLES[self.dataset_name]

        pref_x = sconfig(f'datasets.{self.dataset_name}.x-name')

        if not self.multi_class_classify:
            # for extracting the (sentence, span, entity type) from the prompts, e.g.
            # `2. In the sentence "Renowned climber John Smith reaches the summit of {{Mount Everest}} for the fifth time.", is the enclosed span of text "Mount Everest" a named person entity?`
            self.pattern_question_search = re.compile(rf'(?P<idx>\d+)\. In the {pref_x.lower()} (?P<sentence>.+), is the enclosed span of text (?P<span>.+) a named (?P<entity_type>.+) entity\?\n')

            # LLM generated labels for each (sentence, span, entity type), e.g. `2. Yes`, `3. No`
            self.pattern_label_search = re.compile(r'(?P<idx>\d+)\. (?P<label>Yes|No)\n')
        else:
            if not (self.multi_class_classify == 'group-span' and self.cot):
                raise NotImplementedError

            pat_space = rf'[ \t]*'  # pattern for horizontal spaces, no newline char
            pat_words = rf'[^.,]+?'  # pattern for smallest span of words, no comma or period

            # `span` to filter out samples that don't follow the formatting, e.g.
            #   `Text Span: "London" - named location entity`
            # self.pattern_span_search = re.compile(rf'{pat_space}Text Span: (?P<span>.+) (?!entity)(.*)\n', re.IGNORECASE)
            # drop spans that contains `entity` in the same row
            self.pattern_span_search = re.compile(rf'{pat_space}Text Span:(?!.*entity).*\n', re.IGNORECASE)

            # each sentence group starts w/ the enum index, assumes the original sentence is not re-generated
            # matches the (span, label) pair as a single classification output, e.g.
            # `Text Span: "Munich"
            # Label: "Munich" is a location. The span is named location entity.`
            # since a single enum index for all spans in a group, don't match the index here
            self.pattern_pair_search = [
                re.compile(rf'{pat_space}Text Span: (?P<span>.+)\n{pat_space}Label: (?P<reasoning>.+) The span is (?P<entity_type>.+)\.\n', re.IGNORECASE),
                re.compile(rf'{pat_space}Text Span: (?P<span>.+)\n{pat_space}Label: (?P<reasoning>.+) The span is a (?P<entity_type>.+)\.\n', re.IGNORECASE),

                # for edge case: final label formatting not generated, e.g.
                #   In CoNLL-03
                #       `Label: "Dr. Anthony Fauci" is a specific person's name. A specific person's name is a person entity.`
                #       `Label: "The Met Museum" refers to a specific museum, so it is a named organization entity.`
                #       `Label: "Chimamanda Ngozi Adichie" is a full name of an individual, so it should be a named person entity.`
                #       `Label: "Great Wall of China" is a specific landmark and a named location entity.`
                #   In MIT-Movie
                #       `Label: "1980s" is a specific time period when the movies were released. A time duration of movie release counts as a Year entity.`
                #       `Label: "Judgement" is not a named entity.` when all other samples in the same completion have CoT reasoning
                #       `Label: "latest" is a superlative adjective. It does not specify a named entity.`
                #       `Label: "famous actors" refers to specific people who starred in the movie. Specific people are named Actor entities.`
                re.compile(rf'{pat_space}Text Span: (?P<span>.+)\n{pat_space}Label: (?P<reasoning>.+)[.,] (?P<entity_type_prefix>.+) (is|and|be|as|does|are|thus) (?P<entity_type>.+?)\.\n', re.IGNORECASE),
                re.compile(rf'{pat_space}Text Span: (?P<span>.+)\n{pat_space}Label: (?P<reasoning>.+) (is|and|be|as|does|are|thus) (?P<entity_type>.+?)\.\n', re.IGNORECASE),
            ]
            self.pattern_pair_search_no_cot = [
                # for edge case: no CoT reasoning generated, e.g.
                #   `Label: named person entity`, `named entity of other type`
                re.compile(rf'{pat_space}Text Span: (?P<span>.+)\n{pat_space}Label: (?P<entity_type>{pat_words} entity)\n', re.IGNORECASE),
                re.compile(rf'{pat_space}Text Span: (?P<span>.+)\n{pat_space}Label: (?P<entity_type>{pat_words} entity{pat_words})\n', re.IGNORECASE)
            ]
            self.pattern_pair_search += self.pattern_pair_search_no_cot

            self.pattern_span = re.compile(rf'^{pat_space}Text Span: (?P<span>.+)$', re.IGNORECASE)
            self.pattern_label = re.compile(rf'^{pat_space}Label: (?P<label>.+)$', re.IGNORECASE)
            self.pattern_label_no_fmt = re.compile(rf'^{pat_space}(?P<span>.+) is (?P<reasoning>.+) (?P<entity_type>not a named entity)\.$', re.IGNORECASE)

            # for edge case: entire sample don't follow the formatting, e.g.
            #   `1. "Apple Inc." is the name of a specific company, so it is a named organization entity.`
            #   `   "thousands" is a numerical quantity and not a named entity.
            #   `Text Span: "Border Network for Human Rights" - named organization entity`
            #   `Text Span: "Graz" - "Graz" is the name of a specific city. The name of a city is a location entity. The span is a named location entity.`
            # TODO: A format I can't handle cos 2 spans in 1 line, e.g.
            #   `Label: "police" and "activists" are groups of people but not specific organizations or individuals. The spans are not named entities.`
            pat_prefix_edge = rf'(Text Span: )?'
            # pat_prefix_edge = rf'((Text Span|Label): )?'
            self.pattern_span_search_edge = [
                re.compile(rf'{pat_space}"(?P<span>.+)" is (.+) entity\.\n', re.IGNORECASE),
                re.compile(rf'{pat_space}{pat_prefix_edge}"(?P<span>.+)" - (?P<label>.+)\n', re.IGNORECASE),
            ]
            self.pattern_pair_search_edge = [
                re.compile(rf'{pat_space}"(?P<span>.+)" is (?P<reasoning>.+) (is|and) (?P<entity_type>.+)\.\n', re.IGNORECASE),
                re.compile(rf'{pat_space}"(?P<span>.+)" - (?P<reasoning>.+)[.,] The span is (?P<entity_type>.+)\.\n', re.IGNORECASE),
            ]
            self.pattern_pair_search_edge_no_cot = [
                re.compile(rf'{pat_space}{pat_prefix_edge}"(?P<span>.+)" - (?P<entity_type>.+)\n', re.IGNORECASE),
            ]
            self.pattern_pair_search_edge += self.pattern_pair_search_edge_no_cot

    def get_instruction(self, n_classify: int = 20):
        if self.dataset_name == 'conll2003-no-misc':
            # discard for after so long
            # return (f"I have {n_classify} sentences from news stories. "
            #         "All entity names occurred in the sentences are identified. "
            #         "Many of the identified entities belong to one of the following entity types:\n"
            #         "[person, location, organization].\n"
            #         "Please categorize each named entity into one of the following entity types:\n"
            #         "[person, location, organization, ambiguous, other, uncertain].\n"
            #         "If an entity fall into multiple categories above, tag it as 'ambiguous'.\n"
            #         "If an entity doesn't belong to any of the above categories, tag it as 'other'.\n"
            #         "If you are unsure about an entity’s category, tag it as 'uncertain'. ")

            if not self.multi_class_classify:  # binary span classification
                ret = (f"Here are {n_classify} sentences from news stories. Potential named entities occurred in the sentences are identified. "
                       f"Your task is to classify whether spans of text are named entities of the following entity types:\n"
                       f"[person, location, organization].\n")
                ret += "In particular, for each sentence with a span of text enclosed in double curly braces and an entity type given, "
                if self.cot:
                    act = 'analyze and classify'
                else:
                    act = 'classify'
                ret += f"please {act} whether the enclosed span of text is a named entity of the given type or not "
                ret += ("in context of the sentence.\n"
                        "If a span of text is not a named entity, the label should be No.")
            else:  # use multi-class classification to save token count thus cost
                # TODO: prompt fix: specify sentence count; drop `double curly braces`
                ret = (f"Here are some sentences from news stories with potential named entities identified. "
                       f"Your task is to classify whether the spans of text are named entities of the following entity types:\n"
                       f"[person, location, organization]. \n")
                if self.cot:
                    act = 'analyze and classify'
                else:
                    act = 'classify'
                ret += ("In particular, for each sentence with a span of text enclosed in double curly braces, "
                        f"please {act} the enclosed span of text in context of the sentence into one of the following categories:\n")
                # ret += ("[Is named entity and class is person, Is named entity and class is location, Is named entity and class is organization, "
                #         "Is named entity and class is other, Not a named entity]\n")
                # a more compact version
                # ret += "[named person entity, named location entity, named organization entity, other named entity, not a named entity]\n"
                ret += f'{pl.nc(self.t2gt.gen_entity_types)}\n'

                if self.insert:
                    ret += f"Please use the definitions below to {act} the text spans.\n"
                    if self.insert == 'defn':
                        # after demo update
                        defn = schemas.conll2003_no_misc_defn3
                    else:
                        assert self.insert == 'schema'
                        raise NotImplementedError
                    ret += f'{defn}\n\n---\n'

                # ignore since should be clear from the classes above
                # ret += "If a span of text is not a named entity, the label should be Not a named entity.\n"
            return ret
        elif self.dataset_name == 'mit-movie':
            assert self.multi_class_classify
            if n_classify == 1:
                ret = 'Here is a spoken query to a dialogue system about movies.'
            else:
                ret = f'Here are {n_classify} spoken queries to a dialogue system about movies.'
            act = 'classify'
            if self.cot:
                q = 'the query' if n_classify == 1 else 'each query'
                act = f'analyze {q} and classify'
            ret += f' Your task is to {act} whether the spans of text are named entities of the following entity types:\n'
            ret += f'{pl.nc(self.entity_types)}.\n'

            act = 'analyze and classify' if self.cot else 'classify'
            ret += (f'In particular, for each query with a span of text, '
                    f'please {act} the span of text in context of the query into one of the following categories:\n')
            ret += f'{pl.nc(self.t2gt.gen_entity_types)}.\n'

            # explicit annotation rule to correct CLS label errors
            # ret += "Actor and Director entities must be person's names.\n"
            # ret += ("Named Director, Actor and Character entities must be person's names. "
            #         'Certain categories such as "strong female character" are not named entities.\n')
            ret += ("Named Director, Actor and Character entities must be person's names. "
                    'Certain categories of people such as "strong female character" are not named entities.\n')

            # ret += 'Named Song entities must be song names.\n'
            ret += "Named Song entities must be specific song names. "
            # ret += 'General mentions and artist names such as "popular songs", "soundtrack" and "Beyoncé" are not named Song entities.\n'
            # ret += 'Song categories and artist names such as "popular songs", "soundtrack" and "Beyoncé" are not named Song entities.\n'
            # ret += 'Song categories and artist names such as "popular songs", "soundtrack" and "Beyoncé" are not named Song entities.\n'
            ret += 'Song categories, song references and artist names such as "popular songs", "soundtrack" and "Beyoncé" are not named Song entities.\n'

            # ret += "Named Viewers' Rating, MPAA Rating and Plot entities must contain specific information. "
            # ret += 'General mentions such as "viewers rating", "main plot" and "MPAA Rating" are not named entities.\n'
            # ret += 'General mentions and information requests such as "viewers\' rating", "main plot" and "MPAA Rating" are not named entities.\n'
            # ret += 'General mentions and request types such as "viewers\' rating", "main plot" and "MPAA Rating" are not named entities.\n'

            # ret += "Named Viewers' Rating and MPAA Rating entities must contain specific information. "
            # ret += "Named Viewers' Rating and MPAA Rating entities must contain actual ratings. "
            # ret += 'General mentions and information requests such as "viewers\' rating" and "MPAA Rating" are not named entities.\n'
            # ret += 'General mentions such as "viewers\' rating" and "MPAA Rating" are not named entities.\n'
            # ret += 'General mentions and request types such as "viewers\' rating" and "MPAA Rating" are not named entities.\n'
            # ret += 'Geneal request types such as "viewers\' rating" and "MPAA Rating" are not named entities.\n'
            ret += 'Geneal mentions such as "viewers\' rating" and "MPAA Rating" are not named Viewers\' Rating and MPAA Rating entities.\n'

            # ret += 'Named Plot entities must contain actual plot themes or elements.'
            ret += 'Named Plot entities must be actual plot themes or elements. '
            # ret += 'Plot request and plot categories such as "main plot" are not named Plot entities.\n'
            # ret += 'General reference and plot categories such as "main plot" are not named Plot entities.\n'
            ret += 'General mentions such as "main plot" are not named Plot entities.\n'
            # ret += 'General mentions such as "main plot" and "plot summary" are not named Plot entities.\n'
            # ret += 'Simply referring to plot such as "main plot" and "plot summary" are not named Plot entities.\n'
            # ret += 'Simply mention plot and plot categories such as "main plot" and "plot summary" are not named Plot entities.\n'
            # ret += '"main plot" and "plot summary" simply mention plots and are not named Plot entities.\n'
            return ret
        else:
            raise NotImplementedError

    def get_prompt(self, samples: Iterable[Dict[str, Any]] = None, manual_demo: bool = False, **kwargs):
        if manual_demo:
            kwargs['add_demos'] = self.manual_demos
        return super().get_prompt(samples=samples, **kwargs)

    def get_span_unique_type(self):
        """
        Get the `unique_type` argument for `load_processed`
        """
        if not self.multi_class_classify:
            return 'type'
        else:
            assert self.multi_class_classify in ['independent-span', 'group-span']  # sanity check
            if self.multi_class_classify == 'independent-span':  # classify each span independently
                return 'span'
            else:  # group all spans in a sentence together
                return 'sentence'

    def load_samples(self, dir_name: str = None, shuffle: bool = None) -> LoadSamplesOutput:
        # Sentences-w/-spans are already shuffled in last sep, shuffle the samples again
        out = load_processed(
            dataset_name=self.dataset_name, dir_name=dir_name, kind='span', logger=self.logger,
            shuffle=shuffle, span_unique_type=self.get_span_unique_type()
        )
        samples, samples_w_spans = out.samples, out.samples_w_span

        if 'entity_spans' in samples[0]:
            assert all('entity_spans' in samp for samp in samples)  # sanity check
            n_sent = len(samples)
            n_pair = sum(len(samp['entity_spans']) for samp in samples)
        else:
            assert 'entity_span' in samples[0]
            assert all('entity_span' in samp for samp in samples)
            n_sent = len(set(samp['sentence'] for samp in samples))  # sentences should be all unique
            n_pair = len(samples)
        return LoadSamplesOutput(processed_samples=out, d_log={'#unique-sentences': n_sent, '#unique (sentence, span) pairs': n_pair})

    def write_completions(
            self, n_classify: int = 10, n_demo: Optional[int] = 5, samples_dir_name: str = None, output_dir_nm: str = None,
            shuffle_samples: Union[bool, int] = None, prompt_args: Dict[str, Any] = None, prompt_seed: int = None, **kwargs
    ):
        """
        :param n_classify: #sentences to classify in a single prompt
        :param n_demo: #demo sentences
        :param samples_dir_name: directory name of the samples (sentence&spans) to be classified
        :param output_dir_nm: output directory name of the LLM completions
        :param prompt_args: arguments for prompt construction
        :param prompt_seed: seed for randomness in prompt demo sample construction
        :param shuffle_samples: whether to shuffle the (sentence, span) samples
        :param kwargs: additional arguments for LLM completion
        """
        output_path = dataset_name2data_dir(**self.dir_args, output_dir=f'{self.sample_type}-Res', output_postfix=output_dir_nm).path
        add_file_handler(logger=self.logger, file_path=os_join(output_path, 'completion.log'))

        out = self.load_samples(dir_name=samples_dir_name, shuffle=shuffle_samples)
        out, d_log_samples = out.processed_samples, out.d_log
        # only classify samples that have spans, cos otherwise, not needed anyway
        samples = out.samples_w_span or out.samples
        samples_group = group_n(samples, n=n_classify)
        if prompt_seed:
            random.seed(prompt_seed)
        prompts = [self.get_prompt(samples=samp, n_demo=n_demo, **(prompt_args or dict())) for samp in samples_group]
        if prompt_seed:
            random.seed()  # reset seed

        d_log = {
            'dataset-name': self.dataset_name, 'multi-class-classify': self.multi_class_classify,
            '#classify': n_classify, '#demo': n_demo, 'get-prompt-args': prompt_args,
            'generated-samples-dir-name': samples_dir_name, 'shuffle-samples': shuffle_samples,
            **d_log_samples, 'output-path': output_path
        }
        # sic(prompts)
        # raise NotImplementedError

        completions.write_completions(
            output_path=output_path, logger=self.logger, add_fl_writer=False,
            completion_type=self.processed_type, init_log=d_log, prompts=prompts, save_all_prompts=True, **kwargs
        )

    def sample_x2span(self, s: str = None) -> str:
        """
        Get the span from the span match
        """
        m = self.pattern_span.match(s)
        assert m is not None
        return edit.drop_enclosing_quotes(m.group('span'))

    def process_completions(
            self, samples_dir_name: str = None,
            completions_dir_name: completions.CompletionDirectoryDict = None,
            output_dir_name: str = None, expected_samples_per_completion: int = None,
            shuffle_samples: Union[bool, int] = None, logprobs: bool = True, lowercase: bool = None
    ) -> dataset.NerProcessOutput:
        """
        :param samples_dir_name: directory name of (sentence, spans) samples
        :param completions_dir_name: LLM responses for classification labels
        :param output_dir_name: output directory name
        :param expected_samples_per_completion: #samples expected in each completion
        :param shuffle_samples: whether to load samples in shuffled order
            If an int, use that as the random seed for shuffling
        :param logprobs: whether to extract logprobs from the completions
        :param lowercase: whether to lowercase the samples
        """
        from stefutil import sic

        d_out = dataset_name2data_dir(
            **self.dir_args, output_dir=f'{self.processed_type}-Dataset', output_postfix=output_dir_name, timestamp='short-date')
        output_path, base_path = d_out.path, d_out.base_path
        init_log = {
            'class-name': self.__class__.__qualname__, 'metadata': self.meta(), 'multi-class-classify': self.multi_class_classify,
            'output-path': output_path, 'completions-dir-name': completions_dir_name,
            'expected-samples-per-completion': expected_samples_per_completion,
            'generated-samples-dir-name': samples_dir_name, 'shuffle-samples': shuffle_samples, 'logprobs': logprobs, 'lowercase': lowercase
        }
        n_expect = expected_samples_per_completion

        out = self.load_samples(dir_name=samples_dir_name, shuffle=shuffle_samples)
        out, d_log_samples = out.processed_samples, out.d_log
        samples = out.samples_w_span or out.samples
        sents = [samp['sentence'] for samp in samples]

        it = completions.process_completions_init(
            completions_dir_name=completions_dir_name, completion_type=self.processed_type, logger=self.logger,
            completion_base_path=base_path, output_path=output_path, init_log=init_log, logprobs=logprobs)
        d_log_count = {'#completions': len(it.filepaths), **d_log_samples}
        completions.log_prompt_eg(dir_name=completions_dir_name, base_path=base_path, logger=self.logger)
        prompt_fnm = os_join(base_path, completions_dir_name, 'prompts.json')  # for mapping labels to the entity types
        with open(prompt_fnm) as f:
            prompts = json.load(f)['prompts']
        assert len(prompts) == len(it.filepaths)  # sanity check

        t = Timer()
        if out.samples_wo_span is not None:
            # samples w/o spans didn't go through step 3 cls by construction, add these back
            ret = [NerReadableExample.from_d(sentence=samp['sentence']) for samp in out.samples_wo_span]
        else:
            ret = []
        # save challenging (sentence, spans) pair samples to be re-generated
        #   `challenging` as in, e.g. No CoT generated, incorrect sample format, NER sample extraction failure
        challenging_samples: Dict[str, List[SentenceNSpanSample]] = defaultdict(list)
        # rank hard classifications by logprob
        # for each (sentence, span, entity type) triple, get the logprobs corresponding to the `label` sequence
        triples2logprob: Optional[Dict[Tuple[str, str, str], logprob.LogProbOutput]] = dict() if logprobs else None

        n_cpl = len(it.filepaths)
        # TODO: remember to include samples w/ no spans
        for i_cpl, c in enumerate(it.iter):
            is_last_group = i_cpl == n_cpl - 1
            completion, fnm, p_fnm = c.content, c.filename, c.pretty_filename

            if not self.multi_class_classify:
                if logprobs:
                    raise NotImplementedError
                # assume (sentence, span, entity type) questions are always not re-generated, each line should be a label
                out_label = split.split_samples(
                    completion=completion, pattern=self.pattern_label_search, has_enum_prefix=True, filename=p_fnm)
                assert out_label.success
                ms_lb = out_label.matches
                labels = [m.group('label') for m in ms_lb]
                n_labels_got = len(labels)

                # construct labeled sample from multiple binary CLS labels; sentence=>spans=>types=>binary-labels
                labeled_samples = defaultdict(lambda: SentenceSpanLabelsSample())

                # extract the (sentence, span, entity type) from the prompts
                prompt = prompts[i_cpl]
                # find the part where classification questions begins
                idx_strt = prompt.rfind('---\n')
                assert idx_strt != -1
                prompt = prompt[idx_strt+4:]
                out_ques = split.split_samples(
                    completion=prompt, pattern=self.pattern_question_search, has_enum_prefix=True, filename=p_fnm)
                assert out_ques.success
                ms_ques = out_ques.matches

                assert len(ms_ques) == n_labels_got  # sanity check
                # construct the labeled sample from each binary classification
                for m_q, lb in zip(ms_ques, labels):
                    sent, span, entity_type = m_q.group('sentence', 'span', 'entity_type')
                    sent, span = edit.drop_enclosing_quotes(sent), edit.drop_enclosing_quotes(span)

                    assert sent.count(span) == 1  # TODO: for now, just consider single-occurring spans
                    # drop the enclosing braces
                    sent = sent.replace(self.entity_enclose_open, '').replace(self.entity_enclose_close, '')
                    assert sent in sents  # sanity check, always the case since prompt is constructed from the samples

                    sample = labeled_samples[sent]
                    # add sentence
                    if sample.sentence is None:
                        sample.sentence = sent
                    else:
                        assert sample.sentence == sent
                    # add the entity type label result for a given span
                    spans = sample.spans
                    if all(sp.span != span for sp in spans):
                        sp = LabeledSpan(span=span)
                        spans.append(sp)
                    else:
                        sp = next(sp for sp in spans if sp.span == span)
                    sp.is_type[entity_type] = lb == 'Yes'  # label is in [`Yes`, `No`]
                labeled_samples = list(labeled_samples.values())
                n_samples_got = len(labeled_samples)
                if n_expect:  # unless it's the last completion, #sentences should match
                    if is_last_group:
                        assert n_samples_got <= n_expect
                    else:
                        assert n_samples_got == n_expect

                # for each span, pick out the positive entity types
                # labeled_samples_ = []
                for samp in labeled_samples:
                    annotations = []
                    for sp in samp.spans:
                        assert len(sp.is_type) == len(self.entity_types)  # sanity check
                        types = [et for et, is_typed in sp.is_type.items() if is_typed]
                        annotations.append((sp.span, types))
                    sic(samp.sentence, annotations)
                raise NotImplementedError
            else:
                if completion[-1] != '\n':  # for pattern match
                    completion += '\n'

                # get samples annotated in this completion
                i_s, i_e = i_cpl * n_expect, (i_cpl + 1) * n_expect
                input_samples = samples[i_s:i_e]
                spans_expect = sum([samp['entity_spans'] for samp in input_samples], start=[])
                n_spans_expect = len(spans_expect)

                def add_to_challenging_samples(
                        d: Dict[str, List[SentenceNSpanSample]] = None, kind: str = None, samples_: List[SentenceNSpanSample] = None):
                    d[kind] += [SentenceNSpanSample.from_d(samp) for samp in (samples_ or input_samples)]

                # initial check on the number of spans in the completion
                ms_spans = patterns.find_non_overlap_matches(
                    pattern=self.pattern_span_search, text=completion, return_matches=True, accept_no_match=True)
                n_spans_got = len(ms_spans)
                wrong_format = False  # generated samples don't follow the expected sample format
                span_idxs_drop = []  # generated spans not in prompt, filter out these by index
                if n_spans_got > n_spans_expect:
                    # LLM generated more spans than provided
                    spans = [self.sample_x2span(m.group()) for m in ms_spans]

                    # the new spans not in prompt must be inserted, find the indices for such spans; they will be dropped later
                    diff = find_inserted_diff(lst1=spans, lst2=spans_expect)
                    assert diff.found
                    span_idxs_drop = diff.indices
                    inserted = [spans[i] for i in span_idxs_drop]

                    d_log = dict(
                        n_spans_expect=n_spans_expect, n_spans_got=n_spans_got, spans_expect=spans_expect, spans_got=spans,
                        inserted=inserted, completion=completion, filename=p_fnm)
                    msg = f'Edge case: #spans in completion exceeds #spans requested in prompt w/ {pl.i(d_log)}'
                    kd = 'generated-unrequested-spans'
                    self.ec(msg=msg, kind=kd, args=dict(filename=fnm))
                    add_to_challenging_samples(challenging_samples, kind=kd)
                elif n_spans_got < n_spans_expect:
                    # spans = [m.group() for m in ms_spans]
                    spans = [self.sample_x2span(m.group()) for m in ms_spans]
                    missing_spans = set(spans_expect) - set(spans)
                    # order by the index of the span in the prompt
                    missing_spans = sorted(missing_spans, key=lambda sp: spans_expect.index(sp))
                    d_log = dict(
                        n_spans_expect=n_spans_expect, n_spans_got=n_spans_got,
                        matched_spans=spans, expected_spans=spans_expect, missing_spans=missing_spans,
                        completion=completion, filename=p_fnm)

                    assert n_spans_got < n_spans_expect  # sanity check
                    if n_spans_got > 0:  # uses the expected format, but assume didn't finish generating all the spans, extraction fails
                        msg = f'Edge case: #spans in completion does not match #spans in samples, completion ignored w/ {pl.i(d_log)}'
                        kd = 'generated-incomplete-spans'
                        self.ec(msg=msg, kind=kd, args=dict(filename=fnm), failed=True)
                        add_to_challenging_samples(challenging_samples, kind=kd)
                        continue
                    else:  # assume the complete #spans required are generated, but not in the required format
                        assert n_spans_got == 0
                        wrong_format = True
                        kd = 'incorrect-sample-format'

                        ms_spans_edge = []
                        for p in self.pattern_span_search_edge:
                            # try one by one until able to find all expected samples
                            ms_spans_edge = patterns.find_non_overlap_matches(
                                pattern=p, text=completion, return_matches=True, accept_no_match=True)
                            if len(ms_spans_edge) == n_spans_expect:
                                break
                        n_spans_got_edge = len(ms_spans_edge)
                        if n_spans_got_edge == n_spans_expect:  # sanity check edge case format can find all expected spans
                            failed = False
                        else:
                            failed = True
                            kd = f'{kd}-failed'
                        # spans = [m.group() for m in ms_spans_edge]
                        spans = [self.sample_x2span(m.group()) for m in ms_spans]

                        d_log.update(n_spans_got_edge=n_spans_got_edge, matched_spans_edge=spans)

                        msg = f'Edge case: generated samples do not follow the expected format w/ {pl.i(d_log)}'
                        self.ec(msg=msg, kind=kd, args=dict(filename=fnm), failed=failed)
                        add_to_challenging_samples(challenging_samples, kind=kd)
                        if failed:
                            continue

                # require one pattern to match all samples, to make sure re.Match object index is correct for logprob extraction
                n_match_expect = n_spans_expect + len(span_idxs_drop)
                find_match_args = dict(text=completion, return_matches=True, return_match_map=True, min_match=n_match_expect)
                if not wrong_format:
                    out = patterns.find_non_overlap_matches(**find_match_args, pattern=self.pattern_pair_search)
                else:
                    out = patterns.find_non_overlap_matches(**find_match_args, pattern=self.pattern_pair_search_edge)
                    # ms_pairs, pat_map = out.matches, out.pattern_map
                    # assert len(ms_pairs) == n_spans_expect
                    # cot_found = True  # TODO: check for this
                ms_pairs, pat_map = out.matches, out.pattern_map
                assert len(ms_pairs) == n_match_expect  # should match all (span, label) pairs
                assert len(pat_map) == 1  # sanity check since patterns are not union-ed, one pattern matches all spans
                pat_matched = next(iter(pat_map.keys()))
                if not wrong_format:
                    cot_found = pat_matched not in self.pattern_pair_search_no_cot
                else:
                    cot_found = pat_matched not in self.pattern_pair_search_edge_no_cot
                if not cot_found:
                    d_log = dict(completion=completion, filename=p_fnm)
                    kd = 'no-cot'
                    self.ec(msg=f'Edge case: no CoT reasoning generated w/ {pl.i(d_log)}', kind=kd, args=dict(filename=fnm))
                    add_to_challenging_samples(challenging_samples, kind=kd)

                # pairs = [m.group('span', 'entity_type') for m in ms_pairs]
                pairs = [(m.group('span'), m.group('entity_type'), m.group()) for m in ms_pairs]
                # sic(pairs)
                # raise NotImplementedError

                if not wrong_format:
                    if len(span_idxs_drop) > 0:
                        assert cot_found  # sanity check
                        # pairs = [(span, et) for span, et in pairs if drop_enclosing_quotes(span) not in span_idxs_drop]
                        # sic(span_idxs_drop, pairs)
                        pairs = [(span, et, whole) for i, (span, et, whole) in enumerate(pairs) if i not in span_idxs_drop]
                        # sic('after drop', pairs)
                else:
                    assert len(span_idxs_drop) == 0  # sanity check
                assert len(pairs) == n_spans_expect  # sanity check
                # if '-138' in fnm:
                #     sic(pairs, wrong_format, pat_matched, cot_found)
                it_pr = iter(pairs)
                it_lb_str = None

                label_strs, s2lp = None, None
                if logprobs:
                    lp_args = dict(kind='generate-type', logger=self.logger, ec=self.ec)
                    label_strs, s2lp = [], logprob.Span2LogProb(completion=completion, logprobs=c.logprobs, cot=cot_found, **lp_args)
                    for m in ms_pairs:
                        pair_str = m.group()
                        assert pair_str[-1] == '\n'
                        pair_str = pair_str[:-1]

                        strt, end = m.span()
                        # leading indentation spaces will also be included in the span,
                        # but hard to remove since the initial tokens are, e.g. [`  `, ` Label`]
                        if not wrong_format:
                            lines_pair = pair_str.split('\n')
                            lines_pair = [ln for ln in lines_pair if ln.strip()]  # remove whitespace-only lines
                            if len(lines_pair) != 2:
                                sic(pair_str, lines_pair)
                            assert len(lines_pair) == 2
                            span_str, label_str = lines_pair
                            assert (self.pattern_span.match(span_str) and self.pattern_label.match(label_str)) or \
                                self.pattern_label_no_fmt.match(label_str)

                            # get global span index of the label part, will be used to get the logprobs
                            len_span_str, len_label_str = len(span_str), len(label_str)
                            label_span_strt = strt + len_span_str + 1  # +1 for the newline
                            label_span_end = label_span_strt + len_label_str + 1  # +1 for the newline
                            label_str += '\n'  # needs the trailing newline in the label part since the logprobs group `.\n` into a single token
                        else:
                            # incorrect formatting, (span, cot, label) all in one line => get log prob for the entire thing
                            assert '\n' not in pair_str  # since the trailing newline is removed

                            label_str = f'{pair_str}\n'
                            label_span_strt, label_span_end = strt, end
                        if completion[label_span_strt:label_span_end] != label_str:
                            sic(pair_str, completion[label_span_strt:label_span_end], label_str)
                        assert completion[label_span_strt:label_span_end] == label_str  # sanity check

                        label_strs.append((label_str, label_span_strt, label_span_end))
                    it_lb_str = iter(label_strs)

                has_enum = completions.completion_has_enum_prefix(completion=completion)
                if has_enum:  # sanity check sentence group indices are correct
                    ms_idx = patterns.find_non_overlap_matches(pattern=split.pattern_enumerated_index, text=completion, return_matches=True)
                    if not is_last_group:
                        assert n_expect == len(input_samples)
                    check_idx = split.check_match_group_indices(ms=ms_idx)
                    assert check_idx.match_success
                    if n_expect != len(ms_idx):  # an edge case already logged
                        assert len(ms_idx) < n_expect

                # get the spans one by one, they should match the spans in the original samples
                samples_out = []

                for samp in input_samples:
                    sent, spans = samp['sentence'], samp['entity_spans']
                    assert len(spans) == len(set(spans))  # sanity check all spans are distinct, by construction of 2nd step

                    ets = []
                    add_span_mismatch = False
                    kd_span_mismatch = 'generated-span-mismatch'
                    for span in spans:
                        span_gen, et, whole = next(it_pr)
                        span_gen: str
                        span_gen = edit.drop_enclosing_quotes(span_gen.strip())
                        if span_gen != span:
                            d_log = dict(span_generated=span_gen, span_expected=span)
                            msg = f'Edge case: generated span does not match span in prompt w/ {pl.i(d_log)}'
                            self.ec(msg=msg, kind=kd_span_mismatch, args=d_log)
                            add_span_mismatch = True

                            span = span_gen  # use the generated span instead, for a correct CLS label

                            # sic(span_gen, span)
                        # assert span_gen == span  # sanity check

                        et = self.t2gt.decode(generated_entity_type=et, ec=self.ec, generated_sample=whole)
                        if logprobs:
                            lb_str, lb_s, lb_e = next(it_lb_str)
                            lp_out = s2lp(span=lb_str, index_start=lb_s, index_end=lb_e)
                            triples2logprob[(sent, span, et)] = lp_out
                        ets.append(et)
                    if add_span_mismatch:
                        sic(samp)
                        add_to_challenging_samples(challenging_samples, kind=kd_span_mismatch, samples_=[samp])

                    enms, ets_out = [], []
                    for span, et in zip(spans, ets):
                        if et in self.entity_types:  # keep only spans that are classified as named entities
                            enms.append(span)
                            ets_out.append(et)
                        else:
                            assert et in [NOT_ENTITY_TYPE, OTHER_ENTITY_TYPE]
                            self.ec(msg=f'Info: no-relevant entity span [{pl.i(span)}] dropped', kind='negative-span-drop', args=dict(span=span))
                    # sanity check entity occurrence, should've been checked in the 2nd span-generation step
                    assert all(s.lower() in sent.lower() for s in enms)
                    d_log = dict(sentence=sent, entity_names=spans, entity_types=ets_out)

                    ic = True
                    if edit.entities_differ_in_case_only(entity_names=spans, sentence=sent, ec=self.ec):
                        ic = False
                        msg = f'Edge case: entity names differ in case only w/ {pl.fmt(d_log)}'
                        self.ec(msg=msg, kind='entity-case-diff')

                        # TODO: check span casing matches & BIO format conversion works
                        sic(sent, spans, ets_out)
                        raise NotImplementedError
                    c = check.get_non_overlapping_keyword_counts(sentence=sent, keywords=enms, ignore_case=ic)
                    # sanity check all entity spans labeled are not overlapping;
                    # this could happen since we allow generating overlapping spans in the 2nd step
                    if any(c[enm] == 0 for enm in enms):
                        # rank spans and keep only high-ranked ones
                        if logprobs:
                            enm2lp = {enm: triples2logprob[(sent, enm, et)].avg_logprob for enm, et in zip(enms, ets_out)}
                            enm2et = dict(zip(enms, ets_out))

                            assert set(c.values()) == {0, 1}  # sanity check, otherwise need to improve algorithm

                            def span2span(s: str) -> Tuple[int, int]:
                                """
                                # find the span (start & end index) of the entity span
                                """
                                ms = patterns.find_match(text=sent, keyword=s, ignore_case=ic)
                                assert len(ms) == 1  # sanity check occurred once in the sentence
                                m = ms[0]
                                return m.span()

                            # get span groups that overlap
                            spans_not_found = [enm for enm in enms if c[enm] == 0]
                            conflict_groups = []
                            for span in spans_not_found:
                                span_span = span2span(span)
                                grp = [span]
                                other_spans = [enm for enm in enms if enm != span]
                                for other_span in other_spans:
                                    other_span_span = span2span(other_span)

                                    # add to conflict group if the span indices overlap
                                    if span_pair_overlap(span_span, other_span_span):
                                        grp.append(other_span)
                                conflict_groups.append(grp)

                            # select the one w/ highest logprob; TODO: or use the longest span?
                            enms_in_conflict = list(dict.fromkeys([enm for grp in conflict_groups for enm in grp]))
                            enms_not_in_conflict = [enm for enm in enms if enm not in enms_in_conflict]
                            if len(conflict_groups) == 1 and len(conflict_groups[0]) == 2:  # a simple case, just pick 1 out of 2
                                conflict_group = conflict_groups[0]
                                keep = max(conflict_group, key=lambda enm: enm2lp[enm])
                                enms_keep = [keep]
                                span_groups = [pl.nc(dict(conflict_group=tuple(conflict_group), kept=keep))]  # for compact edge case logging
                            else:
                                raise NotImplementedError
                            enms = enms_not_in_conflict + enms_keep
                            ets_out = [enm2et[enm] for enm in enms]

                            d_log.update(entity_names_after_overlap_filter=enms, entity_types_after_overlap_filter=ets_out)
                            msg = f'Edge case: overlapping entity spans filtered w/ {pl.fmt(d_log)}'
                            self.ec(msg=msg, kind='overlapping-entity-span-filtered', args=dict(span_groups=span_groups))  # just 1 group
                        else:
                            raise NotImplementedError

                        # sanity check issue resolved
                        c = check.get_non_overlapping_keyword_counts(sentence=sent, keywords=enms, ignore_case=ic)
                        assert all([enm] != 0 for enm in enms)

                    if any(c[enm] > 1 for enm in enms):
                        # since spans to classify are all unique, to construct NER samples, we need to duplicate the multi-occurring spans
                        # note each of such span will have the same entity type since distinct annotations are not supported
                        out = edit.duplicate_multi_occurring_entities(
                            entity_names=enms, entity_types=ets_out, entity_name2count=c, d_log=d_log)
                        enms, ets_out = out.entity_names, out.entity_types

                        msg = f'Edge case: Entity span appears multiple times w/ {pl.fmt(d_log)}'
                        self.ec(msg=msg, kind='multi-occur-entity', args=dict(entity_names=[enm for enm in enms if c[enm] > 1]))

                    # sanity check should negate each other
                    assert ic is not edit.entities_differ_in_case_only(entity_names=enms, sentence=sent, ec=self.ec)
                    # re-order entities
                    out = edit.reorder_entities(sentence=sent, entity_names=enms, entity_types=ets_out, ignore_case=ic)
                    if out.reordered:
                        d_log['reordered_entity_names'] = out.entity_names
                        if ets is not None:
                            d_log['reordered_entity_types'] = out.entity_types
                        self.ec(msg=f'Edge Case: Reordered entities w/ {pl.fmt(d_log)}', kind='entity-reorder')
                        enms, ets_out = out.entity_names, out.entity_types
                    assert not check.entities_overlapping(sentence=sent, entity_names=enms, ignore_case=True).overlap  # sanity check

                    samples_out.append(NerReadableExample.from_d(sentence=sent, entity_names=enms, entity_types=ets_out))
                assert next(it_pr, None) is None  # sanity check, all spans should be matched
                ret += samples_out
        if logprobs:
            # sort the triples by logprob in ascending order, smallest ones are the most uncertain
            # such uncertain span classifications can be verified and corrected via binary classification
            triples_w_logprob = [(sent, en, et, lp_.avg_logprob) for (sent, en, et), lp_ in triples2logprob.items()]
            logprob.log_n_save_triples_w_logprob(
                triples_w_logprob=triples_w_logprob, entity_types=self.entity_types.copy() + [OTHER_ENTITY_TYPE, NOT_ENTITY_TYPE],
                output_path=os_join(output_path, 'logprobs.json'), logger=self.logger, top_n_log='all'
            )

        self.logger.info(f'Processed {pl.i(len(ret))} samples from {pl.i(len(samples))} samples')
        if len(challenging_samples) > 0:
            # to be corrected via multi-class classification
            output_fnm = os_join(output_path, 'challenging-sentences-&-spans.json')
            dataset.log_n_save_challenging_samples(
                challenging_samples=challenging_samples, output_path=output_fnm, logger=self.logger, sample_map=asdict)
        # there will not be duplicate samples since all sentences are unique, but run dedup just in case
        return self.finish_ner_processing(
            samples=ret, dedup=True, lowercase=lowercase, output_path=output_path, d_log=d_log_count, time=t.end())

    def merge_datasets(
            self, dataset_dir_name: str = None, dataset_dir_name_re_classify: str = None, output_dir_name: str = None, lowercase: bool = None
    ) -> dataset.NerProcessOutput:
        # since all spans for a given sentence is re-classified as a whole, we can override NER-sample-by-NER-sample
        return dataset.merge_datasets(
            dataset_dir_names=[dataset_dir_name, dataset_dir_name_re_classify], output_dir_name=output_dir_name,
            lowercase=lowercase, logger=self.logger)


if __name__ == '__main__':
    from stefutil import sic

    # dnm = 'conll2003-no-misc'
    # dnm = 'wiki-gold-no-misc'
    dnm = 'mit-movie'
    # dnm = 'mit-restaurant'
    # dnm = 'job-stack'

    s_fmt = 'natural-pair-v2'
    seed = 42  # seed for randomness in prompt demo sample construction

    n_demo_et = 1
    da = dict(include_none_samples=dnm not in ['mit-movie', 'mit-restaurant'])
    # m_d = False
    m_d = True
    shf_s = 42  # sample shuffle seed
    lower = None
    if dnm in ['mit-movie', 'mit-restaurant']:
        lower = True
    # multi_cls = False
    # multi_cls = True
    multi_cls = 'group-span'
    n_idt = 5  # taken from the 2nd step, #sentences to identify entity spans in a single prompt
    if multi_cls == 'group-span':
        # n_cls = 3
        n_cls = 5
    else:
        # n_cls = 10
        n_cls = 15
        # n_cls = 20

    dc = False
    # dc = True
    # de = False
    de = True
    # de = 'seeded'
    if dc or de:
        n_list_ = 3
    else:
        n_list_ = 10

    ins = None
    # ins = 'defn'
    # ins = 'schema'
    # cot_ = False
    cot_ = True
    agg = False
    # agg = 'majority'
    if not agg:
        temp = 0  # greedy decoding by default
    else:
        # temp = 1
        temp = 0.7
    # lp = False
    lp = True

    sic(dnm, s_fmt, n_demo_et, da, n_cls)
    sic(temp, agg, cot_)
    gen = TypeGenerator(dataset_name=dnm, sample_format=s_fmt, insert=ins, cot=cot_, multi_class_classify=multi_cls)

    debug = False
    # debug = True
    post = dict()
    if temp != 1:
        post['t'] = temp
    post = pl.pa(post) if post != dict() else None
    if debug:
        post = f'{post}_debug' if post else 'debug'
    out_dnm = gen.meta(n_identify=n_idt, n_classify=n_cls, n_list=n_list_, diverse_context=dc, diverse_entity=de, lowercase=lower, postfix=post)
    sic(out_dnm)
    # raise NotImplementedError

    redo_failed_sents = False
    # redo_failed_sents = True
    if dnm == 'conll2003-no-misc':
        # spans_dir_nm = '23-10-24_Processed-Span-Data_{fmt=n-p2,#l=5,psg=T}'
        if not dc and not de:
            spans_dir_nm = ''
        elif dc and not de:
            spans_dir_nm = ''
        elif not dc and de:
            if de is True:
                spans_dir_nm = ''
            else:
                assert de == 'seeded'
                # spans_dir_nm = '23-12-30_Sentence&Span-Dataset_{fmt=n-p2,#l=3,#a=5,de=s}_{t=0}_debug'
                # spans_dir_nm = '24-01-03_Sentence&Span-Dataset_{fmt=n-p2,#l=3,#a=5,de=s}_{t=0}'
                spans_dir_nm = '24-01-07_Sentence&Span-Dataset_{fmt=n-p2,#l=3,#i=5,de=s}_{t=0}'
        else:
            assert dc and de
            spans_dir_nm = ''
    elif dnm == 'mit-movie':
        assert not dc and de is True
        # spans_dir_nm = '24-01-14_Sentence&Span-Dataset_{fmt=n-p2,#l=3,#i=5,de=T,cot=T,lc=T}_{t=0}_debug'
        spans_dir_nm = '24-01-14_Sentence&Span-Dataset_{fmt=n-p2,#l=3,#i=5,de=T,cot=T,lc=T}_{t=0}'
    else:
        raise NotImplementedError
    if redo_failed_sents:
        assert dnm == 'conll2003-no-misc' and not dc and de == 'seeded'
        spans_dir_nm = '24-01-07_NER-Dataset_{fmt=n-p2,#l=3,#i=5,#c=5,de=s}_{t=0}_more-edge'
        spans_dir_nm = os_join(spans_dir_nm, 'challenging-sentences-&-spans.json')
    sic(spans_dir_nm)

    def check_prompt():
        from src.generate.step_wise.util import get_prompt_fn_w_samples
        out = load_processed(
            dataset_name=dnm, dir_name=spans_dir_nm, kind='span', logger=_logger,
            shuffle=shf_s, span_unique_type=gen.get_span_unique_type())
        samples = out.samples_w_span or out.samples
        _logger.info(f'Total #samples loaded: {pl.i(len(samples))}')
        # sic(samples[:10])
        # raise NotImplementedError

        get_prompt = get_prompt_fn_w_samples(
            get_prompt=gen.get_prompt, samples=samples, group_size=n_cls, prompt_args=dict(n_demo=n_demo_et, demo_args=da, manual_demo=m_d))
        n = 5
        # n = 1
        prettier.print_prompts(prompt=get_prompt, n=n)

    def write_completion():
        md_nm = 'gpt-3.5-turbo-1106'

        # LLM mostly don't regenerate the sentences in prompt,
        # but still set a max-tokens larger than the input sentences, just in case
        # tok_len_sents = 512 if n_cls == 10 else 256  # TODO: increase this count to account for #entity type and #span?
        # max_tok = round(tok_len_sents * 1.5)

        # LLM regenerates the X part in prompt, needs a large max-tokens
        tok_len_grp = 256  # #token allocated for 5 samples
        if multi_cls == 'group-span':
            # tok_len_grp = 512
            tok_len_grp = 768  # need a larger count since each sample is a sentence w/ multiple spans
        tok_len_x = tok_len_grp * math.ceil(n_cls / 5)
        max_tok = round(tok_len_x * (3 if cot_ else 1.5))
        max_tok = min(max_tok, 4096)  # per ChatGPT3.5
        timeout = 60 if n_cls == 10 else 30

        gen.write_completions(
            samples_dir_name=spans_dir_nm, n_classify=n_cls, n_demo=n_demo_et, shuffle_samples=shf_s,
            prompt_args=dict(demo_args=da, manual_demo=m_d),
            # prompt_seed=seed,  # didn't seem to work
            output_dir_nm=out_dnm, model_name=md_nm, max_tokens=max_tok, temperature=temp, timeout=timeout, logprobs=lp
        )

    def process():
        if dnm == 'conll2003-no-misc':
            assert not dc and de == 'seeded'
            if not multi_cls:
                assert not cot_
                dir_nm = '23-12-30_15-23-08_Type-Res_{fmt=n-p2,#l=3,#a=5,de=s}_{t=0}_debug'
            else:
                assert cot_
                if multi_cls is True:
                    dir_nm = '23-12-31_11-38-34_Type-Res_{fmt=n-p2,#l=3,#c=20,de=s}_{t=0}_debug'
                else:
                    assert multi_cls == 'group-span'  # + CoT
                    if not agg:
                        if not lp:
                            # dir_nm = '24-01-01_19-25-50_Type-Res_{fmt=n-p2,#l=3,#c=5,de=s}_{t=0}_debug'

                            # cot demo update, 1 additional CLS false negative: dropped `Broadway`
                            # dir_nm = '24-01-01_20-59-11_Type-Res_{fmt=n-p2,#l=3,#c=5,de=s}_{t=0}_debug'

                            # cot demo update, 3 wrong CLS
                            # dir_nm = '24-01-02_02-38-59_Type-Res_{fmt=n-p2,#l=3,#c=5,de=s}_{t=0}_debug'

                            # cot demo update, 2 wrong CLS
                            # dir_nm = '24-01-02_12-41-55_Type-Res_{fmt=n-p2,#l=3,#c=5,de=s}_{t=0}_debug'
                            dir_nm = '24-01-02_13-01-24_Type-Res_{fmt=n-p2,#l=3,#c=5,de=s}_{t=0}_debug'
                        else:
                            # cot demo update, 2 wrong CLS
                            # dir_nm = '24-01-02_14-33-29_Type-Res_{fmt=n-p2,#l=3,#c=5,de=s}_{t=0}_debug'

                            # dir_nm = '24-01-04_16-16-37_Type-Res_{fmt=n-p2,#l=3,#c=5,de=s}_{t=0}'

                            # dir_nm = '24-01-04_16-16-37_Type-Res_{fmt=n-p2,#l=3,#c=5,de=s}_{t=0}'
                            # failed samples above, re-classified
                            dir_nm = '24-01-07_22-50-32_Type-Res_{fmt=n-p2,#l=3,#i=5,#c=5,de=s}_{t=0}'
                    else:
                        dir_nm = {'1': ''}
        elif dnm == 'mit-movie':
            assert not dc and de is True
            assert multi_cls == 'group-span' and cot_ and not agg
            # after CoT demo tuning, 1-2 wrong CLS
            # dir_nm = '24-01-15_19-45-08_Type-Res_{fmt=n-p2,#l=3,#i=5,#c=5,de=T,lc=T}_{t=0}_debug'
            # dir_nm = '24-01-15_20-05-34_Type-Res_{fmt=n-p2,#l=3,#i=5,#c=5,de=T,cot=T,lc=T}_{t=0}'

            # after CoT demo & task instruction update, 1-2 wrong CLS
            dir_nm = '24-01-16_16-24-32_Type-Res_{fmt=n-p2,#l=3,#i=5,#c=5,de=T,cot=T,lc=T}_{t=0}'
        else:
            raise NotImplementedError
        gen.process_completions(
            samples_dir_name=spans_dir_nm, shuffle_samples=shf_s, logprobs=lp,
            completions_dir_name=dir_nm, output_dir_name=out_dnm, expected_samples_per_completion=n_cls, lowercase=lower)

    def merge():
        assert dnm == 'conll2003-no-misc' and not dc and de == 'seeded'
        dir_nm = '24-01-07_NER-Dataset_{fmt=n-p2,#l=3,#i=5,#c=5,de=s}_{t=0}_more-edge'
        dir_nm_cls = '24-01-07_NER-Dataset_{fmt=n-p2,#l=3,#i=5,#c=5,de=s}_{t=0}_re-cls'

        gen.merge_datasets(dataset_dir_name=dir_nm, dataset_dir_name_re_classify=dir_nm_cls, output_dir_name=f'{out_dnm}_merge')
    # check_prompt()
    # write_completion()
    process()
    # merge()
