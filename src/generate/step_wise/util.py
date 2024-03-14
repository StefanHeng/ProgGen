import json
import logging
import os.path
import random
from os.path import join as os_join
from typing import List, Tuple, Dict, Union, Iterable, Any, Optional, Callable
from dataclasses import dataclass

from stefutil import *
from src.util import *
from src.util.ner_example import *
from src.util.sample_formats import *
from src.util import api as api_util
from src.data_util import *
from src.generate.schemas import Dataset2Schema
from src.generate.sample2reason import Sample2Reasoning
from src.generate.step_wise.util_3_stage import *
from src.generate.step_wise.util_entity_correct import *


__all__ = [
    'STEP_WISE_DNM', 'get_prompt_fn_w_samples',
    'StepWiseGenerator', 'PromptSample', 'AnnotationGenerator',
    'ProcessedSampleOutput', 'load_processed'
]


_logger = get_logger('Step-wise Gen Util')


STEP_WISE_DNM = sconfig('sub-directory-names.step-wise')


PromptSample = Union[NerExample, Dict[str, Any]]


class StepWiseGenerator:
    def __init__(
            self, dataset_name: str = 'conll2003-no-misc', dataset_loader: DatasetLoader = None, sample_format: str = 'natural-pair-v2'
    ):
        ca(dataset_name=dataset_name, sample_format=sample_format)

        self.dataset_name = dataset_name
        self.entity_types = sconfig(f'datasets.{self.dataset_name}.readable-entity-types').copy()
        self.sample_format = sample_format
        self.original_data_format = sample_fmt2original_data_fmt(sample_format=sample_format)
        self.data_format = sample_fmt2data_fmt(sample_format=sample_format)
        self._loader = dataset_loader

        self.dir_args = dict(dataset_name=self.dataset_name, sub_dir=STEP_WISE_DNM)

    @property
    def loader(self) -> DatasetLoader:
        if self._loader is None:
            self._loader = DatasetLoader(dataset_name=self.dataset_name, data_format=self.original_data_format)
        return self._loader


def get_prompt_fn_w_samples(
        get_prompt: Callable = None, samples: List[Any] = None, sample_key: str = 'samples',
        group_size: int = None, prompt_args: Dict[str, Any] = None, n: int = None
) -> Iterable[str]:
    i = 0
    while True:
        i_s, i_e = i * group_size, (i + 1) * group_size
        samples_ = samples[i_s:i_e]
        if len(samples_) > 0:
            yield get_prompt(**{sample_key: samples_}, **(prompt_args or dict()))
        else:
            break
        i += 1
        if n is not None and i >= n:
            break


def get_diff_spans(str1: str, str2: str) -> List[Tuple[str, ...]]:
    """
    :return: List of consecutive spans where `str1` and `str2` differ
    """
    ret = []
    diff = ['', '']
    for c1, c2 in zip(str1, str2):
        if c1 != c2:
            diff[0] += c1
            diff[1] += c2
        else:
            if diff[0]:
                ret.append(tuple(diff))
                diff = ['', '']
    if diff[0]:
        ret.append(tuple(diff))
    return ret


@dataclass
class SplitFromSentenceSamplesOutput:
    split_output: split.SplitSamplesOutput
    sentence_found: bool = None
    has_enum_prefix: bool = None


class AnnotationGenerator(StepWiseGenerator):
    generate_types = ['entity-name', 'entity-type', 'both', 'entity-correction', 'baseline-both']

    def __init__(
            self, generate_type: str = None, sample_type: str = None, processed_type: str = None,
            entity_sep: str = None, entity_pair_map: EntityPairTemplate = None, insert: Optional[Union[bool, str]] = False,
            logger: logging.Logger = None, ec: prettier.EdgeCases = None, cot: bool = False, multi_class_classify: bool = None,
            batched: bool = None, **kwargs
    ):
        super().__init__(**kwargs)
        ca.assert_options(display_name='Sample Generation Type', val=generate_type, options=AnnotationGenerator.generate_types)
        self.generate_type, self.sample_type, self.processed_type = generate_type, None, None
        if self.generate_type == 'entity-name':
            self.sample_type, self.processed_type = sample_type or 'Span', processed_type or 'Sentence&Span'
        elif self.generate_type == 'both':
            self.sample_type, self.processed_type = sample_type or 'Label', processed_type or 'NER'
        elif self.generate_type == 'entity-type':
            self.sample_type, self.processed_type = sample_type or 'Type', processed_type or 'NER'
        elif self.generate_type == 'entity-correction':
            self.sample_type, self.processed_type = sample_type or 'Correction', processed_type or 'NER'
        else:
            assert self.generate_type == 'baseline-both'
            assert not cot
            self.sample_type, self.processed_type = sample_type or 'Few-Shot-Label', processed_type or 'Eval'

        assert self.sample_format == 'natural-pair-v2'
        if generate_type == 'entity-name':
            self.entity_sep = entity_sep or ';'  # separate by `;` to avoid conflict with `,` in entity names
        else:
            self.entity_sep = entity_sep or get_default_entity_sep(sample_format=self.sample_format)

        self.entity_pair_map = None
        if generate_type in ['both', 'entity-type', 'baseline-both']:
            self.entity_pair_map = entity_pair_map or get_default_entity_pair_map(sample_format=self.sample_format)

        self.insert, self.d2s = insert, None
        if self.insert:
            self.d2s = Dataset2Schema(dataset_name=self.dataset_name, schema_type=self.insert)

        self.logger = logger or _logger
        self.ec: prettier.EdgeCases = ec or prettier.EdgeCases(logger=self.logger)

        self.cot, self.s2r = cot, None
        if self.cot and generate_type == 'both':
            self.s2r = Sample2Reasoning(dataset_name=self.dataset_name)

        self.entity_enclose_open, self.entity_enclose_close, self.s2t = None, None, None
        self.multi_class_classify, self.t2gt = None, None
        self.s2ns, self.s2sr = None, None
        if generate_type == 'entity-name':
            self.s2ns = Sentence2NegativeSpans(dataset_name=self.dataset_name)
            if self.cot:
                self.s2sr = Sentence2SpansReasoning(dataset_name=self.dataset_name)
        elif generate_type == 'entity-type':
            # for generating entity types, 3rd step in 3-stage generation; Double braces should be rare in natural language
            self.entity_enclose_open, self.entity_enclose_close = '{{', '}}'
            mcc = multi_class_classify
            if mcc:
                if mcc is True:
                    mcc = 'independent-span'
                else:
                    ca.assert_options(display_name='Multi-class Classification Type', val=mcc, options=['independent-span', 'group-span'])
            else:
                mcc = False
            self.multi_class_classify = mcc
            self.s2t = Sample2TypeClsTuples(dataset_name=self.dataset_name, cot=self.cot, multi_class_classify=self.multi_class_classify)
            if self.multi_class_classify:
                self.t2gt = Type2GenType(dataset_name=self.dataset_name)
        self.batched = batched if batched is not None else True

    def meta(self, n_annotate: int = None, postfix: str = None, **kwargs) -> str:
        return dataset_meta(sample_format=self.sample_format, cot=self.cot, n_annotate=n_annotate, postfix=postfix, **kwargs)

    def get_demo_instruction(self, n_demo: int = None):
        assert n_demo is not None
        # return 'Here are some examples. Please follow this format.\n\nExamples:'

        if self.generate_type in ['both', 'baseline-both']:
            # explicitly these are example **annotations**
            if self.cot:
                # annot = 'annotations and explanations'
                annot = 'reasonings and annotations'
            else:
                annot = 'annotations'
            ret = f'Here are some example {annot} for your reference. Please follow this format.'
        elif self.generate_type == 'entity-name':
            if self.cot and self.dataset_name != 'mit-movie':
                raise NotImplementedError

            # ret = f'Here are some example entity identifications for your reference. Please follow this format.'
            if self.dataset_name == 'conll2003-no-misc':
                ret = f'Here are some example entity identifications for your reference. Please follow this format.'
            elif self.dataset_name == 'mit-movie':
                # ret = f'Here are some example potential span identifications for your reference. Please follow this format.'

                annot = 'analyses and identifications' if self.cot else 'identifications'
                ret = f'Here are some example keyword {annot} for your reference. Please follow this format.'
            else:
                raise NotImplementedError
        else:
            assert self.generate_type in ['entity-type', 'entity-correction']
            if self.cot:
                annot = 'analyses and classifications'
            else:
                annot = 'classifications'
            ret = f'Here are some example span {annot} for your reference. Please follow this format.'
        return f'{ret}\n\nExamples:'

    def get_instruction(self, **kwargs):
        raise NotImplementedError

    def triple2x(self, x_prefix: str = None, sentence: str = None, entity_span: str = None, entity_type: str = None) -> str:
        """
        Template the X part for entity type classification
            Intended for step 3 of 3-stage generation

        For binary CLS, would be the question
        For multi-class CLS, the entity span
        """
        pref_x_, sent_, en_ = x_prefix.lower(), edit.enclose_in_quote(sentence), edit.enclose_in_quote(entity_span)
        if not self.multi_class_classify:
            return f'In the {pref_x_} {sent_}, is the enclosed span of text {en_} a named {entity_type} entity?'
        else:
            return f'{en_} in the {pref_x_} {sent_}'

    def _highlight_span_in_sentence(
            self, sentence: str = None, span: str = None, highlight_span: str = None, **kwargs
    ) -> str:
        if self.generate_type != 'entity-correction':
            raise ValueError(f'Highlighting span is only supported for entity correction, not {pl.i(self.generate_type)}')

        if highlight_span == 'braces':
            pref, post = '{{', '}}'
        else:
            assert highlight_span == 'brackets'
            pref, post = '[', ']'
        # sanity check sentence don't contain double braces already, since this is used to highlight span
        assert pref not in sentence and post not in sentence
        return prettier.highlight_span_in_sentence(sentence=sentence, span=span, ec=self.ec, pref=pref, post=post, **kwargs)

    def _sample2sample_str(
            self, sample: PromptSample, highlight_span: str = None
    ) -> Union[str, List[str], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Convert a sample to a string, either a demo example or a sample to annotate, intended for `natural-pair-v2` format
        """
        assert self.sample_format == 'natural-pair-v2'

        d_dset = sconfig(f'datasets.{self.dataset_name}')
        pref_x, pref_y = d_dset['x-name'], d_dset['y-name']

        if isinstance(sample, (NerExample, EntityCorrectionSample)):  # demo example ground truth
            if isinstance(sample, NerReadableExample):
                sent, enms, ets = sample.sentence, sample.entity_names, sample.entity_types

                if self.generate_type in ['entity-name', 'both', 'baseline-both']:
                    if self.generate_type == 'entity-name':
                        # add negative spans for a broad coverage of potential entity names
                        enms = list(enms)
                        # sanity check each span appears once
                        assert all(sent.count(enm) == 1 for enm in enms)
                        # sanity check entity spans in occurrence order
                        assert enms == sorted(enms, key=lambda e: sent.index(e))

                        # add negative spans
                        if self.dataset_name == 'conll2003-no-misc':
                            add_negative_spans = True
                        elif self.dataset_name == 'mit-movie':
                            add_negative_spans = False
                        else:
                            raise NotImplementedError

                        if add_negative_spans:
                            neg_enms = self.s2ns(sentence=sent)
                            if len(neg_enms) > 0:
                                enms = enms + neg_enms
                                # sort by occurrence order
                                enms = sorted(enms, key=lambda e: sent.index(e))
                        # pref_y = 'Possible Entity Spans'
                        # pref_y = 'Possible Keyword Spans'

                    if len(enms) == 0:
                        entities = ''  # non-entity will be empty string inside brackets
                    else:
                        if self.generate_type == 'entity-name':  # for 2nd step in 3-stage generation
                            entities = f'{self.entity_sep} '.join(enms)  # ground truth is just the entity names
                        else:
                            assert self.generate_type in ['both', 'baseline-both']  # for 2nd step in 2-stage generation
                            assert len(enms) == len(ets)
                            entities = f'{self.entity_sep} '.join(self.entity_pair_map(enm, et) for enm, et in zip(enms, ets))
                    parts = [f'{pref_x}: {edit.enclose_in_quote(sent)}']

                    if self.generate_type == 'both' and self.cot:
                        parts.append(self.s2r(sample))

                    # if self.generate_type == 'entity-name' and self.cot:
                    #     reason = f'Analysis: {self.s2sr(sample)}'
                    #     parts.append(reason)
                    #     pref_y = 'Keywords'

                    y = f'{pref_y}: [{entities}]'
                    if self.generate_type == 'entity-name' and self.cot:
                        # pref_y = 'Keywords'
                        pref_y = 'Likely Keywords'
                        reason = self.s2sr(sample)
                        y = f'{pref_y}: {reason} Likely keywords are [{entities}].'
                    parts.append(y)
                    if self.cot and self.generate_type == 'both':
                        # add extra line break for consistent Markdown formatting
                        sep = '\n\n'
                    else:
                        sep = '\n'
                    return sep.join(parts)
                else:  # incomplete samples to annotate
                    assert self.generate_type == 'entity-type'  # for 3rd step in 3-stage generation

                    # sanity check sentence don't contain entity enclosing punctuations so that no ambiguity
                    assert self.entity_enclose_open not in sent and self.entity_enclose_close not in sent

                    out = self.s2t(sample=sample)

                    if self.multi_class_classify and self.multi_class_classify == 'group-span':
                        # group all spans for the same sentence together
                        sent = f'{pref_x}: {edit.enclose_in_quote(sent)}'
                        pairs = []
                        for grp in out.tuples:
                            en, et = grp.entity_name, grp.entity_type
                            x = edit.enclose_in_quote(en)

                            y = self.t2gt(entity_type=et, output_kind='demo')
                            if self.cot:
                                reason = grp.reason
                                y = f'{reason} The span is {y}.'  # may flow more naturally
                                # y = f'{reason} The label is {y}.'
                            pairs.append((f'Text Span: {x}', f'Label: {y}'))

                        # try shuffle all spans since potential multiple occurrences of the same span will have the same label => order don't matter
                        #   1 response didn't complete properly
                        # random.shuffle(pairs)
                        return dict(sentence=sent, pairs=pairs)

                    ret = []
                    for grp in out.tuples:
                        en, et, is_typed_entity = grp.entity_name, grp.entity_type, grp.is_typed_entity
                        assert sent.count(en) == 1  # sanity check the span appears in the sentence once and only once
                        # enclose the entity name with double braces
                        sent_ = sent.replace(en, f'{self.entity_enclose_open}{en}{self.entity_enclose_close}')

                        if not self.multi_class_classify:
                            label = 'Yes' if is_typed_entity else 'No'
                            if self.cot:
                                reason = grp.reason
                                label = f'{reason} The answer is {label}.'

                            # a more structured format, doesn't improve CLS acc
                            # parts = [f'{pref_x}: {sent_}', f'Entity Type: {et}', f'Label: {label}']
                            # ret.append('\n'.join(parts))

                            # a more natural QA format
                            q = self.triple2x(x_prefix=pref_x, sentence=sent_, entity_span=en, entity_type=et)
                            ret.append(f'Question: {q}\nAnswer: {label}')
                        else:
                            label = self.t2gt(entity_type=et, output_kind='demo')
                            if self.cot:
                                reason = grp.reason
                                label = f'{reason} The label is {label}.'
                                # label = f'{reason} The span is {label}.'  # may flow more naturally

                            # ret.append(f'{pref_x}: {sent_}\nLabel: {label}')

                            # a new format that highlights span classification, **in context of** sentence
                            x = self.triple2x(x_prefix=pref_x, sentence=sent_, entity_span=en, entity_type=et)
                            ret.append(f'Text Span: {x}\nLabel: {label}')
                    return ret
            else:
                assert isinstance(sample, EntityCorrectionSample)
                assert self.generate_type == 'entity-correction'
                sent, span, lb = sample.sentence, sample.entity_span, sample.label

                if highlight_span:
                    sent = self._highlight_span_in_sentence(sentence=sent, span=span, highlight_span=highlight_span)

                sent = f'{pref_x}: {edit.enclose_in_quote(sent)}'
                span = f'Text Span: {edit.enclose_in_quote(span)}'
                if not self.cot:
                    # label = f'Label: {lb.to_readable()}'
                    tp = 'named entity'
                    # tp = 'term'
                    # desc = False
                    desc = True
                    label = f'Label: {sample.to_readable_label(element_type=tp, with_desc=desc)}'
                else:
                    raise NotImplementedError
                return f'{sent}\n{span}\n{label}'
        else:  # sample to annotate
            assert isinstance(sample, dict)
            sent = sample['sentence']
            if self.generate_type in ['entity-name', 'both', 'baseline-both']:
                return f'{pref_x}: {edit.enclose_in_quote(sent)}'
            elif self.generate_type == 'entity-correction':
                # sic(sample)
                enm = sample.get('entity_span', sample.get('span'))  # an entity annotation triple
                assert enm is not None

                if highlight_span:
                    idx, idx_sup = sample.get('index'), sample.get('index_super')
                    sent = self._highlight_span_in_sentence(
                        sentence=sent, span=enm, highlight_span=highlight_span, span_index=idx, span_index_super=idx_sup)

                sent = f'{pref_x}: {edit.enclose_in_quote(sent)}'
                span = f'Text Span: {edit.enclose_in_quote(enm)}'
                return f'{sent}\n{span}'
            else:
                assert self.generate_type == 'entity-type'

                if 'entity_spans' in sample:  # populate all possible entity names and types myself
                    ens = sample['entity_spans']
                    assert self.entity_enclose_open not in sent and self.entity_enclose_close not in sent  # sanity check

                    if self.multi_class_classify and self.multi_class_classify == 'group-span':
                        sent = f'{pref_x}: {edit.enclose_in_quote(sent)}'
                        # random.shuffle(ens)
                        spans = [f'Text Span: {edit.enclose_in_quote(en)}' for en in ens]
                        return dict(sentence=sent, spans=spans)

                    ret = []
                    for en in ens:
                        assert sent.count(en) == 1
                        sent_ = sent.replace(en, f'{self.entity_enclose_open}{en}{self.entity_enclose_close}')
                        # pref_x_, sent_, en = pref_x.lower(), enclose_in_quote(sent_), enclose_in_quote(en)
                        if not self.multi_class_classify:
                            for et in self.entity_types:
                                q = self.triple2x(x_prefix=pref_x, sentence=sent_, entity_span=en, entity_type=et)
                                ret.append(f'Question: {q}')

                            # to save cost, write (sentence, span) pair once, followed by each entity type grouped; this doesn't improve CLS acc?
                            # prefix = f'In the {pref_x_} {sent_}, is the enclosed span of text {en}'
                            # classes = [f'a named {et} entity?' for et in self.entity_types]
                            # ret.append(dict(prefix=prefix, classes=classes))
                        else:
                            # ret.append(f'{pref_x}: {sent_}')
                            x = self.triple2x(x_prefix=pref_x, sentence=sent_, entity_span=en)
                            ret.append(f'Text Span: {x}')
                    return ret
                else:
                    assert 'entity_span' in sample
                    en = sample['entity_span']
                    assert sent.count(en) == 1
                    sent = sent.replace(en, f'{self.entity_enclose_open}{en}{self.entity_enclose_close}')

                    if not self.multi_class_classify:  # (sentence, span, entity type) triple is provided
                        assert 'entity_type' in sample

                        et = sample['entity_type']
                        q = self.triple2x(x_prefix=pref_x, sentence=sent, entity_span=en, entity_type=et)
                        return f'Question: {q}'
                    else:  # (sentence, span) pair is provided
                        assert 'entity_type' not in sample
                        x = self.triple2x(x_prefix=pref_x, sentence=sent, entity_span=en)
                        return f'Text Span: {x}'

    def _join_samples(self, samples: Iterable[PromptSample], sample2str_args: Dict[str, Any] = None) -> str:
        # nest_sep = '\t'
        # nest_sep = ' ' * 4  # for Markdown formatting
        nest_sep = ''  # no indentation

        def dict2group(sample: Dict[str, Any], index: int) -> str:
            """
            for grouping together sub-items with the correct sub-indices
                # add enum index to the prefix and then sub-indices for each sub item
            """
            if not self.multi_class_classify:  # binary cls w/ entity types
                prefix, items = sample['prefix'], sample['classes']
            else:  # multi-class cls w/ entity spans
                assert self.multi_class_classify == 'group-span'
                prefix, items = sample['sentence'], sample.get('pairs', sample.get('spans'))
                assert items and len(items) > 0
            ret = [prefix]
            for j, elm in enumerate(items, start=1):
                # idx = f'{index}.{j} '

                # for Markdown formatting
                # idx = f'{j}. '
                # idx = '- '

                idx = ''
                if isinstance(elm, str):
                    ret.append(f'{nest_sep}{idx}{elm}')
                else:
                    assert isinstance(elm, tuple)
                    x, y = elm
                    ret.append(f'{nest_sep}{idx}{x}\n{nest_sep}{nest_sep}{y}')
            return '\n'.join(ret)

        # ensure consistent formatting between demo and samples to annotate
        samples = [self._sample2sample_str(sample=sample, **(sample2str_args or dict())) for sample in samples]
        assert len(samples) > 0  # sanity check
        if any(isinstance(sample, list) for sample in samples):
            assert self.generate_type == 'entity-type'  # sanity check
            assert all(isinstance(sample, list) for sample in samples)
            samples = sum(samples, start=[])
            if all(isinstance(sample, str) for sample in samples):  # binary cls samples
                random.shuffle(samples)  # further shuffle within each group
            else:
                assert all(isinstance(sample, dict) for sample in samples)
                sample: Dict
                assert not self.multi_class_classify
                samples = [dict2group(sample=sample, index=i) for i, sample in enumerate(samples, start=1)]
        else:
            if all(isinstance(sample, dict) for sample in samples):
                assert self.multi_class_classify == 'group-span'
                samples = [dict2group(sample=sample, index=i) for i, sample in enumerate(samples, start=1)]
            else:
                assert all(isinstance(sample, str) for sample in samples)  # sanity check

        return '\n\n'.join(f'{i}. {sample}' for i, sample in enumerate(samples, start=1))

    def get_demo_examples(
            self, n_demo: int = 5, demo_type: str = 'n-shot', demo_args: Dict[str, Any] = None, generator: Union[random.Random, int] = None,
            add_demos: List[NerReadableExample] = None, examples: List[Any] = None, sample2str_args: Dict[str, Any] = None,
    ):
        if examples is not None:  # override loading examples from the dataset, intended for entity correction
            assert isinstance(examples, list) and all(isinstance(example, (EntityCorrectionSample, dict)) for example in examples)
        else:
            _demo_args = dict(n_demo=n_demo, demo_type=demo_type)  # no shuffling, will happen later in this call
            _demo_args.update((demo_args or dict()))
            examples = self.loader.get_few_demo_samples(**_demo_args)
        if add_demos:
            assert self.sample_format == 'natural-pair-v2' and all(isinstance(demo, NerReadableExample) for demo in add_demos)
            examples = examples + add_demos

        gen = get_random_generator(generator=generator)
        gen.shuffle(examples)
        return self._join_samples(samples=examples, sample2str_args=sample2str_args)

    def get_prompt(
            self, samples: Iterable[Dict[str, Any]], n_demo: int = 5, demo_type: str = 'n-shot', demo_args: Dict[str, Any] = None,
            generator: Union[random.Random, int] = None, add_demos: List[NerReadableExample] = None, demo_examples: List[Any] = None,
            instruction_args: Dict[str, Any] = None, sample2str_args: Dict[str, Any] = None
    ) -> str:
        samples = list(samples)
        n_annot = len(samples)
        if self.generate_type in ['both', 'entity-name']:
            instr_args = dict(n_annotate=n_annot)
        elif self.generate_type in ['entity-type', 'entity-correction']:
            instr_args = dict(n_classify=n_annot)
        else:
            assert self.generate_type == 'baseline-both'
            instr_args = dict()
        instr_args.update((instruction_args or dict()))
        ret = self.get_instruction(**instr_args)
        if n_demo is not None:
            ret = f'{ret}\n{self.get_demo_instruction(n_demo=n_demo)}'

            generator = get_random_generator(generator=generator)
            examples = self.get_demo_examples(
                n_demo=n_demo, demo_type=demo_type, demo_args=demo_args, generator=generator, add_demos=add_demos, examples=demo_examples,
                sample2str_args=sample2str_args)
            ret = f'{ret}\n\n{examples}\n\n\n'
            sep = '---'  # not too much as it would add to token cost
            ret = f'{ret}{sep}\n\n'

        x_key = 'x-name' if n_annot == 1 else 'x-name-pl'
        x_name = sconfig(f'datasets.{self.dataset_name}.{x_key}').lower()
        if self.generate_type == 'entity-correction':
            x_name = 'span' if n_annot == 1 else 'spans'
            x_name = f'{x_name} of text'

        if self.generate_type in ['both', 'baseline-both']:
            act = 'analyze and annotate' if self.cot else 'annotate'
        elif self.generate_type == 'entity-name':
            if self.cot and self.dataset_name != 'mit-movie':
                raise NotImplementedError

            # act = 'identify named entities in'
            if self.dataset_name == 'conll2003-no-misc':
                act = 'identify named entities in'
            elif self.dataset_name == 'mit-movie':
                # act = 'identify potential spans in'

                act = 'analyze and identify' if self.cot else 'identify'
                act = f'{act} potential spans in'
            else:
                raise NotImplementedError
        elif self.generate_type == 'entity-type':
            if self.cot:
                act = 'analyze and classify spans of text in'
            else:
                act = 'classify spans of text in'
        else:
            assert self.generate_type == 'entity-correction'
            act = 'analyze and classify' if self.cot else 'classify'
        if n_annot == 1:
            pref = f'Please {act} the following {x_name}:'
        else:
            pref = f'Please {act} the following {n_annot} {x_name}:'
        ret = f'{ret}\n{pref}\n\n'
        if n_annot == 1:
            ret += self._sample2sample_str(sample=samples[0], **(sample2str_args or dict()))
        else:
            ret += self._join_samples(samples=samples, sample2str_args=sample2str_args)

        if self.generate_type == 'both' and self.cot:
            ret = f"{ret}\n\n---\n\n"
            # ret += "Let's think step by step."
            ret += "Show your reasoning. Let's think step by step."
        return ret

    def finish_ner_processing(self, samples: List[NerReadableExample], **kwargs) -> dataset.NerProcessOutput:
        assert self.processed_type == 'NER'
        return dataset.finish_ner_processing(
            samples=samples, ec=self.ec, logger=self.logger, dataset_name=self.dataset_name, data_format=self.data_format,
            entity_types=self.entity_types, **kwargs
        )

    def check_sentence_diff(
            self, sentence_in_prompt: str = None, sentence_in_response: str = None, drop_diff_threshold: int = 2, log: bool = True
    ) -> bool:
        """
        Check if the sentence in the response differs from the sentence in the prompt

        :return: a flag for if the difference is too large
        """
        from Levenshtein import distance as edit_dist  # lazy import to save time
        ori_sent, res_sent = sentence_in_prompt, sentence_in_response
        ori_sent_ = edit.drop_enclosing_quotes(ori_sent.lower()).strip()
        res_sent_ = edit.drop_enclosing_quotes(res_sent.lower()).strip()
        dist = edit_dist(ori_sent_, res_sent_)

        drop = False
        if dist > 0:
            # lenient on different ways of spelling; in worse case, missing 2 double quotes
            drop = dist > drop_diff_threshold

            if log:
                diff = get_diff_spans(ori_sent_, res_sent_)
                d_log = dict(original_sentence=ori_sent, sentence_in_response=res_sent, diff=diff)

                msg = f'Sentence from response differs from original sentence'
                if drop:
                    msg = f'{msg} and dropped'
                msg = f'{msg} w/ {pl.i(d_log, indent=1)}'
                self.ec(msg=msg, kind='sentence-mismatch-drop' if drop else 'sentence-mismatch', failed=drop, disable_std_log=not drop)
        return drop

    def _split_from_sentence_samples(
            self, completion: str = None, pattern_w_sentence: patterns.Patterns = None, pattern_wo_sentence: patterns.Patterns = None,
            is_last_group: bool = False, edge_split_args: Dict[str, Any] = None, **split_args
    ) -> SplitFromSentenceSamplesOutput:
        """
        Common code for extracting samples from completion, where sentences (X) are given in prompt
        LLM may or may not generate the sentence, and enum index in response
        """
        split_args_ = dict(completion=completion, ec=self.ec, has_cot=self.cot)
        split_args_.update((split_args or dict()))
        sent_found = True
        try:
            has_enum = True
            out = split.split_samples(pattern=pattern_w_sentence, **split_args_, has_enum_prefix=has_enum)
        except ValueError as e:  # for edge case
            sent_found = False

            # last group may not have prefix if just 1 sample
            has_enum = completions.completion_has_enum_prefix(completion=completion) if is_last_group else True
            split_args_.update(edge_split_args or dict())
            out = split.split_samples(pattern=pattern_wo_sentence, **split_args_, has_enum_prefix=has_enum, return_match_map=True)
            # msg = f'Sentences missing, only entity names found in completion w/ {pl.i(completion)}'
            msg = f'Sentences in prompt not re-generated in LLM completion w/ {pl.i(completion)}'
            self.ec(msg=msg, kind='sentence-not-re-generated', disable_std_log=True)
        return SplitFromSentenceSamplesOutput(split_output=out, sentence_found=sent_found, has_enum_prefix=has_enum)


@dataclass
class ProcessedSampleOutput:
    samples: List[Any] = None
    samples_w_span: List[Any] = None
    samples_wo_span: List[Any] = None


def load_processed(
        dataset_name: str = 'conll2003-no-misc', dir_name: str = None, kind: str = 'sentence', logger: logging.Logger = None,
        shuffle: Union[bool, int] = False, seed_delta: int = None, ori: Union[bool, str] = None, span_unique_type: str = 'sentence',
        first: int = None, data_dir_args: Dict[str, Any] = None
) -> ProcessedSampleOutput:
    """
    Load processed samples from prior steps

    :param dataset_name: dataset name
    :param dir_name: directory name or path to the samples file
    :param kind: one of [`sentence`, `span`]
    :param logger: logger
    :param shuffle: If true, the samples are shuffled
        If an integer, the samples are shuffled with the given random seed
        For sentences, shuffle sentences
        For spans, shuffle each (sentence, span) pair independently
    :param seed_delta: If given and a random seed is used, add this to the seed
    :param ori: If true, load the sentences from the original dataset
    :param span_unique_type: How each sample in the returned list is uniquely identified, one of [`sentence`, `span`, `type`]
        If `sentence`, each sample is a (sentence, list-of-spans) pair
        If `span`, each sample is a (sentence, span) pair
        If `type`, each sample is a (sentence, span, type) triple
    :param first: If given, only load the first `first` samples from the file
    :param data_dir_args: additional arguments for `dataset_name2data_dir`
    """
    ca.assert_options(display_name='Processed Sample Kind', val=kind, options=['sentence', 'span'])
    if ori:
        assert dir_name is None
        if kind == 'sentence':
            path = os_join(pu.proj_path, 'original-dataset', dataset_name)
            fnm = ori if isinstance(ori, str) else 'sentences.json'
        else:  # `span`
            raise NotImplementedError
    else:
        assert dir_name is not None

        dir_args = dict(dataset_name=dataset_name, sub_dir=STEP_WISE_DNM, input_dir=dir_name)
        if data_dir_args is not None:
            dir_args.update(data_dir_args)
        path = dataset_name2data_dir(**dir_args).path
        if os.path.isfile(path):  # a file path is given
            path, fnm = os.path.dirname(path), os.path.basename(path)
        else:
            fnm = 'sentences.json' if kind == 'sentence' else 'sentences-&-spans.json'
    path = os_join(path, fnm)

    kd_str = f'{pl.i(kind.capitalize())} samples'
    if logger:
        d_log = dict(dataset_name=dataset_name, dir_name=dir_name, path=path)
        logger.info(f'Loading processed {kd_str} w/ {pl.i(d_log)}... ')
    with open(path, 'r') as f:
        ret = json.load(f)
    if not isinstance(ret, list):
        assert isinstance(ret, dict) and ('samples' in ret or 'sentences' in ret)
        ret = ret.get('samples', ret.get('sentences', None))
        assert ret is not None
    if first is not None:
        ret = ret[:first]

    ret_w_span, ret_wo_span = None, None
    if shuffle:
        use_seed = isinstance(shuffle, int) and not isinstance(shuffle, bool)
        if use_seed:
            seed = shuffle
            if seed_delta:
                seed += seed_delta
            random.seed(seed)

        if kind == 'span':
            if span_unique_type in ['span', 'type']:
                ret = [[dict(sentence=sample['sentence'], entity_span=span) for span in sample['entity_spans']] for sample in ret]
                ret = sum(ret, start=[])

                if span_unique_type == 'type':  # further expand each sample to each possible entity type
                    ets = sconfig(f'datasets.{dataset_name}.readable-entity-types')
                    ret = [[dict(sentence=sample['sentence'], entity_span=sample['entity_span'], entity_type=et) for et in ets] for sample in ret]
                    ret = sum(ret, start=[])
        shuf_idxs = list(range(len(ret)))
        random.shuffle(shuf_idxs)
        ret = [ret[i] for i in shuf_idxs]

        if use_seed:
            random.seed()  # reset seed
        if logger:
            d_log = dict(shuffle=shuffle, seed_delta=seed_delta, sent_idxs=shuf_idxs)
            logger.info(f'{kd_str} shuffled w/ {pl.i(d_log, indent=1)}')

        if kind == 'span' and span_unique_type == 'sentence' and any(len(sample['entity_spans']) == 0 for sample in ret):
            # if some samples don't have any entity span, drop these by providing partitions
            # Additionally return `span` samples that have at least one entity span
            #   Intended for `span` kind, relevant only when `span_unique_type` is `sentence` by construction
            ret_w_span = [sample for sample in ret if len(sample['entity_spans']) > 0]
            ret_wo_span = [sample for sample in ret if len(sample['entity_spans']) == 0]
            logger.info(f'Kept {pl.i(len(ret_w_span))} (sentence, spans)-pair samples w/ at least one span '
                        f'from {pl.i(len(ret))} samples')

    elif span_unique_type is not None:
        d_log = dict(kind=kind, shuffle=shuffle)
        raise ValueError(f'{pl.i("span_unique_type")} intended for shuffling span samples, but got {pl.i(d_log)}')
    return ProcessedSampleOutput(samples=ret, samples_w_span=ret_w_span, samples_wo_span=ret_wo_span)
