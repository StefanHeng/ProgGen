import re
import math
from typing import Dict, Tuple, List, Union, Optional, Any
from collections import defaultdict, Counter
from dataclasses import astuple

from stefutil import pl, ca, Timer, ordinal
from src.util import sconfig, patterns
from src.util.ner_example import NerReadableExample
from src.data_util import dataset, completions, logprob
import src.data_util.sample_split as split
from src.generate import Np2Transform, schemas
from src.generate.step_wise.generate_from_sentence import FromSentenceGenerator


__all__ = ['SamplesAggregator', 'EntityAnnotationGenerator']


class SamplesAggregator:
    """
    Aggregates list of annotations on the same sentence into one, if possible
    """
    def __init__(self, scheme: str = 'same'):
        ca.assert_options('Sample Aggregation Scheme', scheme, ['same', 'majority'])
        self.scheme = scheme

    def __call__(self, samples: List[NerReadableExample]) -> Optional[NerReadableExample]:
        assert len(samples) > 1 and all(s.sentence == samples[0].sentence for s in samples[1:])  # sanity check
        if self.scheme == 'same':
            if all(s == samples[0] for s in samples[1:]):
                return samples[0]
        elif self.scheme == 'majority':
            c = Counter(samples)  # get counts of each annotation
            if len(c) == 1:  # all annotations are the same
                return samples[0]
            else:
                (maj, c1), (maj2nd, c2) = c.most_common(2)
                # check majority is unique
                if c1 > c2:
                    return maj


# ad-hoc check for markdown bullet, intended for checking if reasoning is present in sample
# e.g. `- <reasoning 1>`
pattern_bullet = re.compile(r'^(\s*)- (?P<reasoning>.*)$', re.IGNORECASE | re.MULTILINE)


def sample_has_reasoning(sample: str) -> bool:
    return bool(patterns.find_match(text=sample, pattern=pattern_bullet))


class EntityAnnotationGenerator(FromSentenceGenerator):
    """
    Merge 2nd and 3rd steps into one: generate entities and categories together given a sentence
    """
    def __init__(self, allowed_entity_types: Union[bool, List[str]] = None, **kwargs):
        gen_type = kwargs.pop('generate_type', 'both')
        super().__init__(generate_type=gen_type, **kwargs)

        d_dset = sconfig(f'datasets.{self.dataset_name}')
        pref_x, pref_y = d_dset['x-decoded-name'], d_dset['y-decoded-name']
        pref_x, pref_y = patterns.options2re_options(options=pref_x), patterns.options2re_options(options=pref_y)
        if not self.cot:
            # for extracting each annotated sample
            # e.g.
            # 1. "I am a sentence."
            # Entity Names: [I, sentence]
            # edge case, newline character after sentence prefix
            # TODO: double quotes should be matched at the same time
            self.pattern_search = [re.compile(
                rf'(?P<idx>\d+)\. {pref_x}: (\n)?[\'"]?(?P<sentence>.*)[\'"]?\n(\s)*?{pref_y}: (?P<entities>.*)\n',
                re.IGNORECASE)]
            # for edge case, Sentence missing, only Entity Names
            self.pattern_search_edge = [
                re.compile(rf'(?P<idx>\d+)\. (\n)?({pref_y}: )?(?P<entities>.*)\n', re.IGNORECASE),
                # even worse, just 1 sample to complete, LLM doesn't generate a number
                re.compile(rf'{pref_y}: (?P<entities>.*)\n', re.IGNORECASE)]
        else:
            # with multi-line reasoning
            # e.g.
            # 1. Sentence: "a sentence"
            # Reasoning:
            # - reason 1
            # - reason 2
            # ...
            # Entity Names: [I, sentence]
            # match the smallest possible block
            # edge case: a reasoning may not be generated
            self.pattern_search = [
                # sentence should be on the same line
                re.compile(rf'(?P<idx>\d+)\. ({pref_x}: )?"(?P<sentence>[^\n]+)"\n((\n)?(Reasoning:\n)?(?P<reasoning>.*?)\n(\s*\n)?({pref_y}: (?P<entities>.*?)|- None)|- None\.)\n', re.IGNORECASE | re.DOTALL),
                # only differ in quotes
                re.compile(rf'(?P<idx>\d+)\. ({pref_x}: )?\'(?P<sentence>[^\n]+)\'\n((\n)?(Reasoning:\n)?(?P<reasoning>.*?)\n(\s*\n)?({pref_y}: (?P<entities>.*?)|- None)|- None\.)\n', re.IGNORECASE | re.DOTALL),
                re.compile(rf'(?P<idx>\d+)\. ({pref_x}: )?(?P<sentence>[^\n]+)\n((\n)?(Reasoning:\n)?(?P<reasoning>.*?)\n(\s*\n)?({pref_y}: (?P<entities>.*?)|- None)|- None\.)\n', re.IGNORECASE | re.DOTALL),
                # re.compile(
                #     r'(?P<idx>\d+)\. (\n)?[\'"]?(?P<sentence>.*?)[\'"]?\n((\n)?(Reasoning:\n)?(?P<reasoning>.*?)\n(\s*\n)?((Entity Names|Named Entities): (?P<entities>.*?)|- None)|- None\.)\n',
                #     re.IGNORECASE | re.DOTALL
                # ),
                # edge case, no reasoning and no entities found
                # e.g. 9. Sentence: 'sentence'
                # - None.
                # re.compile(r'(?P<idx>\d+)\. Sentence: (\n)?[\'"]?(?P<sentence>.*?)[\'"]?\n(\n)?- None(\.)?\n', re.IGNORECASE)
            ]
            # get the sentence & entities rows
            self.pattern_extract_sent = [
                re.compile(rf'^(?P<idx>\d+)\. ({pref_x}: )?"(?P<sentence>.*?)"\n', re.IGNORECASE),
                re.compile(rf'^(?P<idx>\d+)\. ({pref_x}: )?\'(?P<sentence>.*?)\'\n', re.IGNORECASE),
                re.compile(rf'^(?P<idx>\d+)\. ({pref_x}: )?(?P<sentence>.*?)\n', re.IGNORECASE)]
            self.pattern_extract_entities = [
                re.compile(rf'{pref_y}: (?P<entities>.*)$', re.IGNORECASE),
                # edge case, e.g. `- None.`
                re.compile(rf'- None(\.)?$', re.IGNORECASE)]

            # sentence missing
            self.pattern_search_edge = [re.compile(
                r'(?P<idx>\d+)\.( )?\n((\n)?(Reasoning:\n)?(?P<reasoning>.*?)\n(\s*)?(\n)?({pref_y}: (?P<entities>.*?)|- None)|- None\.)\n',
                re.IGNORECASE | re.DOTALL)]

        # taken from `generate_dataset::Completion2Samples`
        # `?` inside sent needed to allow matching trailing period
        self.pattern_sent = [
            re.compile(rf'^(?P<idx>\d+)\. ({pref_x}: )?"(?P<sent>.*?)"$', re.IGNORECASE),
            re.compile(rf'^(?P<idx>\d+)\. ({pref_x}: )?\'(?P<sent>.*?)\'$', re.IGNORECASE),
            re.compile(rf'^(?P<idx>\d+)\. ({pref_x}: )?(?P<sent>.*?)$', re.IGNORECASE)]
        self.pattern_entities = [
            # edge case, trailing period; white spaces inside brackets
            re.compile(rf'^{pref_y}:( )?\[(?P<entities>.*)]$', re.IGNORECASE),
            # edge case, no enclosing brackets
            re.compile(rf'^{pref_y}:( )?(?P<entities>.*)$', re.IGNORECASE),
            # edge case, no prefix
            re.compile(rf'^\[?((?P<entities>.*)])?$', re.IGNORECASE),
            # for sample edge case above, would result in enumerated index
            re.compile(rf'^(?P<idx>\d+)\. {pref_y}:( )?\[(?P<entities>.*)]$', re.IGNORECASE)
        ]
        # if not self.batched or self.generate_type == 'baseline-both':  # in both cases, generate 1 annotation at a time
        if not self.batched:
            pat_entities_n_ba = [
                # edge case: newline after prefix, e.g.
                #   `Named Entities: \n[comedy (Genre), 1990s (Year)]`
                #   Further w/o enclosing brackets: `Named Entities: \n
                #       comedy (Genre), 1990s (Year)`
                #   Further newline to join each entity annotations, e.g.
                #       `Named Entities: \n
                #       classic romance movies (Genre),\n
                #       Netflix (Title)`
                re.compile(rf'^{pref_y}: \n\[(?P<entities>.*)]$', re.IGNORECASE),
                re.compile(rf'^{pref_y}: \n(?P<entities>.*)$', re.IGNORECASE),
                re.compile(rf'^{pref_y}: \n(?P<entities>(.+\n)*.+)$', re.IGNORECASE),
            ]
            if self.dataset_name == 'mit-movie':
                # Edge case: weird term after entity annotation, e.g.
                #   `Named Entities: [2017 (Year), Dunkirk (Title)] \nDirector` => just drop the trailing term
                pat_entities_n_ba += [
                    re.compile(rf'^{pref_y}: \[(?P<entities>.*)] \n(Director)$', re.IGNORECASE),
                ]
            self.pattern_entities = pat_entities_n_ba + self.pattern_entities

        self.allowed_entity_types = None
        if isinstance(allowed_entity_types, list):
            self.allowed_entity_types = allowed_entity_types
        elif allowed_entity_types is True:
            self.allowed_entity_types = self.entity_types
        self.nt = Np2Transform(
            dataset_name=self.dataset_name,
            pattern_sentence=self.pattern_sent, pattern_entity=self.pattern_entities, entity_sep=self.entity_sep,
            entity_pair_map=self.entity_pair_map, allowed_entity_types=self.allowed_entity_types,
            batched=self.batched, generate_type=self.generate_type, drop_puncs='both', ec=self.ec)

    def meta(self, **kwargs) -> str:
        return super().meta(insert=self.insert, **kwargs)

    def get_instruction(self, n_annotate: int = 20):
        assert self.sample_format == 'natural-pair-v2'
        if self.dataset_name == 'conll2003-no-misc':
            # return (f"I have {n_annotate} sentences from news stories. "
            #         "For each sentence, please identify all entity names occurred. "
            #         "Especially identify named entities that belong to one of the following entity types:\n"
            #         "[person, location, organization].\n"
            #         "Please list such entities on the following line, in the order of occurrence.\n"
            #         "If no entity is found in a sentence, list 'None'.")

            # additional requirement on listing the same entity multiple times
            # return (f"I have {n_annotate} sentences from news stories. "
            #         "For each sentence, please identify all entity names occurred. "
            #         "Especially identify named entities that belong to one of the following entity types:\n"
            #         "[person, location, organization].\n"
            #         "Please list such entities on the following line, in the order of occurrence.\n"
            #         "If an entity occurs multiple times in a sentence, list it as many times as it occurs.\n"
            #         "If no entity is found in a sentence, list 'None'.")

            # stress on annotating *named* entities only
            # return (f"I have {n_annotate} sentences from news stories. "
            #         "For each sentence, please identify all named entities occurred. "
            #         "Especially identify named entities that belong to one of the following entity types:\n"
            #         "[person, location, organization].\n"
            #         "Please list such named entities on the following line, in the order of occurrence.\n"
            #         "If a named entity occurs multiple times in a sentence, list it as many times as it occurs.\n"
            #         "If no named entity is found in a sentence, list 'None'.")

            # consistent formatting w/ 1-stage
            if n_annotate == 1:
                # ret = f'Here is a sentence from news stories. '
                ret = f'Here is a sentence from a news story. '  # more fluent
            else:
                ret = f'Here are {n_annotate} sentences from news stories. '
            if self.cot:
                ret += ("Please analyze each sentence and identify all named entities occurred that belong to one of the following entity types:\n"
                        "[person, location, organization].\n")
            else:
                ret += ("For each sentence, please identify all named entities occurred that belong to one of the following entity types:\n"
                        "[person, location, organization].\n")
            if self.insert:
                ret += "Please use the definitions below to identify the named entities.\n"
                if self.insert == 'defn':
                    # after demo update
                    defn = schemas.conll2003_no_misc_defn3
                else:
                    assert self.insert == 'schema'
                    raise NotImplementedError
                ret += f'{defn}\n\n---\n\n'
            if self.cot:
                ret += "Show your reasoning in bullet points.\n"
            # if self.cot:
            #     ret += "Please analyze the sentences and list such named entities with the corresponding entity types on the following line, "
            # else:
            #     ret += "Please list such named entities with the corresponding entity types on the following line, "
            ret += "Please list such named entities with the corresponding entity types on the following line, "
            ret += ("in the order of occurrence.\n"
                    "If no entity is found in the generated sentence, leave the brackets empty.")
            return ret
        elif self.dataset_name == 'wiki-gold-no-misc':
            if n_annotate == 1:
                ret = ('Here is a sentence from a Wikipedia article. '
                       'Please identify all named entities occurred that belong to one of the following entity types:\n')
            else:
                ret = (f'Here are {n_annotate} sentences from Wikipedia articles. '
                       f'For each sentence, '
                       f'please identify all named entities occurred that belong to one of the following entity types:\n')
            ets = sconfig(f'datasets.{self.dataset_name}.readable-entity-types').copy()
            ret += f'{pl.nc(ets)}\n'
            ret += ('Please list such named entities with the corresponding entity types on the following line, '
                    'in the order of occurrence.\n'
                    'If no entity is found in the generated sentence, leave the brackets empty.')
            return ret
        elif self.dataset_name in ['mit-movie', 'mit-restaurant']:
            if self.cot:
                raise NotImplementedError
            kd = 'movies' if self.dataset_name == 'mit-movie' else 'restaurants'
            if n_annotate == 1:
                ret = (f'Here is a spoken query to a dialog system about {kd}. '
                       f'Please identify all named entities occurred that belong to one of the following entity types:\n')
            else:
                ret = (f'Here are {n_annotate} spoken queries to a dialog system about {kd}. '
                       'For each query, please identify all named entities occurred that belong to one of the following entity types:\n')
            ets = sconfig(f'datasets.{self.dataset_name}.readable-entity-types').copy()
            ret += f'{pl.nc(ets)}\n'
            # no instruction on no-entity for assume all queries have entities
            ret += ('Please list such named entities with the corresponding entity types on the following line, '
                    'in the order of occurrence.')
            return ret
        elif self.dataset_name == 'job-stack':
            if n_annotate == 1:
                ret = 'Here is a sentence from a job posting on StackOverflow. '
            else:
                ret = f'Here are {n_annotate} sentences from job postings on StackOverflow. '
            # ret += 'Please identify all named entities occurred that belong to one of the following entity types:\n'
            # add `for each` prefix
            ret += 'For each sentence, please identify all named entities occurred that belong to one of the following entity types:\n'

            ets = sconfig(f'datasets.{self.dataset_name}.readable-entity-types').copy()
            ret += f'{pl.nc(ets)}\n'
            ret += ('Please list such named entities with the corresponding entity types on the following line, '
                    'in the order of occurrence.\n'
                    'If no entity is found in the generated sentence, leave the brackets empty.')
            return ret
        else:
            raise NotImplementedError

    def _drop_reason(self, str_sample: str) -> str:
        """
        Split the completion into sentence and entities for cot samples

        Need to drop the reasoning part
        """
        from stefutil import sic

        sents = patterns.find_match(text=str_sample, pattern=self.pattern_extract_sent)
        if len(sents) != 1:
            sic(str_sample, sents)
            raise NotImplementedError
        assert len(sents) == 1
        ret = sents[0].group()

        idxs = patterns.find_match(text=str_sample, pattern=re.compile(r'(?P<idx>\d+)\. '))
        if len(idxs) != 1:
            sic(str_sample)
            raise NotImplementedError
        assert len(idxs) == 1  # sanity check `str_sample` indeed contains 1 sample only

        entities = patterns.find_match(text=str_sample, pattern=self.pattern_extract_entities)
        if len(entities) != 1:
            sic(str_sample, entities)
        assert len(entities) == 1
        return f'{ret}\n{entities[0].group()}'

    def process_completions(
            self, sentences: Union[str, List[str]] = None, shuffle_sentences: Union[bool, int] = False,
            with_unlabeled: Union[bool, int] = None, n_demo: Optional[int] = 5, demo_args: Dict[str, Any] = None,
            completions_dir_name: completions.CompletionDirectoryDict = None, output_dir_name: str = None,
            expected_samples_per_completion: int = None, aggregate: Union[bool, str] = False, lowercase: bool = None,
            load_sample_args: Dict[str, Any] = None, sub_dir: str = None, logprobs: bool = False
    ) -> dataset.NerProcessOutput:
        """
        Ensure the entities generated by the model are valid, matching sentences, in that order
        """
        init_out = self.process_completions_init(
            sentences=sentences, shuffle_sentences=shuffle_sentences, with_unlabeled=with_unlabeled,
            n_demo=n_demo, demo_args=demo_args, completions_dir_name=completions_dir_name, output_dir_name=output_dir_name,
            expected_samples_per_completion=expected_samples_per_completion, aggregate=aggregate, load_sample_args=load_sample_args,
            sub_dir=sub_dir
        )
        base_path, output_path, init_log, sentences, n_expect, d_log_count = astuple(init_out)

        d_dset = sconfig(f'datasets.{self.dataset_name}')
        x_nm, y_nm = d_dset['x-name'], d_dset['y-name']

        def process_single(
                it: completions.CompletionIter
        ) -> Tuple[List[NerReadableExample], Optional[List[float]], Optional[List[List[float]]]]:
            ret_ = []
            # log prob corresponding to each sample, list of entity logprobs corresponding to each sample
            lps_, et_lps_ = ([], []) if logprobs else (None, None)
            n_cpl = len(it.filepaths)
            global_i_sample = 0
            for i_cpl, c in enumerate(it.iter):
                is_last_group = i_cpl == n_cpl - 1
                completion, p_fnm = c.content, c.pretty_filename
                if completion[-1] != '\n':  # append newline if not present
                    completion += '\n'

                if self.batched:
                    out = self._split_from_sentence_samples(
                        completion=completion, is_last_group=is_last_group, edge_split_args=dict(silent=True), filename=p_fnm,
                        pattern_w_sentence=self.pattern_search, pattern_wo_sentence=self.pattern_search_edge)
                    out, sent_miss = out.split_output, not out.sentence_found
                    if sent_miss and not out.success:
                        self.ec(msg=f'No entity annotations found in any enumerated sample w/ {pl.i(out.d_log)}', kind='entity-annot-format')
                        str_samples = []
                    else:
                        assert out.success
                        str_samples = out.samples
                    samples_span, samples_grouped = out.spans, out.grouped  # for logprob extraction
                    n_samples_got = len(str_samples)
                    keep_samples = True
                    # unless it's the last completion, number of samples should match
                    if n_expect and not is_last_group and n_samples_got != n_expect:
                        assert n_samples_got < n_expect

                        # in such case, sentences with annotations missing will be skipped
                        d_log = dict(filename=p_fnm, completion=completion, samples=str_samples)
                        msg = f'Expected {pl.i(n_expect)} samples, but decoded {pl.i(n_samples_got)} w/ {pl.i(d_log)}'
                        self.ec(msg=msg, kind='less-sample-count')

                        if n_samples_got <= math.ceil(n_expect / 2):
                            # too little samples, question annotation quality, ignore all samples from current completion
                            d_log = {'filename': p_fnm, '#got': n_samples_got, '#expect': n_expect}
                            msg = f'Too little samples decoded and completion ignored w/ {pl.i(d_log)}'
                            self.ec(msg=msg, kind='less-sample-count-drop-completion', failed=True)
                            keep_samples = False
                    check_idxs = out.indices_check
                    if keep_samples and out.has_enum_prefix and not check_idxs.match_success:
                        raise ValueError(f'Completion indices are not in order w/ {pl.i(check_idxs.d_log)}')
                    if not keep_samples:
                        str_samples = []
                else:
                    # LLM annotates a single sample per completion, the output is just 1 entity annotation
                    # m = self.pattern_entities.match(completion)
                    m = patterns.match_row(text=completion, pattern=self.pattern_entities)  # sanity check is indeed one entity annotation
                    assert m is not None
                    # label_str = m.group('entities')
                    str_samples = [completion]
                    samples_span, samples_grouped = [m.span()], [m.group('entities')]
                    sent_miss = True  # to be added later
                    n_samples_got = 1

                s2lp = None
                if logprobs:
                    # code pretty much like 1-stage extraction
                    lp_kd = '1-stage-annotations'  # use logprob for the label part; TODO: difference w/ 2-stage?
                    s2lp = logprob.Span2LogProb(
                        logprobs=c.logprobs, completion=completion, sample_type=lp_kd, ec=self.ec, logger=self.logger)
                for i_sample, (str_sample, sample_span, sample_grouped) in enumerate(
                        zip(str_samples, samples_span, samples_grouped), start=global_i_sample):
                    str_sample_ori = str_sample
                    ori_sent = sentences[i_sample]
                    if sent_miss:  # add back the sentence
                        ms_idxs: List[re.Match] = patterns.find_non_overlap_matches(
                            pattern=split.pattern_enumerated_index, text=str_sample, return_matches=True, accept_no_match=True)
                        assert len(ms_idxs) <= 1
                        if len(ms_idxs) == 1:
                            # drop the enumerated index, will be re-added later
                            m = ms_idxs[0]
                            assert m.start() == 0
                            str_sample = str_sample[m.end():]
                            if str_sample[0] == '\n':
                                str_sample = str_sample[1:]

                        str_sample = f'{i_sample - global_i_sample + 1}. {x_nm}: {ori_sent}\n{str_sample}'
                    if self.cot:
                        if not sample_has_reasoning(sample=str_sample):
                            d_log = dict(filename=p_fnm, sample=str_sample)
                            self.ec(msg=f'Sample reasoning not found w/ {pl.i(d_log)}', kind='reasoning-missing')
                        str_sample = self._drop_reason(str_sample=str_sample)
                    # this operation will always be successful
                    out = self.nt.str2x_n_y(sample=str_sample, filename=p_fnm)
                    if not out.success:
                        continue
                    out = self.nt.y2entity_n_type(sample=out.sample, has_entity_type=self.generate_type == 'both')
                    if not out.success:  # e.g. entity type not annotated, skip
                        continue

                    trip_sample = out.sample
                    res_sent = trip_sample.sentence

                    drop = self.check_sentence_diff(sentence_in_prompt=ori_sent, sentence_in_response=res_sent)
                    if drop:
                        continue
                    enms, ets = trip_sample.entity_names, trip_sample.entity_types
                    # even though sentences may not match, respect the generated sentences as it definitely has the entities
                    out = self.nt.sanitize(sentence=trip_sample.sentence, entity_names=enms, entity_types=ets, filename=p_fnm)
                    if out.success:  # otherwise, skip
                        ret_.append(out.sample)

                        if logprobs:
                            # a sample may not be just the annotation part since LLM could re-generate the sentence
                            # assert sent_miss
                            # this is from sample extraction, still contain the enclosing brackets, so drop them
                            entities_raw = sample_grouped.entities
                            entities_raw = f'{y_nm}: {entities_raw}'

                            lp_out = logprob.ner_sample2logprobs(
                                sample=out.sample, sample_str=str_sample_ori, sample_span=sample_span, span2logprob=s2lp,
                                entities_str=entities_raw,
                                entity_sep=self.entity_sep, pattern_entity=self.pattern_entities, entity_pair_map=self.entity_pair_map)
                            # raise NotImplementedError
                            lps_.append(lp_out.logprob)
                            et_lps_.append(lp_out.entity_logprobs)
                global_i_sample += n_samples_got if is_last_group else n_expect  # since prompts are grouped by `n_expect`
            assert global_i_sample == len(sentences)  # sanity check
            self.logger.info(f'Processed {pl.i(len(ret_))} samples from {pl.i(len(sentences))} sentences')
            return ret_, lps_, et_lps_

        t = Timer()
        proc_args = dict(completion_type=self.processed_type, logger=self.logger, logprobs=logprobs)
        init_args = {**proc_args.copy(), **dict(completion_base_path=base_path, output_path=output_path, init_log=init_log)}
        if aggregate:
            assert isinstance(completions_dir_name, dict)
            ret = dict()
            n_grp = len(completions_dir_name)

            d_n_cpl = dict()
            completions.process_completions_init(**init_args)
            lps, et_lps = None, None
            for i, (k, v) in enumerate(completions_dir_name.items(), start=1):
                completions.log_prompt_eg(dir_name=v, base_path=base_path, logger=self.logger)
                grp_ord = f'{pl.i(ordinal(i))}/{pl.i(n_grp)}'
                grp_str = f'group {pl.i(grp_ord)}: {pl.i(k)}'
                self.logger.info(f'Processing {grp_str} w/ directory {pl.i(v)}')
                it_ = completions.iter_completions(dir_name=v, base_path=base_path, **proc_args)
                if logprobs:
                    raise NotImplementedError('merge logprobs across runs')
                ret[k] = samples = process_single(it=it_)
                self.logger.info(f'Processed {grp_str} w/ {pl.i(len(samples))} samples')
                d_n_cpl[k] = len(it_.filepaths)
            grp2n = {k: len(v) for k, v in ret.items()}
            self.logger.info(f'Processed {pl.i(len(ret))} groups w/ {pl.i(grp2n)} samples in each group')

            # aggregate: only keep samples that are annotated in all groups, and the same annotations
            sent2samples = defaultdict(list)
            samples = sum(ret.values(), start=[])
            for s in samples:
                sent2samples[s.sentence].append(s)
            lst_samples = [samples for samples in sent2samples.values() if len(samples) == n_grp]
            sa = SamplesAggregator(scheme=aggregate)
            ret = [sa(samples=lst) for lst in lst_samples]
            ret = [s for s in ret if s is not None]  # drop samples that cannot be aggregated
            self.logger.info(f'Aggregated {pl.i(len(ret))} samples from {pl.i(len(sent2samples))} sentences')
            d_log_count['#completions'] = d_n_cpl
        else:
            it_ = completions.process_completions_init(completions_dir_name=completions_dir_name, **init_args)
            completions.log_prompt_eg(dir_name=completions_dir_name, base_path=base_path, logger=self.logger)
            ret, lps, et_lps = process_single(it=it_)
            d_log_count['#completions'] = len(it_.filepaths)
        return self.finish_ner_processing(
            samples=ret, dedup=True, lowercase=lowercase, output_path=output_path, d_log=d_log_count, time=t,
            logprobs=logprobs, sample_logprobs=lps, entity_logprobs=et_lps
        )
