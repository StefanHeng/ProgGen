import os
import random
from os.path import join as os_join
from copy import deepcopy
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass

from stefutil import get_logger, pl, Timer, add_file_handler, get_random_generator, group_n
from src.util import dataset_name2data_dir
from src.util.ner_example import NerReadableExample
from src.data_util import prettier, completions
from src.generate.step_wise.util import AnnotationGenerator, load_processed


__all__ = ['FromSentenceGenerator']


_logger = get_logger(__name__)


# DEBUG = True
DEBUG = False


_allowed_diff = [
    {'s', 'z'},  # e.g. `subsidized` vs `subsidised`
    {'-', ' '},  # e.g. `sea-level` vs `sea level`
    {'"', '.'}  # e.g. `city."` vs `city.`
]


def _differ_in_1_allowed_char(str1: str, str2: str) -> bool:
    """
    :return: whether `str1` and `str2` differ in 1 character only
        Intended for a lenient check of sentence mismatch
    """
    diff = []
    for c1, c2 in zip(str1, str2):
        if c1 != c2:
            diff.append({c1, c2})
    return len(diff) == 1 and diff[0] in _allowed_diff


_allowed_prefixes = ['"']
_allowed_postfixes = ['.', '"', '."']


def drop_start_n_end_punc(sent: str) -> List[str]:
    """
    Include all versions, for edge case, e.g. `a new organization called "hanks for hope."`
    :param sent:
    :return:
    """
    ret = [sent]
    # pref_len, post_len = sent[0] in _allowed_prefixes, sent[-1] in _allowed_postfixes
    pref_len = next((len(p) for p in _allowed_prefixes if sent.startswith(p)), 0)
    post_len = next((len(p) for p in _allowed_postfixes if sent.endswith(p)), 0)
    if pref_len > 0:
        ret.append(sent[pref_len:])
    if post_len > 0:
        ret.append(sent[:-post_len])
    if pref_len > 0 and post_len > 0:
        ret.append(sent[pref_len:-post_len])
    return ret


def sentences_match(sent_gen: str, sent_ori: str):
    """
    A lenient comparison on whether the generated sentence from annotations match the original sentence in prompt
    """
    sent_gen, sent_ori = sent_gen.lower(), sent_ori.lower()  # ignore case
    if sent_gen == sent_ori:
        return True
    else:
        # drop leading and trailing punctuations in the allowed list for both
        # effectively ignore these punctuations in the comparison
        sent_gen, sent_ori = drop_start_n_end_punc(sent_gen), drop_start_n_end_punc(sent_ori)
        return len(set(sent_gen) & set(sent_ori)) > 0  # any overlap means a match


@dataclass
class LoadSentencesOutput:
    sentences: List[str] = None
    ori_sentences: List[str] = None
    gen_sentences: List[str] = None


@dataclass
class CompletionInitOutput:
    base_path: str = None
    output_path: str = None
    init_log: Dict[str, Any] = None
    sentences: Union[List[str], Dict[str, List[str]]] = None
    n_expect: int = None
    d_log_count: Dict[str, int] = None


class FromSentenceGenerator(AnnotationGenerator):
    """
    For shared functionalities between `SpanGenerator` and `LabelGenerator`
    Both starts generating from sentences, differ in either just identify or both identify & tag
    """
    def __init__(self, original_sentences: Union[bool, str] = None, **kwargs):
        super().__init__(**kwargs)
        self.s2s = None

        assert self.sample_format == 'natural-pair-v2'

        self.original_sentences = original_sentences

    # def get_prompt(self, sentences: Iterable[str] = None, **kwargs):
    #     return super().get_prompt(samples=[dict(sentence=sent) for sent in sentences], **kwargs)

    def _load_gen_sentences(
            self, sentences: Union[str, List[str]] = None, shuffle: Union[bool, int] = False, seed_delta: int = None, **kwargs):
        if isinstance(sentences, str):
            if self.original_sentences is True:
                return load_processed(dataset_name=self.dataset_name, ori=sentences, shuffle=False).samples
            else:  # rephrased sentences are in the same directory
                load_args = dict(dataset_name=self.dataset_name, kind='sentence', logger=self.logger)
                return load_processed(**load_args, dir_name=sentences, shuffle=shuffle, seed_delta=seed_delta, **kwargs).samples
        else:
            assert isinstance(sentences, list)
            return sentences

    def _load_ori_sentences(self, with_unlabeled: Union[bool, int] = None, n_demo: Optional[int] = 5, demo_args: Dict[str, Any] = None):
        n_shot_args = {'n_demo': n_demo, **demo_args}
        n = with_unlabeled if isinstance(with_unlabeled, int) else 100
        out = self.loader.get_few_demo_and_n_samples(n_shot_args=n_shot_args, n_args=dict(n=n, shuffle=True))
        return [eg.sentence for eg in out.n_samples]  # no need to shuffle cos shuffled already, annotate in that order

    def _load_sentences(self, load_ori_args: Dict[str, Any] = None, load_gen_args: Dict[str, Any] = None, **kwargs):
        ret, ori_sents, gen_sents = [], None, None
        if load_ori_args:
            ori_sents = self._load_ori_sentences(**load_ori_args)
            if DEBUG:
                ori_sents = ori_sents[:25]
            ret += ori_sents
        if load_gen_args:
            gen_sents = self._load_gen_sentences(**load_gen_args, **kwargs)
            if DEBUG:
                gen_sents = gen_sents[:25]
            ret += gen_sents
        return LoadSentencesOutput(sentences=ret, ori_sentences=ori_sents, gen_sentences=gen_sents)

    def write_completions(
            self, n_annotate: int = None, n_identify: int = None, n_demo: Optional[int] = 5, sentences: Union[str, List[str]] = None,
            output_dir_nm: str = None,
            resume: int = None, prompt_args: Dict[str, Any] = None, with_unlabeled: Union[bool, int] = None,
            shuffle_sentences: Union[bool, int] = False, aggregate: Union[int, bool] = None, prompt_seed: int = None,
            load_sample_args: Dict[str, Any] = None,
            **kwargs
    ):
        """
        :param n_annotate: #sentences to annotate in a single prompt
            for 2-stage generation, step 2: generate annotations
        :param n_identify: #sentences to identify potential entity spans in a single prompt
            for 3-stage generation, step 2: generate entity spans
        :param n_demo: #sentences to show as demo
        :param sentences: List of sentences to annotate or directory name containing sentences to annotate
        :param output_dir_nm: completion output directory name
        :param resume: If given, resume from the i-th sentence as if the first i-1 sentences have been annotated
        :param prompt_args: Additional arguments to `get_prompt`
        :param with_unlabeled: If true, also take some sentences in the original train set to annotate
        :param shuffle_sentences: If true, the sentences are shuffled
            If an integer, the sentences are shuffled with the given random seed
        :param aggregate: If true, write completions multiple times for the same set of sentences
        :param prompt_seed: If given, used as random seed for generating prompts
        :param load_sample_args: Additional arguments to `load_processed`
        """
        out_path = dataset_name2data_dir(**self.dir_args, output_dir=f'{self.sample_type}-Res', output_postfix=output_dir_nm).path

        gen = get_random_generator(generator=prompt_seed)

        def run_single(i_run: int = None, seed_delta: int = None, output_path: str = None):
            if aggregate:
                assert i_run is not None
                output_path = os_join(out_path, f'run-{i_run}')
                os.makedirs(output_path, exist_ok=True)
            else:
                assert i_run is None  # sanity check

            # automatically removes prior file handler
            add_file_handler(logger=self.logger, file_path=os_join(output_path, 'completion.log'))
            ppt_args = prompt_args or dict()
            ori_args, gen_args = None, None
            if with_unlabeled:
                demo_args = ppt_args.get('demo_args', dict())
                ori_args = dict(n_demo=n_demo, with_unlabeled=with_unlabeled, demo_args=demo_args)
            if sentences:
                # if not DEBUG:
                #     # TODO: for a complete run, first take a subset of samples then shuffle
                #     raise NotImplementedError('load sentence w/ subset first if on DEBUG, then shuffle')
                #     gen_args = dict(sentences=sentences, shuffle=shuffle_sentences, seed_delta=seed_delta)
                gen_args = dict(sentences=sentences, shuffle=shuffle_sentences)
            out = self._load_sentences(load_ori_args=ori_args, load_gen_args=gen_args, **(load_sample_args or dict()))
            sents, sents_ori, sents_gen = out.sentences, out.ori_sentences, out.gen_sentences
            # sic(sents[:20], len(sents))
            # raise NotImplementedError

            if seed_delta is not None:
                assert DEBUG
                # for now, just do a random shuffle again on different seeds; TODO: remove
                random.seed(seed_delta)
                random.shuffle(sents)
                random.seed()

            ann, idf = n_annotate is not None, n_identify is not None  # sanity check
            assert (ann or idf) and not (ann and idf)
            sents_group = group_n(sents, n=n_annotate or n_identify)

            prompts = []
            for sents in sents_group:
                samples = [dict(sentence=sent) for sent in sents]
                prompts.append(self.get_prompt(samples=samples, n_demo=n_demo, generator=gen, **ppt_args))
            # print_prompts(prompt=prompts[:5])
            # raise NotImplementedError

            if resume is not None:
                if self.sample_type != 'Label' and aggregate:  # intended for 2-stage generation 2nd step only
                    raise NotImplementedError
                prompts = prompts[resume:]

            check_prompt_ = False
            # check_prompt_ = True
            if check_prompt_:
                from stefutil import sic
                sic(len(sents), len(prompts))
                prettier.print_prompts(prompt=prompts[:5])
                raise NotImplementedError
            d_log = {
                'dataset-name': self.dataset_name, '#annotate': n_annotate, '#identify': n_identify, '#demo': n_demo,
                'get-prompt-args': ppt_args,
                'generated-sentences-dir-name': sentences, 'unlabeled-flag': with_unlabeled,
                '#sentence-to-annotate': len(sents), 'shuffle-sentences': shuffle_sentences,
                'resume-index': resume, 'prompt-seed': prompt_seed, 'aggregate': aggregate,
            }

            if with_unlabeled:
                d_log.update({'#sentence-from-train': len(sents_ori), '#sentence-generated': len(sents_gen)})
            ret = deepcopy(d_log)
            d_log.update({'output-path': output_path, 'resume-index': resume})

            completions.write_completions(
                output_path=output_path, logger=self.logger, add_fl_writer=False,
                completion_type=self.processed_type, init_log=d_log, prompts=prompts, save_all_prompts=True, **kwargs
            )
            return ret

        # sic(aggregate)
        # raise NotImplementedError
        if not aggregate:
            run_single(output_path=out_path)
        else:
            if self.sample_type != 'Span':  # intended for generating spans in 3-stage generation only
                raise NotImplementedError

            is_first_run = True
            d_log_check = None

            n_run_given = isinstance(aggregate, int) and not isinstance(aggregate, bool)
            n_run = aggregate if n_run_given else 3
            t = Timer()
            for i in range(n_run):
                # use different sample shuffling order for different runs
                # i_run is used for output directory => 1-indexed; seed_delta is used for shuffling => 0-indexed
                d_log_c = run_single(i_run=i+1, seed_delta=i, output_path=out_path)
                if is_first_run:
                    d_log_check = deepcopy(d_log_c)
                    is_first_run = False
                else:
                    assert d_log_check == d_log_c
            self.logger.info(f'Completions on {pl.i(n_run)} runs finished in {pl.i(t.end())}')

    def process_completions_init(
            self, sentences: Union[str, List[str]] = None, shuffle_sentences: Union[bool, int] = False,
            with_unlabeled: Union[bool, int] = None, n_demo: Optional[int] = 5, demo_args: Dict[str, Any] = None,
            completions_dir_name: completions.CompletionDirectoryDict = None,
            output_dir_name: str = None,
            expected_samples_per_completion: int = None, aggregate: Union[bool, str, int] = False, load_sample_args: Dict[str, Any] = None,
            sub_dir: str = None
    ):
        dir_args = deepcopy(self.dir_args)
        if sub_dir is not None:
            dir_args['sub_dir'] = sub_dir
        d_out = dataset_name2data_dir(
            **dir_args, output_dir=f'{self.processed_type}-Dataset', output_postfix=output_dir_name, timestamp='short-date')
        output_path, base_path = d_out.path, d_out.base_path
        init_log = {
            'class-name': self.__class__.__qualname__, 'metadata': self.meta(), 'output-path': output_path,
            'completions-dir-name': completions_dir_name, 'expected-samples-per-completion': expected_samples_per_completion,
            'aggregate': aggregate, 'shuffle-sentences': shuffle_sentences, 'sub-dir': sub_dir,
        }

        ori_args, gen_args = None, None
        if with_unlabeled:
            ori_args = dict(n_demo=n_demo, with_unlabeled=with_unlabeled, demo_args=demo_args)
        if sentences:
            gen_args = dict(sentences=sentences, shuffle=shuffle_sentences)
        sentences = self._load_sentences(load_ori_args=ori_args, load_gen_args=gen_args, **(load_sample_args or dict())).sentences
        if DEBUG:
            sentences = sentences[:25]

        if aggregate:
            if self.generate_type == 'both':
                raise NotImplementedError

            if DEBUG:  # just re-shuffle the sentences
                sents = sentences
                sentences = dict()
                for i in range(3):
                    sents_ = deepcopy(sents)
                    random.seed(i)
                    random.shuffle(sents_)
                    random.seed()
                    sentences[f'run-{i+1}'] = sents_
            else:  # TODO: should 1call load sentences 3 times, w/ different seeds
                raise NotImplementedError

        return CompletionInitOutput(
            base_path=base_path, output_path=output_path, init_log=init_log, sentences=sentences,
            n_expect=expected_samples_per_completion, d_log_count={'#sentence': len(sentences)}
        )

    def process_completions(
            self, sentences: Union[str, List[str]] = None, shuffle_sentences: Union[bool, int] = False,
            with_unlabeled: Union[bool, int] = None, n_demo: Optional[int] = 5, demo_args: Dict[str, Any] = None,
            completions_dir_name: completions.CompletionDirectoryDict = None,
            output_dir_name: str = None,
            expected_samples_per_completion: int = None, aggregate: Union[bool, str] = False, lowercase: bool = None
    ) -> List[NerReadableExample]:
        """
        Ensure the entities generated by the model are valid, matching sentences, in that order
        """
        raise NotImplementedError
