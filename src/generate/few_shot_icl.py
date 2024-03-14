"""
For few-shot In-Context Learning (ICL) baseline w/ ChatGPT3.5 to compare w/ our approach

Use a similar prompt template as SimPrompt

Directly generate an annotation for a single sentence input, given a few demo annotations,
"""

import json
import random
from os.path import join as os_join
from typing import Dict, List, Union, Any
from dataclasses import asdict

from stefutil import *
from src.util import *
from src.util.ner_example import *
from src.data_util import *
from src.generate.step_wise import *


__all__ = ['FewShotIclPredictor']


_logger = get_logger('Few Shot ICL')

DEBUG = False
# DEBUG = True


FEW_SHOT_ICL_DIR_NAME = 'few-shot'


def load_test_set_samples(dataset_name: str = None, kind: str = 'readable') -> Union[List[NerReadableExample], List[NerBioExample]]:
    """
    get test set sentences to run inference
    """
    # use the version saved to local file, cos these have duplicates dropped
    ca.assert_options(display_name='Data Sample Kind', val=kind, options=['readable', 'bio'])
    if dnm in DatasetLoader.from_hf:
        fnm = 'test'
    elif dnm in DatasetLoader.from_local_bio_file:
        fnm = 'test-all'
    else:
        raise NotImplementedError
    test_fnm = f'{kind}-{fnm}'
    test_set_path = os_join(pu.proj_path, 'original-dataset', dataset_name, f'{test_fnm}.jsonl')

    if kind == 'readable':
        def load_single(s: str) -> NerReadableExample:
            return NerReadableExample(**json.loads(s))
    else:
        assert kind == 'bio'

        def load_single(s: str) -> NerBioExample:
            return NerBioExample(**json.loads(s))
    with open(test_set_path) as f:
        return [load_single(s) for s in f]


class FewShotIclPredictor(EntityAnnotationGenerator):
    """
    By defn., classify each sentence in MIT-Movie test set, one in each prompt
    """

    def __init__(self, **kwargs):
        super().__init__(generate_type='baseline-both', batched=False, allowed_entity_types=True, **kwargs)
        self.dir_args['sub_dir'] = FEW_SHOT_ICL_DIR_NAME  # override default `step-wise` dir name

        # for processing samples
        # just drop quotes, for brackets should not be generated to begin with,
        #   since the sentence is not re-generated and sentences are all in the original dataset
        # self.nt = Np2Transform(
        #     dataset_name=self.dataset_name, entity_sep=self.entity_sep, drop_puncs='quote', ec=self.ec,
        #     entity_pair_map=self.entity_pair_map, allowed_entity_types=self.entity_types)
        # self.nt.batched = self.batched = False

        # pref_y = sconfig(f'datasets.{self.dataset_name}.y-decoded-name')
        # pref_y = options2re_options(options=pref_y)
        # self.pattern_entities = re.compile(rf'^({pref_y}):( )?\[(?P<entities>.*)](\.)?$', re.IGNORECASE)
        # self.pattern_entities

    def get_instruction(self):
        """
        Always a single sample sentence to classify
        :return:
        """
        if self.dataset_name == 'conll2003-no-misc':
            ret = 'Suppose you are a news writer. You are given a sentence from a news story.'
        elif self.dataset_name == 'wiki-gold-no-misc':
            ret = 'Suppose you are a Wikipedia editor. You are given a sentence from a Wikipedia article.'
        elif self.dataset_name in ['mit-movie', 'mit-restaurant']:
            kd = 'movies' if self.dataset_name == 'mit-movie' else 'restaurants'
            ret = ("Suppose you are a user of a dialog system or conversational agent. "
                   f"You are given a spoken query related to {kd} to the dialog system.")
        else:
            raise NotImplementedError
        ret = f'{ret} Please identify all named entities occurred that belong to one of the following entity types:\n'
        ets = sconfig(f'datasets.{self.dataset_name}.readable-entity-types')
        ets = ', '.join(ets)
        ret += f'[{ets}].\n'
        ret += ("Please list such named entities with the corresponding entity types on the following line, "
                "in the order of occurrence.\n")
        return ret

    def load_test_samples(self) -> List[NerReadableExample]:
        # load the readable version since feeding into LLM and for my own processing
        ret = load_test_set_samples(dataset_name=self.dataset_name, kind='readable')
        if DEBUG:
            ret = ret[:25]
        return ret

    def write_completions(
            self, n_demo: int = 5, output_dir_nm: str = None, prompt_seed: int = None, demo_args: Dict[str, Any] = None, **kwargs
    ):
        # since this is a few-shot baseline, no complicated stuff, 1 sample 1 prompt, not much variation, no shuffling needed also
        output_path = dataset_name2data_dir(**self.dir_args, output_dir=f'{self.sample_type}-Res', output_postfix=output_dir_nm).path
        add_file_handler(logger=self.logger, file_path=os_join(output_path, 'completion.log'))

        test_samples = self.load_test_samples()

        gen_ = random.Random(prompt_seed) if prompt_seed else None
        # convert to dict to signal a sample to annotate, per `AnnotationGenerator._sample2sample_str` API
        prompts = [self.get_prompt(samples=[asdict(s)], n_demo=n_demo, generator=gen_, demo_args=demo_args) for s in test_samples]
        # sic(prompts[:5], len(prompts))
        # raise NotImplementedError

        d_log = {
            'dataset-name': self.dataset_name, '#demo': n_demo,
            '#test-set-sentence-to-predict': len(test_samples), 'prompt-seed': prompt_seed, 'output-path': output_path
        }
        completions.write_completions(
            output_path=output_path, logger=self.logger, add_fl_writer=False, completion_type=self.processed_type,
            init_log=d_log, prompts=prompts, save_all_prompts=True, **kwargs)

    def process_completions(self, completions_dir_name: completions.CompletionDirectory = None, output_dir_name: str = None):
        """
        Get the LLM predictions for each sample, and then evaluate the performance
        """
        out = dataset_name2data_dir(
            **self.dir_args, output_dir=f'Few-Shot-Baseline-Eval', output_postfix=output_dir_name, timestamp='short-date'
        )
        output_path, base_path = out.path, out.base_path
        init_log = {
            'class-name': self.__class__.__qualname__, 'output-path': output_path,
            'completions-dir-name': completions_dir_name,
        }

        test_samples = self.load_test_samples()

        it = completions.process_completions_init(
            completions_dir_name=completions_dir_name, completion_base_path=base_path, output_path=output_path, init_log=init_log,
            completion_type=self.processed_type, logger=self.logger)
        completions.log_prompt_eg(dir_name=completions_dir_name, base_path=base_path, logger=self.logger)

        d_dset = sconfig(f'datasets.{self.dataset_name}')
        x_nm, y_nm = d_dset['x-name'], d_dset['y-name']
        samples_pred = []
        for i_cpl, c in enumerate(it.iter):
            completion, fnm, p_fnm = c.content, c.filename, c.pretty_filename
            sample = test_samples[i_cpl]
            sent = sample.sentence

            # the entire LLM response should be the annotation for that sentence
            #   so saves my splitting, just extract the entities part
            # m = self.pattern_entities.match(completion)
            m = patterns.match_row(text=completion, pattern=self.pattern_entities)  # sanity check is indeed 1 entity
            # if m is None:
            #     sic(completion, self.pattern_entities)
            assert m is not None
            # label_str = m.group('entities')
            # cannot use standard processing here, cos LLM can generate in edge case formatting, e.g.
            #   `Named Entities:
            #   1. Japan (location)
            #   2. Asian Cup (organization)
            #   3. Syria (location)
            #   4. Group C (organization)`
            #   I should have added an explicit template in prompt, e.g. enclose in double brackets...
            sample_str = f'1. {x_nm}: "{sent}"\n{completion}'  # add dummy index following step-wise annotation pattern
            # sic(sample_str)

            # make sure each sample can be processed
            # out = self.nt.str2x_n_y(sample=GroupedStrSample(sentence=sent, entities=label_str, entities_raw=completion))
            out = self.nt.str2x_n_y(sample=sample_str)
            assert out.success
            out = self.nt.y2entity_n_type(sample=out.sample, resolve_failure=True)
            assert out.success

            trip_sample = out.sample
            res_sent = trip_sample.sentence
            if res_sent != sent:
                sic(fnm, sample, trip_sample)
                sic(sent, res_sent, completion)
            assert res_sent == sent  # sanity check original test set sentence is not changed

            enms, ets = trip_sample.entity_names, trip_sample.entity_types

            if self.dataset_name == 'conll2003-no-misc' and fnm.endswith('-196'):
                # LLM doesn't generate multi-occurring entities, and it's a weird sample: all person names, e.g.
                #   `Barbarians - 15 - Tim Stimpson (England); 14 - Nigel Walker (Wales), 13 - Allan Bateman (Wales), ...`
                # this will make my pipeline prohibitive to run, cos duplicating the missing LLM entities to a total of 29
                #   and reordering a list of **29** entities to match the sentence, so just ignore this sample
                samples_pred.append(NerReadableExample.from_d(sentence=sent))
                continue

            out = self.nt.sanitize(sentence=trip_sample.sentence, entity_names=enms, entity_types=ets, resolve_failure=True)
            assert out.success
            samples_pred.append(out.sample)
        if self.ec.have_edge_case:
            self.logger.info(self.ec.summary())

        # save the LLM predictions for future reference
        out_fnm = os_join(output_path, f'readable-test-pred')
        d_write = dict(dataset_name=self.dataset_name, entity_types=self.entity_types)
        dataset.write_dataset(samples=[asdict(s) for s in samples_pred], output_filename=out_fnm, kind='readable', **d_write)

        # get prediction performance
        # can do this just in the readable format, since prediction & ground truths are already in entity chunks
        # but to leverage `seqeval` and `nervaluate` libraries, have to convert to BIO format
        lst_trues, lst_preds = [], []  # sample => list of BIO-tags
        for s_pred, s_true in zip(samples_pred, test_samples):
            assert s_pred.sentence == s_true.sentence  # sanity check
            pred_tags = list(s_pred.to_bio().ner_tags)
            true_tags = list(s_true.to_bio().ner_tags)
            lst_trues.append(true_tags)
            lst_preds.append(pred_tags)
        out = eval.get_entity_cls_report(trues=lst_trues, preds=lst_preds, entity_types=self.entity_types)
        df_ent, df_ent_partial = out.exact, out.partial

        f1_exact, f1_partial = df_ent['f1-score']['micro avg'], df_ent_partial['f1-score']['micro avg']
        d_metric = {'f1-exact': to_percent(f1_exact), 'f1-partial': to_percent(f1_partial)}
        self.logger.info(f'F1-score: {pl.i(d_metric)}')
        for (kd, df) in ([('entity', df_ent), ('entity-partial', df_ent_partial)]):
            eval.write_cls_report(df=df, kind=kd, split='test', output_path=output_path, logger=self.logger)
        self.logger.info(f'Entity-wise f1: {pl.i(eval.cls_report_df2class_wise_scores(df=df_ent))}')


if __name__ == '__main__':
    dnm = 'conll2003-no-misc'
    # dnm = 'wiki-gold-no-misc'
    # dnm = 'mit-movie'
    # dnm = 'mit-restaurant'
    # dnm = 'job-stack'

    seed = 42  # seed for randomness in prompt demo sample construction
    n_demo_ = 1  # 1-shot demo
    da = None if dnm in ['mit-movie', 'mit-restaurant'] else dict(include_none_samples=True)
    lower = dnm in ['mit-movie', 'mit-restaurant']

    temp = 0

    gen = FewShotIclPredictor(dataset_name=dnm)

    post = dict()
    if temp != 1:
        post['t'] = temp
    post = pl.pa(post) if post != dict() else None
    if DEBUG:
        post = f'{post}_debug' if post else 'debug'
    out_dnm = gen.meta(postfix=post)

    def check_prompt():
        n = 5
        test_samples = load_test_set_samples(dataset_name=dnm, kind='readable')[:n]
        test_samples = [asdict(s) for s in test_samples]
        gen_ = get_random_generator(generator=seed)
        get_prompt = get_prompt_fn_w_samples(
            get_prompt=gen.get_prompt, samples=test_samples, group_size=1, prompt_args=dict(n_demo=n_demo_, demo_args=da, generator=gen_))
        prettier.print_prompts(prompt=get_prompt, n=n)

    def write_completion():
        md_nm = 'gpt-3.5-turbo-1106'
        # LLM don't regenerate the sentence in prompt, just generates the entity annotations
        # since just 1 annotation for 1 sentence, shouldn't take many tokens, to be safe, 128 should be well enough
        if dnm == 'conll2003-no-misc':
            max_tok = 256  # for some weird and long samples, e.g. a huge list of person names
        else:
            max_tok = 128
        timeout = 20

        gen.write_completions(
            n_demo=n_demo_, demo_args=da, prompt_seed=seed, output_dir_nm=out_dnm,
            model_name=md_nm, max_tokens=max_tok, temperature=temp, timeout=timeout, logprobs=True)

    def process():
        if dnm == 'conll2003-no-misc':
            dir_nm = '24-02-07_12-49-39_Few-Shot-Label-Res_{fmt=n-p2}_{t=0}'
        elif dnm == 'wiki-gold-no-misc':
            dir_nm = '24-02-07_21-53-40_Few-Shot-Label-Res_{fmt=n-p2}_{t=0}'
        elif dnm == 'mit-movie':
            # dir_nm = '24-02-03_15-38-37_Few-Shot-Label-Res_{fmt=n-p2}_{t=0}_debug'
            # dir_nm = '24-02-03_16-39-44_Few-Shot-Label-Res_{fmt=n-p2}_{t=0}_debug'  # accidentally ran
            dir_nm = '24-02-03_18-55-08_Few-Shot-Label-Res_{fmt=n-p2}_{t=0}'
        elif dnm == 'mit-restaurant':
            dir_nm = '24-02-08_01-21-39_Few-Shot-Label-Res_{fmt=n-p2}_{t=0}'
        else:
            raise NotImplementedError
        gen.process_completions(completions_dir_name=dir_nm, output_dir_name=out_dnm)
    # check_prompt()
    # write_completion()
    process()
