"""
For generating rephrased sentences given sentences in the original dataset
"""

import re
import json
from os.path import join as os_join
from typing import Tuple

from stefutil import *
from src.util import *
from src.util.ner_example import *
from src.data_util import *
from src.generate.step_wise.util import *


_logger = get_logger('Gen X rephrase')


if __name__ == '__main__':
    dnm = 'conll2003-no-misc'
    ori = 'train-1k-sentences.json'
    post = '1k-rephrase'
    n_reph = 5
    # temp = 1
    temp = 0
    if temp != 1:
        post = f'{post}_t={temp}'
    dir_args = dict(dataset_name=dnm, sub_dir=STEP_WISE_DNM)

    def sents2prompt(sentences: Tuple[str]) -> str:
        assert dnm == 'conll2003-no-misc'
        n_sent = len(sentences)
        assert n_sent != 1
        ret = (f'Here are {n_sent} sentences from news stories. '
               'They cover diverse named entities that belong to the following entity types:\n'
               '[person, location, organization].\n\n'
               'Please paraphrase the following sentences:\n\n')
        sents = '\n\n'.join([f'{i}. Sentence: {enclose_in_quote(s)}' for i, s in enumerate(sentences, start=1)])
        # ret += f'{sents_}\n\n\n---'
        ret += sents
        return ret

    def get_prompts():
        sents = load_processed(dataset_name=dnm, ori=ori).samples
        sents_group = group_n(sents, n=n_reph)
        return [sents2prompt(sentences=s) for s in sents_group]

    def check_prompt():
        prompts = get_prompts()
        # sic(prompts[:10], len(prompts))
        print_prompts(prompt=prompts[:10])
    # check_prompt()

    def write_completion():
        debug = False
        # debug = True
        output_dir = dataset_name2data_dir(**dir_args, output_dir='Sentence-Res', output_postfix=post).path
        sic(output_dir)

        prompts = get_prompts()
        if debug:
            prompts = prompts[:5]
        d_log = {'dataset-name': dnm, '#example rephrased': n_reph}
        write_completions(
            output_path=output_dir, prompts=prompts, logger=_logger, completion_type='Sentence', init_log=d_log,
            temperature=temp, max_tokens=512  # 3 dc sentences ~120 tokens, 500 tokens for 5 sentences should be well-enough
        )
    # write_completion()

    def process():
        # dir_nm = '23-11-27_21-33-48_Sentence-Res_1k-rephrase_debug'
        # dir_nm = '23-11-28_14-58-55_Sentence-Res_1k-rephrase_t=0_debug'
        dir_nm = '23-11-28_15-14-17_Sentence-Res_1k-rephrase_t=0'

        # in some rare cases, the original sentence is in the completion
        # e.g. 1. Sentence: "The seeding number is indicated."
        pattern_ori = re.compile(r'^((?P<idx>\d+)\. )?Sentence: (?P<sent>.+)$', re.IGNORECASE)
        # e.g. `1. Paraphrase: "Colorado hosts a game against Cincinnati."`
        pattern_sent = re.compile(r'^((?P<idx>\d+)\. )?(Paraphrase: |Paraphrased: )?(?P<sent>.+)$', re.IGNORECASE)

        out = dataset_name2data_dir(**dir_args, output_dir='Sentence-Dataset', output_postfix=post, timestamp='short-date')
        base_path, output_path = out.base_path, out.path
        n_expect = n_reph
        d_log = {'output-path': output_path, 'completions-dir-name': dir_nm, 'expected-samples-per-completion': n_expect}
        out = process_completions_init(
            completion_base_path=base_path, completions_dir_name=dir_nm, output_path=output_path,
            completion_type='Sentence', logger=_logger, init_log=d_log)
        log_prompt_eg(dir_name=dir_nm, base_path=base_path, logger=_logger)
        ec = EdgeCases(logger=_logger)

        sents = []
        t = Timer()
        for c in out.iter:
            completion = c.content

            _sents = []
            lines = completion2lines(completion=completion)
            for ln in lines:
                if match(text=ln, pattern=pattern_ori) is not None:
                    ec(msg=f'Original sentence in completion: [{ln}]', kind='original-sentence')
                    continue
                m = match(text=ln, pattern=pattern_sent)
                assert m is not None
                sent = m.group('sent').strip()
                _sents.append(sent)

            if len(_sents) != n_expect:
                d_log = {'filename': c.pretty_filename, '#expect': n_expect, '#got': len(_sents), 'sentences': _sents}
                msg = f'Expected {pl.i(n_expect)} samples, but decoded {pl.i(len(_sents))} w/ {pl.i(d_log)}.'
                ec(msg=msg, kind='wrong-sentence-count')
            sents += [drop_enclosing_quotes(s) for s in _sents]

        d_log_count = {'#sentence-extracted': len(sents)}
        out = de_duplicate_samples(samples=sents, logger=_logger)
        sents, n_drop = out.samples, out.n_dup_drop
        assert n_drop == 0
        d_log_count.update({'#duplicate-dropped': n_drop, '#sentence-kept': len(sents)})
        out_fnm = os_join(output_path, 'sentences.json')
        with open(out_fnm, 'w') as f:
            json.dump(sents, f, indent=4)
        _logger.info(ec.summary())
        _logger.info(f'Processed sentences w/ {pl.i(d_log_count)} in {pl.i(t.end())}')
    process()
