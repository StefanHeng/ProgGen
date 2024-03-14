import os
import re
import json
import math
import time
import glob
import logging
import asyncio
from os.path import join as os_join
from typing import List, Tuple, Dict, Any, Union, Optional, Callable, Iterable
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from stefutil import *
from src.util import *
from src.data_util import prettier


__all__ = [
    'completion_has_enum_prefix', 'log_prompt_eg',
    'WriteCompletionsOutput', 'write_completions',
    'completion_file2index', 'completion_dir_name2file_paths',
    'CompletionDirectory', 'CompletionDirectoryDict',
    'CompletionIter', 'iter_completions', 'process_completions_init',
    'completion2lines',
]


_logger = get_logger('Gen Util')


CompletionDirectory = Union[str, List[str]]
CompletionDirectoryDict = Union[CompletionDirectory, Dict[str, CompletionDirectory]]


# Check for enum, e.g. `15. whatever`
# a heuristic; TODO: more accurate checks if needed
_pattern_has_enum = re.compile(r'^(?P<idx>\d+)\. (?P<sent>.*)$')


def completion_has_enum_prefix(completion: str) -> bool:
    lns = completion.split('\n')
    return any(_pattern_has_enum.match(ln) is not None for ln in lns)


@dataclass
class GptGenArgs:
    model_name: str = 'gpt-3.5-turbo'
    prompt: Union[str, List[str]] = None
    temperature: float = 1
    max_tokens: int = 512
    logprobs: bool = False
    seed: int = None


def _save_prompts(
        prompts: Union[List[str], List[List[str]]] = None,
        output_path: str = None, save_all: bool = None,
        completion_fnms: Union[str, List[str]] = None, logger: logging.Logger = None
):
    assert isinstance(prompts, list) and len(prompts) > 0
    e0 = prompts[0]
    if isinstance(e0, str):
        is_group = False
    else:
        assert isinstance(e0, list)
        is_group = True
    prompt_groups = None
    if is_group:
        prompt_groups = prompts
        prompts = sum(prompts, start=[])

    if save_all is None:  # save all prompts if there are more than 2 unique prompts
        save_all = len(set(prompts)) > 1
    # write prompt to json and a log file
    pf = 'prompts' if save_all else 'prompt'
    pf_j = os_join(output_path, f'{pf}.json')
    pf_l = os_join(output_path, f'{pf}.log')

    prompt0 = prompts[0]
    with open(pf_j, 'w') as fl:
        d = dict(prompts=prompts) if save_all else dict(prompt=prompt0)
        json.dump(d, fl, indent=4)

    if is_group:
        def prompt_iter() -> Iterable[Tuple[str, str]]:
            g = 0
            for i_g, pg in enumerate(prompt_groups, start=1):
                for j, prompt in enumerate(pg, start=1):
                    g += 1
                    yield f'Global index {g}, Group {i_g}, #{j}', prompt
    else:
        def prompt_iter() -> Iterable[Tuple[str, str]]:
            for idx, prompt in enumerate(prompts, start=1):
                yield f'#{idx}', prompt

    fnm_iter = None
    if completion_fnms is not None:
        if isinstance(completion_fnms, str):
            completion_fnms = [completion_fnms]
        assert len(completion_fnms) == len(prompts)
        fnm_iter = iter(completion_fnms)
    with open(pf_l, 'w') as fl:
        if save_all:
            prompts_ = []
            sep = '-' * 200  # separate prompts by dashes
            for i, p in prompt_iter():
                if fnm_iter:
                    i = f'{i}, Completion file: {next(fnm_iter)}'
                prompts_.append(f'{i}:\n{p}\n\n\n{sep}')
            sep_ = '\n' * 5
            fl.write(sep_.join(prompts_))
            fl.write(sep_)
        else:
            fl.write(f'Prompt:\n{prompt0}\n')
    if logger:
        logger.info(f'Prompts written to {pl.i(stem(output_path))}')


def log_prompt_eg(dir_name: CompletionDirectory, base_path: str = None, logger: logging.Logger = None):
    dir_nm = dir_name[0] if isinstance(dir_name, list) else dir_name
    path = os_join(base_path, dir_nm, 'prompts.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            prompt0 = json.load(f)['prompts'][0]
        # prompt0 = pl.i(prompt0)
        prompt0 = prettier.color_code_prompt(prompt0)
        logger.info(f'Generation prompt example:\n{prompt0}...')
        logger.info('End of prompt')


@dataclass
class WriteCompletionsOutput:
    output_dir: str = None
    output_path: str = None
    cost: Dict[str, Any] = None


def _write_completions(
        prompts: Union[str, List[str]] = None, output_dir: str = None, group_size: int = None,
        model_name: str = 'gpt-3.5-turbo', max_tokens: Union[int, List[int]] = 1024, temperature: float = 1, seed: int = None,
        logger: logging.Logger = None, save_all_prompts: Optional[bool] = None, completion_fnms: Union[str, List[str]] = None,
        dir_name2prompts: Dict[str, Any] = None,
        use_openai_parallel: bool = True, timeout: int = None, logprobs: bool = False
) -> WriteCompletionsOutput:
    """
    Get a group of prompts, make API cals, and write prompts and completions to files
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    if completion_fnms is not None:
        if isinstance(completion_fnms, str):
            completion_fnms = [completion_fnms]
        assert len(prompts) == len(completion_fnms)
    else:
        completion_fnms = [f'completion-{i}.txt' for i in range(1, len(prompts) + 1)]

    logger = logger or _logger
    n_prompt = len(prompts)

    def _log_prompt_eg(prompt: Union[str, List[str]]):
        if isinstance(prompt, list):
            prompt = sample_single(prompt)
        # prompt = pl.i(prompt)
        prompt = prettier.color_code_prompt(prompt)
        logger.info(f'Generating completions w/ {pl.i(n_prompt)} calls and prompt example:\n{prompt}...')
        logger.info('End of prompt')

    if isinstance(max_tokens, int):
        max_tokens_ = max_tokens
        lst_max_tokens = [max_tokens] * n_prompt
    else:
        assert isinstance(max_tokens, list) and len(max_tokens) == n_prompt
        lst_max_tokens = max_tokens
        max_tokens_ = max(max_tokens)

    # try to estimate the total number of tokens near worst case so that APIs don't get halted often
    tc = api.TokenCounter(gpt_model=model_name)
    lens = [tc(p) for p in prompts]
    avg_prompt_len, std_prompt_len = np.mean(lens), np.std(lens)
    exp_total_seq_len_bound = max_tokens_ + avg_prompt_len + std_prompt_len

    tm = Timer()
    if use_openai_parallel:  # use the script provided by OpenAI that maximizes throughput
        from src.util.parallel_api_call import parallel_api_call  # lazy import to save time

        d_log = {
            '#prompt': n_prompt, 'prompt-group-size': group_size,
            'model-name': model_name, 'max-new-tokens': max_tokens, 'temperature': temperature, 'logprobs': logprobs,
            # 'seed': seed,
            'output-dir': output_dir, 'save-all-prompts': save_all_prompts, 'completion-filenames': completion_fnms,
            'average-prompt-length': round(avg_prompt_len), 'prompt-length-std': round(std_prompt_len),
            'expected-total-sequence-length': round(exp_total_seq_len_bound), 'timeout': timeout
        }
        logger.info(f'Writing prompts and completions w/ {pl.i(d_log, indent=1)}')

        _save_prompts(prompts=prompts, output_path=output_dir, save_all=save_all_prompts, completion_fnms=completion_fnms, logger=logger)
        if dir_name2prompts is not None:
            for dir_nm, ppt in dir_name2prompts.items():
                # infer the completion filenames from `completion_fnms`, if a match, use them
                completion_fnms_ = [x for x in completion_fnms if x.startswith(dir_nm)]
                # otherwise, use the default filenames
                if len(completion_fnms_) != len(ppt):
                    completion_fnms_ = [f'completion-{i}.txt' for i in range(1, len(ppt) + 1)]
                _save_prompts(
                    prompts=ppt, output_path=os_join(output_dir, dir_nm), save_all=save_all_prompts, completion_fnms=completion_fnms_, logger=logger)
        _log_prompt_eg(prompt=prompts)
        # raise NotImplementedError

        lst_msgs = [[dict(role='user', content=x)] for x in prompts]
        n_ppt = len(prompts)
        # set incremental seed, to make sure different outputs when using temperature
        # if seed is not None and temperature > 0:
        if seed is not None:
            assert isinstance(seed, int)
            # lst_seed = [seed + i for i in range(n_ppt)]  # seeding seems to make worse diversity in LLM outputs??
            # lst_seed = [seed + i * 20 for i in range(n_ppt)]
            lst_seed = [seed + i * 7 for i in range(n_ppt)]
        else:
            lst_seed = [None] * n_ppt

        # lst_seed = [None] * n_ppt  # no seeding, seems to cause less diversity in LLM outputs
        # lst_seed = [seed] * n_ppt  # TODO: debugging prior results

        lst_args = [
            dict(model=model_name, messages=lst_msg, temperature=temperature, max_tokens=max_tok, logprobs=logprobs, seed=sd)
            for lst_msg, max_tok, sd in zip(lst_msgs, lst_max_tokens, lst_seed)
        ]
        # write prompts to jsonl per OpenAI's script
        requests_fnm = 'requests.jsonl'
        with open(os_join(output_dir, requests_fnm), 'w') as fl:
            for args in lst_args:
                fl.write(json.dumps(args) + '\n')

        use_chat = api.openai_model_use_chat(model_name)
        if use_chat:
            request_url = 'https://api.openai.com/v1/chat/completions'
        else:
            raise NotImplementedError

        if model_name in ['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-1106', 'text-embedding-ada-002']:
            tok_enc_name = 'cl100k_base'
        else:
            raise NotImplementedError
        # tier = 2
        tier = 3
        if tier == 2:
            max_tok_per_min = {'gpt-3.5-turbo': 90_000, 'gpt-3.5-turbo-1106': 90_000}
        else:
            assert tier == 3
            max_tok_per_min = {'gpt-3.5-turbo': 160_000, 'gpt-3.5-turbo-1106': 160_000}
        max_tok_per_min = max_tok_per_min[model_name]
        # a rough guide, assuming each request completes ~1 min, trial and error based on API Rate Limit Error
        max_req_per_min = round(max_tok_per_min / exp_total_seq_len_bound)
        if timeout is None:
            timeout = math.ceil((exp_total_seq_len_bound / 1024) + 0.5) * 60  # timeout proportional to expected seq len
            timeout = max(timeout, 120)  # at least 2 min
        asyncio.run(parallel_api_call(
            requests_filepath=requests_fnm,
            output_directory=output_dir,
            request_url=request_url,
            api_key=api.get_openai_api_key(),
            max_tokens_per_minute=max_tok_per_min,
            max_requests_per_minute=max_req_per_min,
            token_encoding_name=tok_enc_name,
            max_attempts=2**31 - 1,  # set tp max int, keep trying until the error is resolved, ensure each prompt is replied to
            logger=logger,
            n_prompt=n_prompt,
            completion_filenames=completion_fnms,
            is_chat_model=use_chat,
            timeout=timeout,
            logprobs=logprobs
        ))
    else:
        import openai  # lazy import to save time
        if logprobs:
            raise NotImplementedError
        if group_size is None:
            seq_lens = [256, 256 + 128, 512, 768, 1024, 1024 + 512, 2048, 3072]
            # find the smallest seq_len that is larger than the average token count
            max_seq_len_grp = next((x for x in seq_lens if x >= exp_total_seq_len_bound), None)
            if max_seq_len_grp is None:
                raise NotImplementedError(f'average token count {avg_prompt_len}, max token count {max_tokens} may be too large')
            group_size = {256: 32, 256 + 128: 24, 512: 16, 768: 10, 1024: 8, 1024 + 512: 6, 2048: 4, 3072: 3}[max_seq_len_grp]
            logger.info(f'Using default group size {pl.i(group_size)} '
                        f'for expected total sequence length {pl.i(round(exp_total_seq_len_bound))} < {pl.i(max_seq_len_grp)}')
        d_log = {
            '#prompt': n_prompt, 'prompt-group-size': group_size,
            'model-name': model_name, 'max-new-tokens': max_tokens, 'temperature': temperature, 'logprobs': logprobs, 'seed': seed,
            'average-prompt-length': round(avg_prompt_len), 'expected-total-sequence-length': round(exp_total_seq_len_bound),
            'output-dir': output_dir, 'save-all-prompts': save_all_prompts, 'completion-filenames': completion_fnms
        }
        logger.info(f'Writing prompts and completions w/ {pl.i(d_log, indent=1)}')

        prompt_groups = [list(g) for g in group_n(prompts, group_size)]
        _save_prompts(prompts=prompts, output_path=output_dir, save_all=save_all_prompts, completion_fnms=completion_fnms, logger=logger)
        _log_prompt_eg(prompt=prompts)

        glob_idx = 1
        # iterate each prompt group, if error, sleep and retry on the same group
        i_g = glob_idx
        max_tok = max_tokens or 1024
        if openai.api_key is None:
            api.set_openai_api_key()
        while i_g <= len(prompt_groups):
            pg = prompt_groups[i_g-1]
            call_args = GptGenArgs(model_name=model_name, prompt=pg, max_tokens=max_tok, temperature=temperature, logprobs=logprobs, seed=seed)
            try:
                completions = api.openai_completion_call(args=call_args, verbose=True)
            except openai.error.RateLimitError as e:
                # print stack trace and continue generation
                t_sleep = 60
                logger.error(f'RateLimitError: {pl.s(e, c="r")}\nSleeping for {pl.i(t_sleep)}s...')
                time.sleep(t_sleep)
                continue
            except Exception as e:
                t_sleep = 20
                logger.error(f'API call error: {pl.s(e, c="r")}\nSleeping for {pl.i(t_sleep)}s...')
                time.sleep(t_sleep)
                continue

            for i, c in enumerate(completions, start=glob_idx):
                with open(os_join(output_dir, completion_fnms[i-1]), 'w') as fl:
                    fl.write(c)
                logger.info(f'Completion written {pl.i(i)}/{pl.i(n_prompt)}')
            glob_idx += len(prompt_groups[i_g-1])
            i_g += 1
            time.sleep(2)  # sleep to try to avoid API call limit
    t_e = tm.end()
    logger.info(f'Completions done in {pl.i(t_e)}')

    # get the total API querying cost
    # # get the unique completion directories, i.e. the set of direct parent directory corresponding to each completion file
    # cpl_fnms = [os_join(output_dir, x) for x in completion_fnms]
    # cpl_dir_nms = {os.path.dirname(x) for x in cpl_fnms}
    # cpl_dir_nms = list(dict.fromkeys(cpl_dir_nms))  # reduce to unique completion directories
    # # get the relative folder names w.r.t. root generated data directory
    # cpl_dir_nms = [os.path.relpath(x, pu.generated_data_path) for x in cpl_dir_nms]
    #
    # cpl_dir_nms = [x.split(os.sep) for x in cpl_dir_nms]  # split into individual dir names
    # # sanity check there's multiple levels of dir names, the 1st-level dir name being a dataset name
    # assert all(len(x) > 1 for x in cpl_dir_nms)
    # dnms = set()
    # for dir_nms in cpl_dir_nms:
    #     dnm = dir_nms[0].replace('_', '-')
    #     ca(dataset_name=dnm)
    #     dnms.add(dnm)
    # assert len(dnms) == 1  # sanity check the same dataset name
    # dnm = dnms.pop()
    # cpl_dir_nms = [os.sep.join(x[1:]) for x in cpl_dir_nms]  # drop the root-level dataset name & join back to a directory name

    # despite completion-*.txt file write are in subdirectories, the `jsonl` request results will always be in the root directory
    # cpl_dir_nms = output_dir
    # now get the dataset name from the parent directory
    # dirs = os.path.dirname(output_dir).split(os.sep)
    # dnm = dirs[-1].replace('_', '-')
    # sub_dir = None
    p1, p2, fnm = stem(output_dir, top_n=2, as_list=True)
    dnm, sub_dir = p2.replace('_', '-'), None

    dnms_pool = list(sconfig('datasets').keys())
    if dnm not in dnms_pool:  # there's a sub-directory between output dir and dataset name
        # dnm, sub_dir = dirs[-2], dirs[-1]
        dnm, sub_dir = p1, p2
        dnm = dnm.replace('_', '-')
    # if dnm not in dnms_pool:
    #     sic(dnm, dnms_pool)
    assert dnm in dnms_pool  # sanity check found an existing dataset name

    # now, get the querying cost
    cost = api.get_api_query_token_cost(dataset_name=dnm, sub_dir=sub_dir, completion_dir_name=output_dir, model_name=model_name)
    logger.info(f'OpenAI API token querying cost: {pl.i(cost, indent=1)}')
    return WriteCompletionsOutput(output_dir=fnm, output_path=output_dir, cost=cost)


def write_completions(
        output_path: str = None,
        logger: logging.Logger = None,
        init_log: Dict[str, Any] = None,
        add_fl_writer: bool = True,
        completion_type: str = None,
        prompts: List[str] = None,
        dir_name2prompts: Dict[str, List[str]] = None,
        log_callback: Callable[[logging.Logger], None] = None,
        **kwargs
) -> WriteCompletionsOutput:
    """
    syntax sugar

    :param output_path: Output directory
    :param logger: logger, a logging file will be written
    :param init_log: initial log message
    :param add_fl_writer: whether to add a file handler to the logger
    :param completion_type: completion type, e.g. `sample`, `sentence`
    :param prompts: prompts to generate completions
    :param dir_name2prompts: If provided, save partitions of the prompts to each subdirectory
    :param log_callback: callback function to log additional info after file handler is added
    :param kwargs: other arguments for `api_util.write_completions`
    """
    if add_fl_writer:
        add_file_handler(logger=logger, file_path=os_join(output_path, 'write-completion.log'))
    tp = f'{completion_type} ' if completion_type else ''
    msg = f'Generating {tp}completions'
    if init_log:
        msg = f'{msg} w/ {pl.i(init_log, indent=1)}'
    logger.info(f'{msg}...')
    if log_callback is not None:
        log_callback(logger)

    return _write_completions(prompts=prompts, dir_name2prompts=dir_name2prompts, output_dir=output_path, logger=logger, **kwargs)


# Each returned completion file name is e.g. `completion-1.txt`
pattern_completion_file = re.compile(r'^completion-(?P<idx>\d+)$')


def completion_file2index(fnm: str = None) -> int:
    m = pattern_completion_file.match(stem(fnm))
    assert m is not None
    return int(m.group('idx'))


def completion_dir_name2file_paths(path: str = None) -> List[str]:
    if not os.path.isdir(path):
        raise ValueError(f'Directory {pl.i(path)} not found, make sure you are in the correct working directory.')
    return sorted(glob.iglob(os_join(glob.escape(path), '*.txt')), key=completion_file2index)


@dataclass
class CompletionFile:
    path: str = None
    filename: str = None
    filename_w_dir_name: str = None
    pretty_filename: str = None
    content: str = None
    logprobs: List[Dict[str, Any]] = None


@dataclass
class CompletionIter:
    filepaths: List[str] = None
    iter: Iterable[CompletionFile] = None


def iter_completions(
        dir_name: CompletionDirectory, base_path: str = None, logger: logging.Logger = None, completion_type: str = None,
        logprobs: bool = False
) -> CompletionIter:
    cpl_paths = [dir_name] if isinstance(dir_name, str) else dir_name
    cpl_paths = [os_join(base_path, dnm) for dnm in cpl_paths]
    file_paths = sum((completion_dir_name2file_paths(path=pa) for pa in cpl_paths), start=[])
    logprobs_paths = None
    if logprobs:
        logprobs_paths = [os_join(os.path.dirname(fp), 'logprobs', f'{stem(fp)}.json') for fp in file_paths]
    logger.info(f'Processing {pl.i(len(file_paths))} completion files...')

    def gen():
        if logprobs:
            assert len(file_paths) == len(logprobs_paths)
            paths = zip(file_paths, logprobs_paths)
        else:
            paths = file_paths
        it = tqdm(paths, desc=f'Processing {pl.i(completion_type)} completion files', unit='fl')
        for path in it:
            fp, lp = path if logprobs else (path, None)
            _dir_nm = os.path.basename(os.path.dirname(fp))
            fnm_stem = stem(fp)
            fnm_dir_nm = f'{_dir_nm}/{fnm_stem}'
            fnm_log = f'{pl.i(_dir_nm)}/{pl.i(fnm_stem)}'
            it.set_postfix(fnm=fnm_log)
            logger.info(f'Processing completion file {pl.i(fnm_log)}...')

            with open(fp, 'r') as fl:
                completion = fl.read()
            logprobs_ = None
            if logprobs:
                lp: str
                with open(lp, 'r') as fl:
                    logprobs_ = json.load(fl)
            yield CompletionFile(
                path=fp, filename=fnm_stem, filename_w_dir_name=fnm_dir_nm, pretty_filename=fnm_log, content=completion, logprobs=logprobs_)
    return CompletionIter(filepaths=file_paths, iter=gen())


def process_completions_init(
        completion_base_path: str = None,
        completions_dir_name: str = None,
        output_path: str = None,
        completion_type: str = 'NER',
        logger: logging.Logger = None,
        init_log: Dict[str, Any] = None,
        add_fl_writer: bool = True,
        logprobs: bool = False
) -> CompletionIter:
    """
    Initializes completion file processing, in particular
        1) sets up logger & output directory
        2) finds completion files
        3) iterate over completion file contents
    """

    logger = logger or _logger
    if add_fl_writer:
        add_file_handler(logger=logger, file_path=os_join(output_path, f'write-{completion_type.lower()}.log'))
    tp = f'{completion_type} ' if completion_type else ''
    tp = pl.i(tp)
    msg = f'Processing {tp}completions'
    if init_log:
        msg = f'{msg} w/ {pl.i(init_log, indent=1)}'
    logger.info(f'{msg}...')

    if completions_dir_name is not None:
        return iter_completions(
            dir_name=completions_dir_name, base_path=completion_base_path, logger=logger, completion_type=completion_type, logprobs=logprobs
        )


def completion2lines(completion: str = None) -> List[str]:
    return [ln.strip() for ln in completion.split('\n') if ln.strip()]
