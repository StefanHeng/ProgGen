"""
Utility functions for data generation
"""


import os
import re
import json
import asyncio
import argparse
from os.path import join as os_join
from typing import List, Tuple, Dict, Any, Union
from dataclasses import dataclass

import numpy as np

from stefutil import *
from src.util import *


__all__ = [
    'get_openai_api_key', 'set_openai_api_key', 'TokenCounter', 'tc35', 'get_api_query_token_cost',
    'dispatch_openai_requests', 'openai_completion_call'
]


_logger = get_logger(__name__)


def get_openai_api_key() -> str:
    # first try if there's an os env var
    if 'OPENAI_API_KEY' in os.environ:
        return os.environ['OPENAI_API_KEY']
    else:  # must have one in the project
        path = os_join(pu.proj_path, 'auth', 'openai-api-key.json')
        if not os.path.exists(path):
            raise ValueError('OpenAI API key not found. '
                             'Either set the env var `OPENAI_API_KEY` '
                             'or create a file `openai-api-key.json` in the `auth` directory')
        with open(path, 'r') as f:
            return json.load(f)['api-key']


def set_openai_api_key():
    import openai  # lazy import to save time
    openai.api_key = get_openai_api_key()


class TokenCounter:
    """
    Count the number of tokens for OpenAI API call

    Intended for estimating API call cost
    """

    def __init__(self, gpt_model: str = 'gpt-3.5-turbo'):
        self.gpt_model = gpt_model
        self._enc = None

    @property
    def enc(self):
        if self._enc is None:
            import tiktoken  # lazy import to save time
            self._enc = tiktoken.encoding_for_model(self.gpt_model)
        return self._enc

    def __call__(self, text: str) -> int:
        return len(self.enc.encode(text))


# token count for gpt-3.5-turbo
tc35 = TokenCounter()


def openai_model_use_chat(model_name: str = 'gpt-3.5-turbo') -> bool:
    """
    Check if the model is a chat model
    """
    return ('gpt-3.5' in model_name or 'gpt-4' in model_name) and model_name != 'gpt-3.5-turbo-instruct'


async def dispatch_openai_requests(
    messages_list: List[Union[str, List[Dict[str, Any]]]],
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> Union[List[str], Tuple[str]]:
    """Dispatches requests to OpenAI API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model_name: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
    Returns:
        List of responses from OpenAI API.
    """
    import openai  # lazy import to save time
    if openai_model_use_chat(model_name):
        async_responses: List = [
            openai.ChatCompletion.acreate(
                model=model_name,
                messages=x,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for x in messages_list
        ]
    else:
        async_responses: List = [
            openai.Completion.acreate(
                model=model_name,
                prompt=x,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for x in messages_list
        ]
    return await asyncio.gather(*async_responses)


def openai_completion_call(args: Union[argparse.Namespace, Any], n_call: int = None, verbose: bool = False) -> List[str]:
    """
    Calls OpenAI API for text completion
    """

    is_str = isinstance(args.prompt, str)
    if not is_str:
        assert isinstance(args.prompt, (list, tuple))
        if n_call is not None:
            raise ValueError(f'{pl.i("n_call")} must be {pl.i("None")} when {pl.i("args.prompt")} is already a sequence of prompts')

    tm = None
    if verbose:
        d_log = {k: v for k, v in vars(args).items() if k != 'prompt'}
        d_log['#prompt'] = 1 if is_str else len(args.prompt)
        _logger.info(f'Calling OpenAI Completion API w/ {pl.i(d_log)}...')
        tm = Timer()

    md_nm = args.model_name
    if openai_model_use_chat(md_nm):
        if is_str:
            msg_lst = [[{"role": "user", "content": args.prompt}], ] * (n_call or 20)
        else:
            msg_lst = [[{"role": "user", "content": x}] for x in args.prompt]
        response: List = asyncio.run(
            dispatch_openai_requests(
                messages_list=msg_lst,
                model_name=md_nm,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
        )
        ret = [x['choices'][0]['message']['content'] for x in response]
    else:  # other text completion model
        if is_str:
            msg_lst = [args.prompt] * (n_call or 30)
        else:
            msg_lst = args.prompt
        response: List = asyncio.run(
            dispatch_openai_requests(
                messages_list=msg_lst,
                model_name=md_nm,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
        )
        ret = [x['choices'][0]['text'] for x in response]
    if verbose:
        _logger.info(f'Completed {pl.i(len(ret))} calls in {pl.i(tm.end())}')
    return ret


@dataclass
class TokenCountOutput:
    n_chat: int = None
    n_prompt_tokens: List[int] = None
    n_completion_tokens: List[int] = None


_pat_ppt = re.compile(r'prompt(\d+).txt')
_pat_res = re.compile(r'response(\d+).txt')


def _path2token_counts(path: str = None) -> TokenCountOutput:
    toks_ppt, toks_cpl = [], []

    res_path = os_join(path, 'requests_results.jsonl')
    if os.path.exists(res_path):
        with open(res_path, 'r') as fl:
            chats = [json.loads(x) for x in fl.readlines()]
        for chat in chats:
            assert len(chat) == 2
            req, res = chat

            ppts = req['messages']
            assert len(ppts) == 1  # sanity check
            ppt = ppts[0]
            assert ppt['role'] == 'user'  # sanity check
            # ppt = ppt['content']  # no need to get prompt, just use the token counts in the response

            usage = res['usage']
            n_ppt, n_cpl = usage['prompt_tokens'], usage['completion_tokens']
            toks_ppt.append(n_ppt)
            toks_cpl.append(n_cpl)
        n_chat = len(chats)
    else:  # for locally copied version of prompts & responses from web-based ChatGPT
        # must adhere to this API
        #   prompts are either many `prompt{number}.txt` files or a single `prompt.txt` file if the same prompt
        #   responses are formatted as `response{number}.txt` files
        #   the number of prompt & response files should match
        ppt_files = [x for x in os.listdir(path) if _pat_ppt.match(x)]
        res_files = [x for x in os.listdir(path) if _pat_res.match(x)]
        n_ppt_fl, n_res_fl = len(ppt_files), len(res_files)
        if n_ppt_fl == 0:
            path_ppt = os_join(path, 'prompt.txt')
            assert os.path.exists(path_ppt)
            with open(path_ppt, 'r') as fl:
                ppt = fl.read()
            toks_ppt = [tc35(ppt)] * len(res_files)
        else:
            assert n_ppt_fl == n_res_fl
            for ppt_file in ppt_files:
                with open(os_join(path, ppt_file), 'r') as fl:
                    ppt = fl.read()
                toks_ppt.append(tc35(ppt))

        for res_file in res_files:
            with open(os_join(path, res_file), 'r') as fl:
                res = fl.read()
            toks_cpl.append(tc35(res))
        n_chat = len(res_files)
    return TokenCountOutput(n_chat=n_chat, n_prompt_tokens=toks_ppt, n_completion_tokens=toks_cpl)


def get_api_query_token_cost(
        dataset_name: str = None, completion_dir_name: Union[str, List[str]] = None, sub_dir: str = None, model_name: str = 'gpt-3.5-turbo-1106'
) -> Dict[str, Any]:
    """
    Compute the exact cost of token querying for a completion directory
        Intended to process requests saved via `parallel API requests`, see `write_completions`
    """
    # can just use the `response.usage`, simple, no need to actually use the openai tokenizer
    # cost of 1K tokens from openai pricing; as in `$0.0010 / 1K tokens`
    type2price = {
        'gpt-3.5-turbo-1106': {'prompt': 0.0010, 'completion': 0.0020},
        'gpt4': {'prompt': 0.03, 'completion': 0.06},
    }
    if model_name not in type2price:
        raise NotImplementedError

    if not isinstance(completion_dir_name, list):
        completion_dir_name = [completion_dir_name]
    paths = [dataset_name2data_dir(dataset_name=dataset_name, sub_dir=sub_dir, input_dir=dir_nm).path for dir_nm in completion_dir_name]
    n_chat, toks_ppt, toks_cpl = 0, [], []
    for path in paths:
        out = _path2token_counts(path=path)
        n_chat += out.n_chat
        toks_ppt += out.n_prompt_tokens
        toks_cpl += out.n_completion_tokens

    def n_token2price(n_tok: int, kind: str = 'prompt') -> float:
        return n_tok * type2price[model_name][kind] / 1000

    def pretty_price(price: float) -> str:
        d = 4 if price < 0.01 else 2
        return f'${round(price, d)}'

    def kind2d(toks: List[int], kind: str = 'prompt') -> Tuple[float, Dict[str, Any]]:
        # get total #token & average #token per chat, also the price
        avg, n_tok = np.mean(toks), sum(toks)
        price = n_token2price(n_tok=n_tok, kind=kind)
        return price, {'average-#token': round(avg, 2), 'total-#token': fmt_num(n_tok), 'token-query-cost': pretty_price(price)}

    prc_ppt, d_tok_ppt = kind2d(toks_ppt, kind='prompt')
    prc_cpl, d_tok_cpl = kind2d(toks_cpl, kind='completion')
    total_price = prc_ppt + prc_cpl
    ret = dict(prompt=d_tok_ppt, completion=d_tok_cpl)
    ret = {k2: {k1: v1 for k1, v1 in v2.items()} for k2, v2 in ret.items()}  # reverse the ordering of 1st and 2nd level keys
    dir_nms_log = [stem(d, top_n=1) for d in completion_dir_name]
    if len(dir_nms_log) == 1:
        dir_nms_log = dir_nms_log[0]
    ret = {'completion-dir-name': dir_nms_log, '#chat': n_chat, **ret, 'total-token-query-cost': pretty_price(total_price)}
    return ret
