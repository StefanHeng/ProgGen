import argparse
from typing import Dict, Iterable, Union, Any


__all__ = [
    'optimize_imports',
    'add_argument', 'get_chat_completion_args', 'process_args',
    'DATASET_NAME2TOPIC_DIM'
]


import os


def optimize_imports():
    """
    optimize imports for `stefutil`
    """
    for k in ['SU_USE_PLT', 'SU_USE_ML', 'SU_USE_DL']:
        os.environ[k] = 'F'


def add_argument(parser: argparse.ArgumentParser = None, arg: Union[str, Iterable[str]] = None):
    """
    For shared argument addition across scripts
    """
    if isinstance(arg, str):
        # about original dataset
        if arg == 'dataset_name':
            dnms = ['conll2003-no-misc', 'wiki-gold-no-misc', 'mit-movie', 'mit-restaurant']
            parser.add_argument(
                '--dataset_name', type=str, required=False, default=None, choices=dnms,
                help='Dataset Name. One of the 4 datasets studied.'
            )
        elif arg == 'n_demo':
            parser.add_argument(
                '--n_demo', type=int, default=1,
                help='`n` as in n-shot demo as setup in the paper, where n is the number of occurrences for each entity class.'
            )
        elif arg == 'include_negative_sample':
            parser.add_argument(
                '--include_negative_sample', type=int, default=-1,
                help='Flag (> 0 or not) for whether to include additional `n_demo` (see `n_demo` above) negative samples as defined in the paper. '
                     'If a negative number is given, '
                     'defaults to 1 (True) for `conll2003-no-misc` and `wiki-gold-no-misc` and 0 (False) for `mit-movie` and `mit-restaurant`.'
            )

        # about prompt construction
        elif arg == 'prompt_seed':
            parser.add_argument(
                '--prompt_seed', type=int, required=False, default=42,
                help='Seed for prompt construction.'
            )

        # about OpenAI Chat Completion API
        elif arg == 'chat_model_name':
            parser.add_argument(
                '--chat_model_name', type=str, required=False, default='gpt-3.5-turbo-1106',
                help='Name of the OpenAI API Chat Completion model to use.'
            )
        elif arg == 'chat_max_tokens':
            parser.add_argument(
                '--chat_max_tokens', type=int, required=False, default=1024,
                help='Max number of tokens to generate for each prompt in the OpenAI API Chat Completion.'
            )
        elif arg == 'chat_temperature':
            parser.add_argument(
                '--chat_temperature', type=float, required=False, default=1,
                help='Temperature for OpenAI API Chat Completion.'
            )
        elif arg == 'chat_seed':
            parser.add_argument(
                '--chat_seed', type=int, required=False, default=42,
                help='Seed for OpenAI API Chat Completion.'
            )
        elif arg == 'chat_logprobs':
            parser.add_argument(
                '--chat_logprobs', type=bool, required=False, default=False,
                help='Flag for whether to include logprobs in OpenAI API Chat Completion.'
            )
        elif arg == 'chat_timeout':
            parser.add_argument(
                '--chat_timeout', type=int, required=False, default=30,
                help='Timeout for each OpenAI API call before retry.'
            )

        # about NER sample processing
        elif arg == 'lowercase':
            parser.add_argument(
                '--lowercase', type=int, default=-1,
                help='Flag (> 0 or not) for whether to lowercase the samples (before deduplication and downstream model training). '
                     'If a negative number is given, '
                     'defaults to 0 (False) for `conll2003-no-misc` and `wiki-gold-no-misc` and 1 (True) for `mit-movie` and `mit-restaurant`.'
            )
        else:
            raise NotImplementedError
    else:
        for a in arg:
            add_argument(parser=parser, arg=a)
    return parser


chat_completion_keys = ['chat_model_name', 'chat_max_tokens', 'chat_temperature', 'chat_seed', 'chat_logprobs', 'chat_timeout']


def get_chat_completion_args(args: argparse.Namespace = None) -> Dict[str, Any]:
    """
    Get arguments for OpenAI Chat Completion API
    """
    ret = dict()
    for k in chat_completion_keys:
        if hasattr(args, k):
            ret[k[5:]] = getattr(args, k)  # drop the starting `chat_`
    return ret if ret != dict() else None


def process_args(args: argparse.Namespace = None) -> argparse.Namespace:
    """
     Shared processing
        including dynamically setup default values
    """
    if hasattr(args, 'include_negative_sample') and args.include_negative_sample < 0:
        if args.dataset_name in ['conll2003-no-misc', 'wiki-gold-no-misc']:
            args.include_negative_sample = 1
        else:
            assert args.dataset_name in ['mit-movie', 'mit-restaurant']  # sanity check
            args.include_negative_sample = 0
        args.include_negative_sample = bool(args.include_negative_sample)
    if hasattr(args, 'lowercase') and args.lowercase < 0:
        if args.dataset_name in ['conll2003-no-misc', 'wiki-gold-no-misc']:
            args.lowercase = 0
        else:
            assert args.dataset_name in ['mit-movie', 'mit-restaurant']
            args.lowercase = 1
        args.lowercase = bool(args.lowercase)

    chat_args = get_chat_completion_args(args)
    if chat_args is not None:
        setattr(args, 'chat_args', chat_args)
    return args


# the dataset-independent `topic` attribute dimension (internal) name for each dataset
DATASET_NAME2TOPIC_DIM = {
    'conll2003-no-misc': 'news-category',
    'wiki-gold-no-misc': 'topic',
    'mit-movie': 'query-category',
    'mit-restaurant': 'meal-category'
}
