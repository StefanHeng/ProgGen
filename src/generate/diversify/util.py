import os
import re
import random
from os.path import join as os_join
from typing import List, Dict, Union, Iterable, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from stefutil import *
from src.util import *
from src.util import api
from src.data_util import *


__all__ = [
    'ATTR_PROMPT_DNM', 'DIVERSE_CONTEXT_DNM', 'DIVERSE_ENTITY_DNM', 'DIVERSE_DNM',
    'ENTITY_KEY', 'ENTITY_KEY_SEEDED', 'ENTITY_KEY_MIXED', 'ENTITY_TYPE_DEFAULT',
    'N_List', 'GenSetup', 'OptionGenerator'
]


ATTR_PROMPT_DNM = sconfig('sub-directory-names.attr-prompt')  # for legacy diverse context
DIVERSE_CONTEXT_DNM = sconfig('sub-directory-names.diverse-context')
DIVERSE_ENTITY_DNM = sconfig('sub-directory-names.diverse-entity')
DIVERSE_DNM = sconfig('sub-directory-names.diversity')


ENTITY_KEY = '__ENTITIES__'  # Attribute key for named entities; underscores to prevent name conflict
ENTITY_KEY_SEEDED = '__ENTITIES-SEEDED__'  # seeded from category
ENTITY_KEY_MIXED = '__ENTITIES-MIXED__'  # mix vanilla and seeded entities
ENTITY_TYPE_DEFAULT = '__default__'  # default entity type for diverse entity


N_List = Union[int, Dict[str, int]]


@dataclass
class GenSetup:
    get_prompt_args: Dict[str, Any] = None  # arguments for `get_prompt`
    n_list: int = None  # #options to generate
    output_dir: str = None  # relative output directory
    output_key: Union[str, List[str]] = None  # key for output file, from `keyword` or [seed-keyword, keyword]
    max_tokens: int = None  # max tokens for GPT completion


@dataclass
class ProcessCompletionOutput:
    output_path: str = None
    key2value: Dict[str, Any] = None
    key2size: Dict[str, Any] = None


class OptionGenerator:
    """
    Shared functionalities for diverse context & diverse entity
    """
    def __init__(self, dataset_name: str = 'conll2003-no-misc', keyword2n_list: Dict[str, N_List] = None, keywords: List[str] = None):
        """
        :param dataset_name: Dataset name
        :param keyword2n_list: Maps from keyword, either attribute name or entity type for diverse context and entity respectively,
            to number of options to generate
        :param keywords: List of keywords to generate options for
        """
        self.dataset_name = dataset_name

        self.keyword2n_list = keyword2n_list
        self.keywords = keywords

        # to be defined in subclass
        self.dir_args, self.sample_type = None, None
        self.logger = None
        self.ec = prettier.EdgeCases()

    def get_prompt(
            self, n_list: int = None, entity_type: str = None, seed_category: str = None, generator: Union[int, random.Random] = None
    ) -> str:
        raise NotImplementedError

    def get_n_list_map(self, n_list: N_List = None) -> Dict[str, int]:
        ret = self.keyword2n_list.copy()
        if isinstance(n_list, int):
            ret = {k: n_list for k in self.keywords}
        elif isinstance(n_list, dict):
            assert all(k in self.keywords for k in n_list.keys())  # sanity check
            ret.update(n_list)
        else:
            assert n_list is None
        return ret

    def _iter_setups(
            self, n_list: N_List = None, seeded_keywords: List[str] = None, key_name: str = None,
            max_tokens_multiplier: Union[int, float] = None
    ) -> Iterable[GenSetup]:
        d_n_list = self.get_n_list_map(n_list=n_list)
        max_tokens_multiplier = max_tokens_multiplier or 20
        if seeded_keywords is None:
            for k in self.keywords:
                nl = d_n_list[k]
                args = {key_name: k, 'n_list': nl}
                # separate sub-folder for each entity
                # generating 100 CoNLL-03 org entities had total seq len 1K, so ~10 token per entity;
                # x20 as a safe token upper bound, since GPT may generate descriptions
                yield GenSetup(get_prompt_args=args, n_list=nl, output_dir=k, output_key=k, max_tokens=round(nl * max_tokens_multiplier))
        else:  # TODO: dynamic seeded keywords for each keyword
            for seed in seeded_keywords:
                for k in self.keywords:
                    nl = d_n_list[k]
                    args = {key_name: k, 'seed_category': seed, 'n_list': nl}
                    max_tokens = round(nl * max_tokens_multiplier)
                    out_dir = os_join(seed, k)
                    yield GenSetup(get_prompt_args=args, n_list=nl, output_dir=out_dir, output_key=[seed, k], max_tokens=max_tokens)

    def iter_setups(self, n_list: N_List = None) -> Iterable[GenSetup]:
        raise NotImplementedError

    def key2prompt_args(self, key: str) -> Dict[str, Any]:
        raise NotImplementedError

    def _write_completions(
            self, n_prompt: int = 5, n_list: N_List = None, prompt_seed: int = None, output_dir_nm: str = None, max_tokens: int = None,
            init_log: Dict[str, Any] = None, **kwargs
    ) -> completions.WriteCompletionsOutput:
        """
        :param n_prompt: # Prompt for each entity type
        :param n_list: # Entity to generate in each prompt
        :param prompt_seed: Seed for random prompt generation
        :param output_dir_nm: Output directory name
        :param max_tokens: Max tokens for each completion
        :param init_log: Initial log for `write_completions`
        :param kwargs: Additional arguments to `write_completions`
        """
        out_dir = f'{self.sample_type}-Res'
        meta = self.meta
        if meta:
            out_dir = f'{out_dir}_{meta}'
        output_dir = dataset_name2data_dir(**self.dir_args, output_dir=out_dir, output_postfix=output_dir_nm).path
        prompts, fnms, max_tokens_ = [], [], []

        setup2fnm_idx = defaultdict(int)  # for pool of seed category may contain duplicate keys
        gen = get_random_generator(generator=prompt_seed)
        for s in self.iter_setups(n_list=n_list):
            args = s.get_prompt_args
            self.logger.info(f'Constructing prompt w/ {pl.i(args)}...')
            # gen entities may include demo at probability => randomized
            prompts += [self.get_prompt(**args, generator=gen) for _ in range(n_prompt)]
            max_tokens_ += [max_tokens or s.max_tokens] * n_prompt
            os.makedirs(os_join(output_dir, s.output_dir), exist_ok=True)

            i_strt = setup2fnm_idx[s.output_dir]
            fnms += [os_join(s.output_dir, f'completion-{i+i_strt+1}.txt') for i in range(n_prompt)]
            setup2fnm_idx[s.output_dir] += n_prompt
        return completions.write_completions(
            output_path=output_dir, logger=self.logger,
            completion_type=self.sample_type,
            init_log={'dataset-name': self.dataset_name, '#example requested': n_list, **(init_log or dict())},
            prompts=prompts, completion_fnms=fnms, max_tokens=max_tokens_, **kwargs
        )

    def _match_values(
            self, text: str = None, n_expect: int = None, pattern: patterns.Patterns = None, has_enum_prefix: bool = None, **kwargs
    ) -> List[re.Match]:
        if text[-1] != '\n':
            text += '\n'
        min_ma = kwargs.pop('min_match', round(n_expect * 0.8))
        ms: List[re.Match] = patterns.find_non_overlap_matches(pattern=pattern, text=text, return_matches=True, min_match=min_ma, **kwargs)
        has_enum_prefix = has_enum_prefix or completions.completion_has_enum_prefix(completion=text)
        if has_enum_prefix:
            split.check_match_group_indices(ms=ms, text=text, logger=self.logger, ec=self.ec)

        n = len(ms)
        if n_expect is not None and n != n_expect:
            self.logger.info(f'Expected {pl.i(n_expect)} values, but got {pl.i(n)}')
        return ms

    def extract_values(self, text: str = None, n_expect: int = None) -> List[str]:
        raise NotImplementedError

    def meta(self) -> Optional[str]:
        raise NotImplementedError

    def _process_completions(
            self, completions_dir_name: str = None, expected_samples_per_completion: N_List = None, output_dir_name: str = None,
            drop_duplicates: Union[bool, Dict[str, Any]] = None
    ):
        out_dir = f'{self.sample_type}-Dataset'
        meta = self.meta
        if meta:
            out_dir = f'{out_dir}_{meta}'
        out = dataset_name2data_dir(**self.dir_args, output_dir=out_dir, output_postfix=output_dir_name, timestamp='short-date')
        output_path, base_path = out.path, out.base_path
        init_log = {
            'class-name': self.__class__.__qualname__, 'output-path': output_path,
            'completions-dir-name': completions_dir_name, 'expected-samples-per-completion': expected_samples_per_completion
        }
        proc_args = dict(logger=self.logger, completion_type=self.sample_type)
        completions.process_completions_init(completion_base_path=out.base_path, output_path=output_path, init_log=init_log, **proc_args)

        key2val = dict()
        seeded = False

        for s in self.iter_setups(n_list=expected_samples_per_completion):
            k = s.output_key
            it = completions.iter_completions(dir_name=s.output_dir, base_path=os_join(base_path, completions_dir_name), **proc_args)
            enms = []
            for c in it.iter:
                enms_ = self.extract_values(text=c.content, n_expect=s.n_list)
                enms.extend(enms_)

            if drop_duplicates is None:
                drop_dup = True
            elif isinstance(drop_duplicates, bool):
                drop_dup = drop_duplicates
            else:
                assert isinstance(drop_duplicates, dict)
                drop_dup = drop_duplicates.get(k, True)

            if drop_dup:
                n_enm = len(enms)
                n_uniq = len(set(enms))
                if n_enm > n_uniq:
                    enms = list(dict.fromkeys(enms))  # preserve the order and drop duplicates
                    self.logger.info(f'Found {pl.i(n_enm - n_uniq)} duplicate {pl.i(k)} values and dropped')

            if isinstance(k, list):
                assert len(k) == 2
                k1, k2 = k
                seeded = True
                key2val[k1] = key2val.get(k1, dict())
                key2val[k1][k2] = enms
            else:
                assert isinstance(k, str)
                key2val[k] = enms

        def _get_key2size(k2v: Dict[str, List[str]]) -> Dict[str, int]:
            ret = {k: len(v) for k, v in k2v.items()}
            # get average & total
            n = sum(ret.values())
            avg = round(n / len(ret), 1)
            ret['__average__'] = avg
            ret['__total__'] = n
            return ret
        if seeded:
            # d_sz = {k1: {k2: len(enms) for k2, enms in v.items()} for k1, v in key2val.items()}
            d_sz = {k1: _get_key2size(v) for k1, v in key2val.items()}
        else:
            # d_sz = {et: len(enms) for et, enms in key2val.items()}
            d_sz = _get_key2size(key2val)
        return ProcessCompletionOutput(output_path=output_path, key2value=key2val, key2size=d_sz)


