"""
Utilities for formatting samples
"""

import re
import json
from typing import List, Tuple, Optional

from stefutil import *


__all__ = [
    'EntityPairTemplate', 'EntityPairJoin',
    'EntityPairEnclose', 'EntityPairEncloseType', 'EntityPairEnclosePair', 'EntityPairEncloseBoth', 'get_default_entity_pair_map',
    'TokenMapEnclose', 'TokenMapIdentity', 'get_default_token_map',
    'get_default_entity_sep',
]


class EntityPairTemplate:
    """
    Maps entity name and type to natural language in demos

    An abstract class
    """
    def __call__(self, entity_name: str, entity_type: str) -> str:
        raise NotImplementedError

    def decode(self, pair: str) -> Tuple[str, str]:
        raise NotImplementedError

    def entity_type_missing(self, pair: str) -> bool:
        raise NotImplementedError


class EntityPairJoin(EntityPairTemplate):
    def __init__(self, join_char: str = ':'):
        self.join_char = join_char

        # # add match group for each entity name and type
        # pattern_entity = re.compile(r'(?P<entity_name>.+): (?P<entity_type>.+)')

    def __call__(self, entity_name: str, entity_type: str) -> str:
        return f'{entity_name}{self.join_char} {entity_type}'

    def decode(self, pair: str) -> Tuple[str, str]:
        ret = pair.split(f'{self.join_char} ')  # see `__call__` above
        assert len(ret) == 2
        return ret[0], ret[1]


class EntityPairEnclose(EntityPairTemplate):
    def __init__(self, open_char: str = '[', close_char: str = ']'):
        self.open_char = open_char
        self.close_char = close_char

        self.pattern_decode = None

    def __call__(self, entity_name: str, entity_type: str) -> str:
        raise NotImplementedError

    def decode(self, pair: str) -> Tuple[str, str]:
        patterns = self.pattern_decode if isinstance(self.pattern_decode, list) else [self.pattern_decode]

        m = None
        for p in patterns:
            m = p.match(pair)
            if m is not None:
                break
        if m is None:
            raise ValueError(f'Cannot decode entity pair `{pl.i(pair)}` w/ pattern `{pl.i(self.pattern_decode)}`')
        return m.group('name'), m.group('type')

    def entity_type_missing(self, pair: str) -> bool:
        # for edge case, see `Completion2Samples::__call__`
        return self.open_char not in pair and self.close_char not in pair


class EntityPairEncloseType(EntityPairEnclose):
    """
    Intended for format `natural-pair`
    """
    def __init__(self, open_char: str = '[', close_char: str = ']'):
        super().__init__(open_char=open_char, close_char=close_char)

        _re_open, _re_close = re.escape(open_char), re.escape(close_char)
        self.pattern_decode = [
            re.compile(f'^(?P<name>.+) {_re_open}(?P<type>.+){_re_close}$'),
            # for edge case: white space missing between entity name and entity type, e.g. `pollution(object)`
            re.compile(f'^(?P<name>.+){_re_open}(?P<type>.+){_re_close}$'),
            # for edge case: missing enclose char, e.g. `pollution(object`
            re.compile(f'^(?P<name>.+) {_re_open}(?P<type>.+)$')
        ]

    def __call__(self, entity_name: str, entity_type: str) -> str:
        # `name [entity_type]`
        return f'{entity_name} {self.open_char}{entity_type}{self.close_char}'


class EntityPairEnclosePair(EntityPairEnclose):
    """
    Intended for format `natural-inline`

    `[name: entity_type]`
    e.g. `[BRUSSELS: location]`
    """
    def __init__(self, open_char: str = '[', close_char: str = ']', sep: str = ':'):
        super().__init__(open_char=open_char, close_char=close_char)
        self.sep = sep

        _re_open, _re_close = re.escape(open_char), re.escape(close_char)
        _re_sep = re.escape(sep)
        self.pattern_decode = re.compile(f'^{_re_open}(?P<name>.+){_re_sep} (?P<type>.+){_re_close}$')
        # for search, drop begin & end of sequence, and add `?` for non-greedy match, look for only non-overlapping matches
        self.pattern_decode_search = re.compile(f'{_re_open}(?P<name>.*?){_re_sep} (?P<type>.*?){_re_close}')

    def __call__(self, entity_name: str, entity_type: str) -> str:
        return f'{self.open_char}{entity_name}{self.sep} {entity_type}{self.close_char}'


class EntityPairEncloseBoth(EntityPairTemplate):
    """
    Intended for format `natural-inline-v2`

    `**name** (entity_type)`
    e.g. **BRUSSELS** (location)
    """
    def __init__(
            self, open_char_name: str = '**', close_char_name: str = '**', open_char_type: str = '(', close_char_type: str = ')',
            sep: str = ' '
    ):
        self.open_char_name, self.close_char_name = open_char_name, close_char_name
        self.open_char_type, self.close_char_type = open_char_type, close_char_type
        self.sep = sep

        _re_o_n, _re_c_n = re.escape(open_char_name), re.escape(close_char_name)
        _re_o_t, _re_c_t = re.escape(open_char_type), re.escape(close_char_type)
        _re_sep = re.escape(sep)
        self.pattern_decode = re.compile(f'^{_re_o_n}(?P<name>.+){_re_c_n}{_re_sep}{_re_o_t}(?P<type>.+){_re_c_t}$')
        self.pattern_decode_search = re.compile(f'{_re_o_n}(?P<name>.*?){_re_c_n}{_re_sep}{_re_o_t}(?P<type>.*?){_re_c_t}')

    def __call__(self, entity_name: str, entity_type: str) -> str:
        return f'{self.open_char_name}{entity_name}{self.close_char_name}{self.sep}{self.open_char_type}{entity_type}{self.close_char_type}'

    def decode(self, pair: str) -> Tuple[str, str]:
        m = self.pattern_decode.match(pair)
        if m is None:
            raise ValueError(f'Cannot decode entity pair `{pl.i(pair)}` w/ pattern `{pl.i(self.pattern_decode.pattern)}`')
        return m.group('name'), m.group('type')

    # def clean_sentence(self, sentence: str) -> str:
    #     """
    #     :param sentence: Sentence w/ entity names and types annotated inline
    #     :return: Sentence with just the entity names
    #     """


def get_default_entity_pair_map(sample_format: str) -> Optional[EntityPairTemplate]:
    if 'natural' in sample_format:
        if 'pair' in sample_format:
            if sample_format == 'natural-pair':
                o, c = '[', ']'
            else:
                assert sample_format == 'natural-pair-v2'
                o, c = '(', ')'
            return EntityPairEncloseType(open_char=o, close_char=c)
        else:
            assert 'inline' in sample_format
            assert sample_format in ['natural-inline', 'natural-inline-v2']
            if sample_format == 'natural-inline':
                return EntityPairEnclosePair()
            else:
                assert sample_format == 'natural-inline-v2'
                return EntityPairEncloseBoth()
    else:  # `list-bio` or `line-bio`
        return None


class TokenMapTemplate:
    def __call__(self, token: str) -> str:
        raise NotImplementedError

    def decode(self, token: str) -> str:
        raise NotImplementedError


class TokenMapIdentity(TokenMapTemplate):
    """
    That is, no change
    """
    def __call__(self, token: str) -> str:
        return token

    def decode(self, token: str) -> str:
        return token


class TokenMapEnclose(TokenMapTemplate):
    """
    Similar to`EntityPairTemplate`, for extracting word from enclosing brackets
    e.g. [European]
    """
    def __init__(self, open_char: str = '[', close_char: str = ']'):
        self.open_char = open_char
        self.close_char = close_char

        _re_open, _re_close = re.escape(open_char), re.escape(close_char)
        self.pattern_decode = re.compile(f'^{_re_open}(?P<token>.+){_re_close}$')

    def __call__(self, token: str) -> str:
        return f'{self.open_char}{token}{self.close_char}'

    def decode(self, token: str, strict: str = True) -> str:
        # edge case: last period punctuation followed by a comma
        if not strict and token in [f'{self.open_char}.{self.close_char}{c}' for c in [',', '.']]:
            return '.'
        else:
            m = self.pattern_decode.match(token)
            if m is None:
                d_log = dict(token=token, open_char=self.open_char, close_char=self.close_char, pattern=self.pattern_decode.pattern)
                raise ValueError(f'Cannot decode token w/ {pl.pretty(d_log)}')
            return m.group('token')

    def find_tokens_by_enclosing(self, text: str, token_sep: str = ',') -> List[str]:
        """
        For extracting individual tokens from a sentence

        Plain `split` on separator of comma doesn't work, cos comma may show up in a token...

        Edge case e.g. :
            tokens: "The", "Olympic", "Games", "have", "been", "postponed", "due", "to", "the", "coronavirus", "pandemic", ", causing",
            "disappointment", "for", "athletes", "and", "fans", "around", "the", "world", "."
        Note `, causing` is a single token
        """
        # sanity check each relevant token is enclosed properly
        assert text[0] == self.open_char and text[-1] == self.close_char

        if (self.open_char, self.close_char, token_sep) == ('"', '"', ','):
            return json.loads(f'[{text}]')
        else:  # TODO: may be some edge cases not handled
            # start the next word when seeing `open_char`, end the current word when seeing `close_char`
            ret = []
            cur = ''
            i = 0
            found_open = False
            n_txt = len(text)
            while i < n_txt:
                c = text[i]
                if not found_open and c == self.open_char:
                    assert cur == ''
                    found_open = True
                elif found_open and c == self.close_char:
                    ret.append(cur)
                    cur = ''
                    if i != n_txt - 1:  # has not reached the end
                        if text[i+1:i+4] == f'{token_sep} {self.open_char}':  # normal case
                            i += 2
                        elif text[i+1:i+3] == f'{token_sep}{self.open_char}':  # edge case, missing space after comma
                            i += 1
                        else:
                            raise ValueError(f'Decode error on input {pl.i(text)} at index {pl.i(i)}')
                    found_open = False
                else:
                    cur += c
                i += 1
            assert cur == ''
            return ret


def get_default_token_map(sample_format: str) -> Optional[TokenMapTemplate]:
    if 'bio-list' in sample_format:
        if sample_format == 'bio-list':
            return TokenMapEnclose(open_char='"', close_char='"')
        else:
            assert sample_format == 'bio-list-v2'
            return TokenMapIdentity()
    else:
        return None


def get_default_entity_sep(sample_format: str) -> Optional[str]:
    if sample_format in ['natural-pair', 'bio-list']:
        return ';'
    elif sample_format in ['natural-pair-v2', 'bio-list-v2']:
        return ','
    else:
        return None
