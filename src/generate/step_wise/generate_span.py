import re
import json
import glob
from os.path import join as os_join
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict, astuple

from stefutil import *
from src.util import *
from src.data_util import *
from src.generate import *
from src.generate.step_wise.generate_from_sentence import FromSentenceGenerator


__all__ = ['SentenceNSpanSample', 'SpanGenerator']


@dataclass(eq=True, frozen=True)
class SentenceNSpanSample:
    sentence: str = None
    entity_spans: Tuple[str] = None

    @classmethod
    def from_d(cls, d: Union[str, Dict[str, Any]] = None, **kwargs) -> 'SentenceNSpanSample':
        ret = dict()
        if isinstance(d, str):
            d = json.loads(d)
        if d:
            ret.update(d)
        if kwargs:
            ret.update(kwargs)
        assert {'sentence'} <= set(ret.keys()) <= {'sentence', 'entity_names', 'entity_spans'}
        enm, ens = ret.get('entity_names', None), ret.get('entity_spans')
        assert not (enm and ens)  # sanity check at most one is given
        ret['entity_spans'] = tuple(enm or ens or [])
        return cls(**ret)


class SpanGenerator(FromSentenceGenerator):
    """
    2nd step in step-wise data generation: generate entity spans given sentences
    """
    def __init__(self, **kwargs):
        super().__init__(generate_type='entity-name', **kwargs)

        pref_x = sconfig(f'datasets.{self.dataset_name}.x-name')
        pattern_enum_prefix = rf'((?P<idx>\d+)\. )?'

        if not self.cot:
            # e.g. `2. Named Entities: [Munich; CEO; Volkswagen; Dubai; United Arab Emirates]`
            # edge case: named entities list, w/ Sentence prefix & random string enclosed in double quotes, e.g.
            #   `1. Sentence: "Following the earthquake in Nepal, a team of doctors led by [Linda Jackson; Nepal]"`
            #   `2. Sentence: "The [Metropolitan Museum of Art; New York City; Pablo Picasso]"`
            self.pattern_entity_search = [
                re.compile(rf'{pattern_enum_prefix}Named Entities: \[(?P<entities>.*)]\n', re.IGNORECASE),
                re.compile(rf'{pattern_enum_prefix}Sentence: "(.*)\[(?P<entities>.*)]"\n', re.IGNORECASE),
            ]
            # An edge case where the original sentence is generated, e.g.
            #   `2. Sentence: "John Smith, CEO of XYZ Corporation, announced plans to expand operations in Asia."
            #   Named Entities: [John Smith; XYZ Corporation; Asia]`
            self.pattern_sample_search = re.compile(rf'{pattern_enum_prefix}{pref_x}: (?P<sentence>.*?)\nNamed Entities: \[(?P<entities>.*)]\n', re.IGNORECASE)
            self.pattern_entity = re.compile(rf'^{pattern_enum_prefix}Named Entities: \[(?P<entities>.*)]$', re.IGNORECASE)
        else:
            if self.dataset_name != 'mit-movie':
                raise NotImplementedError
            pref_y = 'Likely Keywords'
            # Analysis followed by annotations, e.g.
            #   `2. Likely Keywords: "Aliens" is a movie title. "science fiction" defines a genre. "1986" refers to a time period. Likely keywords are [Aliens; science fiction; 1986].`
            self.pattern_entity_search = [
                re.compile(rf'{pattern_enum_prefix}{pref_y}: (?P<reason>.*). Likely keywords are \[(?P<entities>.*)]\.', re.IGNORECASE)
            ]
            # No CoT reasoning generated, also wrong formatting, e.g.
            #   no enclosing brackets: `2. Likely Keywords: great plot; battle for love.`
            #   above & separated by `,` instead: `3. Likely Keywords: year, movie, The Godfather, directed`
            #   additionally provided the entity type: `2. Likely Keywords: Steven Spielberg [Director]; Julia Roberts [Actor].`
            self.pattern_entity_search_edge_no_cot = [
                re.compile(rf'{pattern_enum_prefix}{pref_y}: (?P<entities>.*)\.', re.IGNORECASE),
                re.compile(rf'{pattern_enum_prefix}{pref_y}: (?P<entities>.*)', re.IGNORECASE)
            ]
            # TODO: more detailed sample format cases: w/ CoT but wrong format
            self.pattern_entity_search += self.pattern_entity_search_edge_no_cot

            # when both the original X and Y are generated
            self.pattern_sample_search = re.compile(rf'{pattern_enum_prefix}{pref_x}: (?P<sentence>.*?)\n{pref_y}: (?P<reason>.*). Likely keywords are \[(?P<entities>.*)]\.', re.IGNORECASE)

        # for identifying enclosing brackets for highlighted entity spans in the sentence
        # these are dropped for a good sentence matching sanity check
        self.pattern_emph = re.compile(rf'\[(?P<et>[^]]{{1,45}})]', re.IGNORECASE)

        # match `entity name [entity type]`, e.g. `Steven Spielberg [Director]`
        self.pattern_entity_pair = re.compile(rf'^(?P<entity_name>[^[]+) \[(?P<entity_type>[^]]+)]$', re.IGNORECASE)

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

            if self.cot:
                raise NotImplementedError

            # adjust wording after so long
            if n_annotate == 1:
                # ret = f'Here is a sentence from news stories. '
                ret = f'Here is a sentence from a news story. '  # more fluent
            else:
                ret = f'Here are {n_annotate} sentences from news stories. '
            ret += ("For each sentence, please identify all named entities occurred using the format [named entity 1; ...].\n"
                    "Especially identify named entities that belong to one of the following entity types:\n"
                    "[person, location, organization].\n")

            # ret += ("Please list such entities on the following line, in the order of occurrence.\n"
            #         "If an entity occurs multiple times in a sentence, list it as many times as it occurs.\n")
            # no need to list in order and multiple times, cos I will be processing them

            # but should ensure a wide coverage, potentially overlapping, to filter out false positives
            # ret += "Please list all possible named entity spans. They can be overlapping.\n"
            # trying to at least get `Volkswagen` as a standalone span
            #   in the sentence `Munich-born CEO of Volkswagen announces plans to expand electric vehicle production in Dubai, United Arab Emirates.`
            # ret += "Please be exhaustive. List all possible named entity spans even if they may overlap.\n"
            # ret += "Please be exhaustive. Include all possible named entity spans even if they may overlap.\n"
            # ret += "Please be exhaustive. Include all likely named entity spans even if they may overlap.\n"
            # ret += "Please be exhaustive. Try to include all potential named entity spans even if they may overlap.\n"
            # ret += "Try to include all possible named entity spans even if they may overlap each other.\n"
            ret += "Try to include all possible named entity spans at the expense of potential overlaps.\n"

            ret += "If no entity is found in the generated sentence, leave the brackets empty."
            return ret
        elif self.dataset_name == 'mit-movie':
            # if n_annotate == 1:
            #     ret = ('Here is a spoken query to a dialogue system about movies. '
            #            'Please identify all named entities in the query using the format [named entity 1; ...].\n')
            # else:
            #     ret = (f'Here are {n_annotate} spoken queries to a dialogue system about movies. '
            #            'Please identify all named entities in the queries using the format [named entity 1; ...].\n')
            # ret += 'Especially identify named entities that belong to one of the following entity types:\n'
            # ets = self.entity_types.copy()
            # ret += f'{pl.nc(ets)}.\n'
            # ret += 'Try to include all possible named entity spans at the expense of potential overlaps.\n'

            # # say `keyword` instead of named entities, seems to resolve the additional trailing `movie` issue
            # if n_annotate == 1:
            #     ret = ('Here is a spoken query to a dialogue system about movies. '
            #            'Please identify all keywords in the query using the format [keyword 1; ...].\n')
            # else:
            #     ret = (f'Here are {n_annotate} spoken queries to a dialogue system about movies. '
            #            'Please identify all keywords in the queries using the format [keyword 1; ...].\n')

            # adding CoT seemed to reinforce generating just informative spans, so more spans can be included
            act = 'identify'
            if self.cot:
                q = 'the query' if n_annotate == 1 else 'each query'
                act = f'analyze {q} and identify and extract'
            if n_annotate == 1:
                ret = f'Here is a spoken query to a dialogue system about movies.'
            else:
                ret = f'Here are {n_annotate} spoken queries to a dialogue system about movies.'
            ret = f'{ret} Please {act} informative keywords in the queries using the format [keyword 1; ...].\n'

            # y = 'keywords'
            y = 'informative keyword spans'
            ret += f'Especially identify {y} that belong to one of the following entity types:\n'
            ets = self.entity_types.copy()
            ret += f'{pl.nc(ets)}.\n'
            ret += 'Try to include all possible keyword spans at the expense of potential overlaps.\n'

            # Genre terms have incorrect spans, e.g. `robot movie` as Genre span, trying to correct them; didn't seem to work
            # ret += 'Please use the rules below to identify the spans:\n'
            # ret += 'Genre spans should not end in words such as "movie" or "film".\n'
            ret += 'Genre keywords should not end in "movie" or "film".\n'

            # ret += 'Please follow the rules below to identify the spans:\n'
            # ret += 'Genre spans should not have trailing "movie" or "film".\n'

            # ret += 'Spans should not have trailing "movie" or "film" if possible.\n'

            if self.insert:
                ret += "Please use the definitions below to identify the named entities.\n"
                defn = self.d2s()
                ret += f'{defn}\n\n---\n\n'

            # omit since almost all generated sentences have entities
            # ret += "If no entity is found in the generated sentence, leave the brackets empty."
            return ret
        else:
            raise NotImplementedError

    def process_completions(
            self, sentences: Union[str, List[str]] = None, shuffle_sentences: Union[bool, int] = False,
            with_unlabeled: Union[bool, int] = None, n_demo: Optional[int] = 5, demo_args: Dict[str, Any] = None,
            completions_dir_name: completions.CompletionDirectoryDict = None,
            output_dir_name: str = None, expected_samples_per_completion: int = None,
            aggregate: bool = False, drop_quote: bool = None, lowercase: bool = None, sub_dir: str = None
    ) -> List[SentenceNSpanSample]:
        """
        Ensure the entities generated by the model are valid, matching sentences, in that order

        Unlike `LabelGenerator`, `aggregate` for merging overlapping spans
        """
        init_out = self.process_completions_init(
            sentences=sentences, shuffle_sentences=shuffle_sentences, with_unlabeled=with_unlabeled, n_demo=n_demo, demo_args=demo_args,
            completions_dir_name=completions_dir_name, output_dir_name=output_dir_name,
            expected_samples_per_completion=expected_samples_per_completion, aggregate=aggregate, sub_dir=sub_dir
        )
        base_path, output_path, init_log, sentences, n_expect, d_log_count = astuple(init_out)

        def process_single(it, sentences_: List[str] = None) -> Tuple[List[SentenceNSpanSample], Dict[str, List[str]]]:
            # the only failing edge case is span not found in sentence, most commonly due to confused w/ spans in neighboring sentences
            #   => save these failed sentences and re-generate
            ret_ = []
            challenging_sents_: Dict[str, List[str]] = defaultdict(list)

            n_cpl = len(it.filepaths)
            global_i_sample = 0
            for i_cpl, c in enumerate(it.iter):
                is_last_group = i_cpl == n_cpl - 1
                completion, fnm, p_fnm = c.content, c.filename, c.pretty_filename

                # assume sentences are always not re-generated, each line should be entity span list
                # split_args = dict(completion=completion, ec=self.ec, filename=p_fnm)
                # sent_miss = False
                # try:
                #     out = split_samples(pattern=self.pattern_sample_search, **split_args, has_enum_prefix=True)
                # except ValueError as e:  # for edge case
                #     sent_miss = True
                #
                #     # last group may not have prefix if just 1 sample
                #     has_enum = api_util.completion_has_enum_prefix(completion=completion) if is_last_group else True
                #     out = split_samples(pattern=self.pattern_entity_search, **split_args, has_enum_prefix=has_enum, return_match_map=True)
                #     msg = f'Sentences missing, only entity names found in completion w/ {pl.i(completion)}'
                #     self.ec(msg=msg, kind='sentence-not-re-generated')
                out = self._split_from_sentence_samples(
                    completion=completion, is_last_group=is_last_group, filename=p_fnm, return_match_map=True,
                    pattern_w_sentence=self.pattern_sample_search, pattern_wo_sentence=self.pattern_entity_search)
                out, sent_miss = out.split_output, not out.sentence_found
                assert out.success
                str_samples, ms = out.samples, out.matches
                n_samples_got = len(str_samples)

                # unless it's the last completion, number of samples should match
                if n_expect:
                    if is_last_group:
                        assert n_samples_got <= n_expect
                    else:
                        assert n_samples_got == n_expect

                pat_map = out.pattern_map
                assert len(pat_map) == 1  # sanity check since patterns are not union-ed, one pattern matches all spans
                pat_matched = next(iter(pat_map.keys()))
                cot_found = pat_matched not in self.pattern_entity_search_edge_no_cot
                if not cot_found:
                    d_log = dict(completion=completion, filename=p_fnm)
                    kd = 'no-cot'
                    self.ec(msg=f'Edge case: no CoT reasoning generated w/ {pl.i(d_log)}', kind=kd, args=dict(filename=fnm))

                for i_sample, (sample, m) in enumerate(zip(str_samples, ms), start=global_i_sample):
                    sent = sentences_[i_sample]  # get the original sentence

                    if not cot_found:
                        challenging_sents_['no-cot'].append(sent)

                    if sent_miss:
                        sent_gen, entities = sent, m.group('entities')
                    else:  # extract the sentence and entities from the sample
                        sent_gen, entities = m.group('sentence', 'entities')

                        # drop potential brackets that highlights entity spans
                        if len(find_match(text=sent_gen, pattern=self.pattern_emph)) > 0:
                            ori_sent_gen = sent_gen

                            emphs = [m.group('et') for m in find_match(text=sent_gen, pattern=self.pattern_emph)]
                            sent_gen = self.pattern_emph.sub(r'\g<et>', sent_gen)
                            d_log = dict(filename=p_fnm, original=ori_sent_gen, modified=sent, emphasized=emphs)
                            msg = f'Entity-enclosing brackets dropped from sentence w/ {pl.i(d_log)}'
                            self.ec(msg=msg, kind='drop-span-emph', args=dict(original_sentence=ori_sent_gen))
                        drop = self.check_sentence_diff(sentence_in_prompt=sent, sentence_in_response=sent_gen)
                        if drop:
                            challenging_sents_['sentence-mismatch'].append(sent)
                            continue
                    # from now on, always use the original sentence in prompt to check for entities

                    enms = [e.strip() for e in entities.split(self.entity_sep)]
                    d_log = dict(sentence=sent, entities=entities, entity_spans=enms)
                    if self.entity_sep == ';' and ',' in entities and self.entity_sep not in entities:
                        # assume edge case: entities separated by `,` instead of `;`
                        enms = [e.strip() for e in entities.split(',')]
                        kd = 'incorrect-sample-format'
                        self.ec(msg=f'Edge case: entity names separated by `,` instead of `;` w/ {pl.fmt(d_log)}', kind=kd)
                        challenging_sents_[kd].append(sent)

                    # relevant edge cases copied over from `Np2Transform`
                    #   Only failure edge case is entity span not found in sentence, other edge cases for logging only
                    if any(is_none(e) for e in enms):
                        self.ec(msg=d_log, kind='none-entity')
                        d_log['entity-spans'] = enms = [e for e in enms if not is_none(e)]

                    # check each entity is found in the original sentence
                    eis = entities_in_sentence(sentence=sent, entity_names=enms, ignore_case=True)
                    # entity type may be generated, though not asked for; if so, drop them
                    ms_pair = [self.pattern_entity_pair.match(enm) for enm in enms]
                    if not eis.all_found and all(m is not None for m in ms_pair):  # assume entity type is also generated
                        enms = [m.group('entity_name') for m in ms_pair]
                        kd = 'incorrect-sample-format'
                        self.ec(msg=f'Edge case: entity type generated w/ {pl.fmt(d_log)}', kind=kd)
                        challenging_sents_[kd].append(sent)

                        eis = entities_in_sentence(sentence=sent, entity_names=enms, ignore_case=True)

                    if not eis.all_found:
                        if any(has_punc_on_edge(enm).has_quote for enm in enms):
                            # TODO: edge case: entity name annotation is enclosed in quotes, but the sentence doesn't have the quotes
                            if entities_in_sentence(sentence=sent, entity_names=[drop_enclosing_quotes(enm) for enm in enms], ignore_case=True).all_found:
                                raise NotImplementedError

                        # sanity check the missing entity is not due to a different version of the sentence
                        assert not entities_in_sentence(sentence=sent_gen, entity_names=enms, ignore_case=True).all_found

                        d_log['missing_entity_names'] = eis.entities_not_found
                        kd = 'entity-not-found'
                        self.ec(msg=f'Entity names not found as exact match in sentence w/ {pl.fmt(d_log)}', kind=kd, failed=True)
                        challenging_sents_[kd].append(sent)
                        continue

                    ovl_out = entities_overlapping(
                        sentence=sent, entity_names=enms, ignore_case=True, search_in_order=False)
                    overlap, ms = ovl_out.overlap, ovl_out.matches
                    # if overlap and len(ms) == len(enms):
                    if overlap:
                        # we allow this for span generation, cos will reply on span classification to filter out false positives
                        _msg = f'Entity names overlapping in sentence w/ {pl.fmt(d_log)}'
                        self.ec(msg=_msg, kind='entity-overlap')

                    ic = True
                    if entities_differ_in_case_only(entity_names=enms, sentence=sent, ec=self.ec):
                        ic = False
                        msg = f'Edge case: entity names differ in case only w/ {pl.fmt(d_log)}'
                        self.ec(msg=msg, kind='entity-case-diff')
                    c = get_non_overlapping_keyword_counts(sentence=sent, keywords=enms, ignore_case=ic)
                    if any(c[enm] == 0 for enm in enms):  # some entity names not found in sentence
                        miss_ens = [enm for enm, count in c.items() if count == 0]
                        d_log['missing_entity_names'] = miss_ens
                        msg = f'Entity names not found in sentence for overlapping w/ {pl.fmt(d_log)}'
                        # we allow this for span generation since the overlapping spans will be compared & filtered out later
                        self.ec(msg=msg, kind='entity-not-found-overlap')

                    enms_is_unique = len(enms) == len(set(enms))
                    if enms_is_unique:
                        # entity name list are all distinct, but some appear multiple times
                        if any(c[enm] > 1 for enm in enms):
                            # in this case, LLM failed to generate the same entity multiple times
                            d_log['multi-occurring-entity-spans'] = [enm for enm in enms if c[enm] > 1]
                            msg = f'Edge case: multi-occurring entity names not listed multiple times w/ {pl.fmt(d_log)}'
                            self.ec(msg=msg, kind='multi-occur-entity-no-list')
                    else:
                        # in this case, LLM able to generate multiple occurrences of the same entity when it occurs multiple times
                        # remove duplicates since we will be checking every occurrence in the entity CLS step
                        enms = list(dict.fromkeys(enms))
                        msg = f'Edge case: entity names not distinct w/ {pl.fmt(d_log)}'
                        self.ec(msg=msg, kind='generated-multi-occur-entity')

                    out = drop_entities_enclosing_puncs(
                        entity_names=enms, dataset_name=self.dataset_name, drop=drop_quote, ec=self.ec, d_log=d_log)
                    enms = out.entity_names,
                    if len(out.entity_names_modified) > 0:
                        d_log['entity_spans_quote_drop'] = out.entity_names_modified

                    # not needed, but order the entity spans anyway for prettier
                    # since some entities may appear multiple time, we order them by the first occurrence
                    def en2sort_key(en: str):
                        """
                        :return: A value for sorting the entity spans by order
                        """
                        if ic:  # ignore case
                            idx = sent.lower().index(en.lower())
                        else:
                            idx = sent.index(en)
                        return idx, len(en)  # sort by index first, then length, i.e. shorter first
                    en_ = sorted(enms, key=en2sort_key)
                    if en_ != enms:  # check if the entity spans are already in order
                        msg = f'Edge case: entity spans not in order w/ {pl.fmt(d_log)}'
                        self.ec(msg=msg, kind='entity-spans-not-in-order')
                        d_log['entity-spans'] = enms = en_

                    ret_.append(SentenceNSpanSample.from_d(sentence=sent, entity_spans=enms))
                global_i_sample += n_samples_got if is_last_group else n_expect  # since prompts are grouped by `n_expect`
            assert global_i_sample == len(sentences_)  # sanity check
            self.logger.info(f'Processed {pl.i(len(ret_))} sentence-spans pairs from {pl.i(len(sentences_))} sentences')
            return ret_, challenging_sents_

        t = Timer()
        proc_args = dict(completion_type=self.processed_type, logger=self.logger)
        init_args = {**proc_args.copy(), **dict(completion_base_path=base_path, output_path=output_path, init_log=init_log)}
        if not aggregate:
            it_ = process_completions_init(completions_dir_name=completions_dir_name, **init_args)
            d_log_count['#completions'] = len(it_.filepaths)
            log_prompt_eg(dir_name=completions_dir_name, base_path=base_path, logger=self.logger)
            ret, challenging_sents = process_single(it=it_, sentences_=sentences)

            d_log_count['#sample-extracted'] = len(ret)
        else:
            # k different completions under `completions_dir_name`, process each independently
            #   e.g. `completions_dir_name` is `run-1`, `run-2`, `run-3`, ...
            base_path = os_join(base_path, completions_dir_name)
            dir_nms = sorted(glob.iglob(os_join(base_path, 'run-*')))
            dir_nms = [stem(dir_nm) for dir_nm in dir_nms]

            process_completions_init(**init_args)
            nm2ret, challenging_sents = dict(), dict()
            d_n_cpl = dict()
            for i, dir_nm in enumerate(dir_nms, start=1):
                log_prompt_eg(dir_name=dir_nm, base_path=base_path, logger=self.logger)
                self.logger.info(f'Processing {pl.i(i)}-th run w/ directory {pl.i(dir_nm)}')
                it_ = iter_completions(dir_name=dir_nm, base_path=base_path, **proc_args)
                d_n_cpl[dir_nm] = len(it_.filepaths)

                sents = sentences[dir_nm]
                nm2ret[dir_nm], challenging_sents[dir_nm] = samples, _ = process_single(it=it_, sentences_=sents)
                self.logger.info(f'Processed {pl.i(len(samples))} sentence-spans pairs from {pl.i(len(sents))} sentences')
            d_log_count['#completions'] = d_n_cpl
            n_extract = {'#run': len(dir_nms)}
            n_extract.update({nm: len(samples) for nm, samples in nm2ret.items()})
            d_log_count['#sample-extracted'] = n_extract

            # merge the spans for each individual run
            # keep all the samples as long as there's at least 1 run that has it
            sent2lst_spans = defaultdict(list)
            for samp in chain_its(nm2ret.values()):
                samp: SentenceNSpanSample
                sent2lst_spans[samp.sentence].append(samp.entity_spans)
            sic(sent2lst_spans)
            # TODO: handle challenging sentences merge
            raise NotImplementedError

        # summarize extracted entity spans by count; to see if any common incorrect span boundaries can be improved
        all_enms = sum((list(s.entity_spans) for s in ret), start=[])
        c_enm = Counter(all_enms)
        c_enm = dict(c_enm.most_common())
        d_log_enm_count = {'#entity-spans': len(all_enms), '#unique-entity-spans': len(set(all_enms)), 'counts': c_enm}
        self.logger.info(f'Extracted entity spans by count: {pl.fmt(d_log_enm_count)}')

        # there will not be duplicate samples since all sentences are unique
        out_fnm = os_join(output_path, 'sentences-&-spans')
        # a compact version for loading, and a pretty version for human inspection
        with open(f'{out_fnm}.jsonl', 'w') as f:
            for s in ret:
                f.write(f'{json.dumps(asdict(s))}\n')
        with open(f'{out_fnm}.json', 'w') as f:
            json.dump([asdict(s) for s in ret], f, indent=4)

        # also write challenging samples
        if len(challenging_sents) > 0:
            out_fnm = os_join(output_path, 'challenging-sentences.json')
            # write the challenging sentences to prompt to LLM again, in a different shuffle order
            log_n_save_challenging_samples(challenging_samples=challenging_sents, sample_kind='sentence', output_path=out_fnm, logger=self.logger)
        self.logger.info(self.ec.summary())
        self.logger.info(f'Processed samples w/ {pl.i(d_log_count, indent=1)} in {pl.i(t.end())}')
        return ret
