"""
Given a (sentence, entity span, entity label) triple, generate a correction,
    i.e. 4-way classification label, to potentially correct wrong annotations.
The labels are:
    1> Correct span & Correct label
    2> Correct span & Wrong label, i.e. named entity of a different type
    3> Wrong span & Correct label, i.e. refine span boundaries
    4> Wrong span & Wrong label, i.e. not a named entity

To save prompt cost:
    1> select challenging triples, i.e. triples w/ low log prob
    2> batch all triples for the same entity type
        Thus prompt demo will be for each entity type, instead of e.g. all 12 entity types in MIT-Movie
"""


import os
import re
import random
from os.path import join as os_join
from copy import deepcopy
from typing import Dict, Tuple, List, Union, Any
from collections import defaultdict, Counter
from dataclasses import asdict

from tqdm import tqdm

from stefutil import get_logger, pl, ca, add_file_handler, drop_file_handler, group_n, get_random_generator, Timer
from src.util import sconfig, dataset_name2data_dir, patterns, sample_check as check
from src.util.ner_example import NerReadableExample
from src.data_util import dataset, completions, edit
from src.data_util.prettier import sdpc, atc
from src.generate.step_wise.util import AnnotationGenerator
from src.generate.step_wise.util_entity_correct import (
    ENTITY_TYPE_OTHER, CORRECTION_DIR_NAME, entity_type2entity_correction_options, UncertainTriplesOutput,
    load_triples_w_logprob, log_n_select_uncertain_triples,
    CorrectionLabel, LABEL_CORRECT, LABEL_WRONG_BOUNDARY, LABEL_WRONG_TYPE, LABEL_NOT_NAMED_ENTITY, STR_NOT_NAMED_ENTITY,
    CorrectionSample, log_correction_samples, log_n_save_corrections, override_w_manual_corrections,
    ner_sample2corrected_sample
)
from src.generate.step_wise.entity_type2info_dict import Type2EntityCorrectionInfo


__all__ = ['CorrectionGenerator']


_logger = get_logger(__name__)


class CorrectionGenerator(AnnotationGenerator):
    def __init__(
            self, highlight_span: Union[bool, str] = None, challenging_source: str = 'llm', batched: bool = True, zero_shot: bool = False,
            **kwargs
    ):
        super().__init__(generate_type='entity-correction', **kwargs)
        self.dir_args['sub_dir'] = CORRECTION_DIR_NAME

        if highlight_span:
            if highlight_span is True:
                self.highlight_span = 'braces'
            else:
                assert isinstance(highlight_span, str)
                self.highlight_span = highlight_span
            ca.assert_options(display_name='Highlight Span Type', val=self.highlight_span, options=['braces', 'brackets'])
        else:
            self.highlight_span = False
        ca.assert_options(display_name='Source for Challenging Sample Scoring', val=challenging_source, options=['llm', 'bert'])
        self.challenging_source = challenging_source
        self.batched = batched
        self.zero_shot = zero_shot

        self.et2info = None

        pref_x = sconfig(f'datasets.{self.dataset_name}.x-name')
        pattern_enum_prefix = rf'((?P<idx>\d+)\. )?'
        self._pat_space = rf'[ \t]*'  # pattern for horizontal spaces, no newline char
        # match a single (sentence, entity span, entity type) triple sample, e.g.
        #   `2. Query: "can you recommend a movie similar to wonderwall"
        #   Text Span: "similar to wonderwall"
        #   Label: (D).`
        self.pattern_sample_search = [
            re.compile(
                rf'{pattern_enum_prefix}{pref_x}: ["|\'](?P<sentence>.+)["|\']\n'
                rf'Text Span: (?P<span>.+)\n'
                rf'Label: (?P<label>.+)\n',
                re.IGNORECASE
            )
        ]
        # more frequently, the LLM just omit the sentence part
        # Edge case: correction is on newline, e.g.
        #   `2. Text Span: "underwater"
        #    Label: (C). Is a named entity but category is not movie genre
        #    Correct entity type: other`
        # Edge case, no newline in the middle, e.g.
        #   `1. Text Span: "lunch"
        #   Label: (B). The correct span boundary is "for lunch"
        #   2. Text Span: "breakfast"
        #   Label: (A). Correct Entity Annotation.
        #   3. Text Span: "lunch"
        #   Label: (B). The correct span boundary is "for lunch"`
        self.pattern_sample_search_edge = re.compile(
            rf'{pattern_enum_prefix}Text Span: (?P<span>.+)\n'
            # to prevent matching the next sample if no additional whitespace in between
            rf'{self._pat_space}Label: (?P<label>.+\n((?!(\d+\. )?Text Span: ).+\n)?)',
            re.IGNORECASE
        )
        # when the last group is just 1 sample, LLM may simply generate just the label part, e.g.
        #   `Label: (C). Entity type: Genre`
        # Edge case: text span omitted, e.g.
        #   `1. (A). The span is a named entity of type organization
        #   2. (C). The span is a named entity but the category is not organization. The correct entity type is other.
        #   3. (A). The span is a named entity of type organization`
        self.pattern_label_search = [
            re.compile(rf'{self._pat_space}Label: (?P<label>.+)\n', re.IGNORECASE),
            re.compile(rf'{self._pat_space}{pattern_enum_prefix}(?P<label>.+)\n', re.IGNORECASE),
        ]

        # match different label spans for different corrections, e.g.
        #   `Label: (A).`
        #   `Label: (C). The correct entity type is Genre.`
        #   `Label: (B). The correct span boundary is "Fight for Your Right".`

        self._lb_pref = rf'(Label: )?'
        self._sp_pref = rf'(The span )?'
        # filter out additional punctuations to prevent false-positive matches for non-correction labels, e.g.
        #   `(C) - Actor`, `(C): Actor`
        # self._pat_choice = rf'(?P<choice>[^.,-:]+[.,]?)'
        self._pat_choice = rf'(?P<choice>\([^.,-:]\)[.,]?)'
        # self._pat_choice_nc = rf'(?P<choice>[^.,-:]+[.,]?)'  # for no correction
        self._pat_choice_np = rf'(?P<choice>\([^.,-:]\))'  # an edge case: no trailing punctuation
        self._pat_til_choice = rf'{self._pat_space}{self._lb_pref}{self._pat_choice}'
        self._pat_til_choice_np = rf'{self._pat_space}{self._lb_pref}{self._pat_choice_np}'

        # An edge case where the choice desc is also provided, e.g.
        #   `Label: (B). Contains a named movie title entity but the span boundary is not precise. The correct span boundary is "paranormal thriller film".`
        # An edge case where LLM didn't follow the template, e.g.
        #   `Label: (C). Entity type: Review`
        #   `Label: (C). Category is not viewer's rating. (Genre)`
        #   `2. Text Span: "underwater"
        #    Label: (C). Is a named entity but category is not movie genre
        #    Correct entity type: other`
        #   `1. Text Span: "main character"
        #   Label: (B). Contains a named movie character entity but the span boundary is not precise. The correct span boundary is "The main character" in the movie The Lion King.`
        #   `1. Text Span: "Silent era"
        #   Label: (C). The span is not a named entity but a time period.`
        #   `1. Text Span: "Mahershala Ali"
        #    Label: (C) - Actor`
        #   `1. Text Span: "suitable for children and adults"
        #   Label: (C). The span is not a named entity but category is not mpaa rating; Entity Type: Viewers' Rating`
        #   `1. Text Span: "family-friendly"
        #    Label: (C). The span is a named entity but the category is not mpaa rating (Genre)`
        #   `1. Text Span: "suitable for children and adults"
        #   Label: (C). The span is not a named entity, the category is Viewers' Rating.`
        #   `1. Text Span: "lead actor"
        #   Label: (C) Actor`
        #   `1. Text Span: "action-packed"
        #   Label: (C). The span is not a named entity but the category is movie genre.`
        #   `2. Text Span: "trailer release"
        #    Label: (C). The span is a named entity of type Release Date.`
        #   `3. Text Span: 'song "My Heart Will Go On"'
        #   Label: (B). (Correct span boundary: "My Heart Will Go On")`
        #   `3. Text Span: "Making-of featurette"
        #   Label: (C), Type: Other`
        #   `2. Text Span: "British"
        #   (C) [other]`
        #   `2. Text Span: "president"
        #   Label: (C) [entity type: person]`
        #   `Text Span: "South Korean"
        #   Label: (B), "South Korean" is the named location entity`
        #       <= Weird since not sure what label from description, but since label is B, consider this as a span correction...
        #   `(B). "African nations" is a named location entity but the span boundary is not precise.
        #   The correct span boundary should be "African nations".`
        #   `1. Text Span: "Indian"
        #   Label: (C). Type: location`
        #   `3. Text Span: "GRAMMY AWARD"
        #   Label: (B) - "GRAMMY AWARD" should be "GRAMMY AWARD WINNER"`
        #   `1. Text Span: "wildlife conservation group"
        #   Label: (B). Named organization entity but the span boundary is not precise. Correct span: "wildlife conservation group"`
        #   `(C). Not a Named Entity, other`
        #   `(C). The span is a named entity but the category is not location (other).`
        #   `Label: (C). The span is a named entity but the category is not location, it is an other entity type.`
        #   `(C). The span is a named entity but the category is not location, it is Amenity.`
        #   `(C). The span is a named entity but the category is not location, the correct entity type is Restaurant Name`
        #   `(C). Named Entity, Category: Award`
        #   `(C). Award/Recognition`
        #   `(C). Named Entity of type Director`
        #   `(C). The span is a named entity but the category is not character, Genre`
        #   `(B). The span contains a named character entity but the span boundary is not precise. Correct span: "James Bond"`
        #   `(C). Named Entity, Entity Type: Other`
        #   `(C). Named Entity, other1
        #   `(C). TV Show`
        #   `\tLabel: (C). The span is a named entity but the category is not viewers' rating\n\tEntity Type: other`
        #   `(C). Not a named entity, Category: Genre`
        #   `(C). The span is not a named entity. The category is not viewers' rating. Entity Type: other`
        #   `(C). The span is not a named entity, it is an evaluative descriptor.`
        #   `(C). Named entity but the category is not actor, correct entity type: Song`
        #   `Label: (C). The span is a named entity but the category is not actor, it is a named entity of type Musician.`
        #   `(C). Named Entity, but the category is not trailer`
        #   `Label: (C). The span is a named entity but the category is not song, the correct entity type is Actor.`
        #   `(C). The span is not a named entity, Category: MPAA Rating`
        #   `(B). The span contains a restaurant name entity but the span boundary is not precise. The correct span should be "food trucks"`
        #   `Label: (C). The span is a named entity but the category is not location. (Entity Type: other)`
        #   `(C). The span is not a named entity, the correct entity type is Hours.`
        #   `Label: (C). The span is a named entity but the category is not location, it is an other entity type.`
        choice_desc = rf'(?P<choice_desc>{self._sp_pref}[^.]+\.)'
        choice_desc_edge = rf'(?P<choice_desc>{self._sp_pref}[^.]+)'
        # expected LLM-generated signal words for correcting entity span
        self.correction_types_span = ['span boundary', 'correct span boundary', 'entity boundary']
        # for correcting entity type
        self.correction_types_type = ['correct entity type', 'entity type', 'named entity type', 'category', 'named entity', 'type']
        self.correction_types = self.correction_types_span + self.correction_types_type
        pat_crt_tps = patterns.options2re_options(options=self.correction_types)

        pat_ets = patterns.options2re_options(options=self.entity_types)
        self.pattern_label_has_correction = [
            re.compile(rf'^{self._pat_til_choice}( {choice_desc})? \(Correct (?P<correction_type>{pat_crt_tps}): "(?P<correction>.*?)"\)(\.)?$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice}( {choice_desc})? The correct (?P<correction_type>{pat_crt_tps}) is "(?P<correction>.*?)" (?P<later_sentence>.+)(\.)?$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice}( {choice_desc})? The correct (?P<correction_type>{pat_crt_tps}) (is|should be)(:)? (?P<correction>.*?)(\.)?$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice}( {choice_desc_edge}\.?)?( )?\n(\s*)(?P<correction_type>{pat_crt_tps})(:| is) (?P<correction>.+?)$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice}( {choice_desc})? (?P<correction_type>{pat_crt_tps})(:| is) (?P<correction>.+?)\.?$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} (?P<correction_type>Category) is not (?P<entity_type>.+) \((?P<correction>.+)\)$', re.IGNORECASE),

            re.compile(rf'^{self._pat_til_choice} {self._sp_pref}(Contains a )?(named )?(?P<entity_type>[^.]+) entity but the (?P<correction_type>(span )?boundary) is not precise\. (The )?Correct span(:| should be) (?P<correction>.+)$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} {self._sp_pref}Is not a (?P<correction_type>named entity) but (a|an|category is) (not (?P<entity_type>.+); Entity Type: )?(?P<correction>.+?)\.?$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} {self._sp_pref}Is not a named entity(,| but) the (correct )?(?P<correction_type>(category|entity type)) is (?P<correction>.+?)\.?$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} {self._sp_pref}Is not a named entity, it is an (?P<correction>.+?)\.?$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} {self._sp_pref}is Not a (?P<correction_type>named entity). The category is not (?P<entity_type>.+)\. Entity type: (?P<correction>.+)$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} {self._sp_pref}(is )?Not a (?P<correction_type>named entity), Category: (?P<correction>.+)$', re.IGNORECASE),

            re.compile(rf'^{self._pat_til_choice} ({self._sp_pref}Is a )?(?P<correction_type>named entity) but the category is not (?P<entity_type>.+), the correct entity type is (?P<correction>.+?)\.?$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} ({self._sp_pref}Is a )?(?P<correction_type>named entity) but the category is not (?P<entity_type>.+)\. \(?(Entity Type: )?(?P<correction>.+?)\)?$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} {self._sp_pref}Is a (?P<correction_type>named entity) but the category is not (?P<entity_type>.+)[.,] ((correct )?Entity Type:|it is( (an|a))? (named entity of type )?)?(?P<correction>.+?)( entity type)?\.?$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} {self._sp_pref}Is a (?P<correction_type>named entity) but the category is not (?P<entity_type>.+)\n{self._pat_space}?Entity Type: (?P<correction>.+?)\.?$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} ({self._sp_pref}Is a )?(?P<correction_type>named entity) but the category is not (?P<entity_type>.+) [\[(]((entity type|Category): )?(?P<correction>.+)[])]\.?$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} ({self._sp_pref}Is a )?(?P<correction_type>named entity) but the category is not (?P<entity_type>.+), it is (an|a) (?P<correction>.+)( entity type)?\.?$', re.IGNORECASE),

            re.compile(rf'^{self._pat_til_choice} {self._sp_pref}Is a (?P<correction_type>named entity) of type (?P<correction>.+)\.$', re.IGNORECASE),

            re.compile(rf'^{self._pat_til_choice} (?P<correction_type>named entity), Category: (?P<correction>.+)$', re.IGNORECASE),

            re.compile(rf'^{self._pat_til_choice} Correct type\. The (?P<correction_type>entity type) is (?P<correction>{pat_ets})\.$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} (?P<correction_type>named entity) of type (?P<correction>{pat_ets})$', re.IGNORECASE),

            # re.compile(rf'^{self._pat_til_choice} named (?P<entity_type>.+) entity but the (?P<correction_type>span boundary) is not precise. Correct span: (?P<correction>.+)$', re.IGNORECASE),

            re.compile(rf'^{self._pat_til_choice} - (?P<span>.+) should be (?P<correction>.+?)$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} - ((?P<correction_type>Entity type): )?(?P<correction>.+?)$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} \[(?P<correction_type>Entity type): (?P<correction>.+?)]$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} (?P<correction_type>type): (?P<correction>.*)$', re.IGNORECASE),

            re.compile(rf'^{self._pat_til_choice} ((?P<correction_type>type): )?(?P<correction>Other)$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} Not a (?P<correction_type>named entity), (?P<correction>Other)$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} (?P<correction_type>named entity), (Entity type: )?(?P<correction>Other)$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} (?P<correction_type>named entity) of type (?P<correction>Other)$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} \[(?P<correction>Other)]$', re.IGNORECASE),

            re.compile(rf'^{self._pat_til_choice_np} (?P<correction>{pat_ets})$', re.IGNORECASE),

            re.compile(rf'^{self._pat_til_choice} "(?P<correction>.+)" is the named location entity$', re.IGNORECASE),

            re.compile(rf'^{self._pat_til_choice} (?P<correction>(Award/Recognition|TV Show))', re.IGNORECASE),
        ]
        # An edge case where LLM didn't generate the template at all, e.g.
        #   `1. Text Span: "family-friendly"
        #   Label: (C). Viewers' Rating`
        wrong_format_correct_entity_types = self.entity_types.copy() + [ENTITY_TYPE_OTHER]
        pat_crt_ets = patterns.options2re_options(options=wrong_format_correct_entity_types)
        self.pattern_label_has_correction += [
            re.compile(rf'^{self._pat_til_choice} (?P<correction>{pat_crt_ets})$', re.IGNORECASE)
        ]
        # An edge case where LLM didn't generate the correction at all, e.g.
        #   `3. Query: "Could you recommend a {{breathtaking}} movie directed by Steven Spielberg?"
        #   Text Span: "breathtaking"
        #   Label: (B). Contains a named movie review entity but the span boundary is not precise.`
        #   `(C). Correct Type. The span is a named entity but the category is not price.`
        self.pattern_label_has_correction_missing = [
            re.compile(rf'^{self._pat_til_choice} {self._sp_pref}Contains a named (?P<entity_type>[^.]+) entity but the (?P<correction_type>(span )?boundary) is not precise\.?$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} (Correct type\. )?{self._sp_pref}(Is a )?named entity(,?) but the (?P<correction_type>category) is not (?P<entity_type>[^().,[\]\n]+?)\.?$', re.IGNORECASE),
        ]
        # later patters are more general, so this specific one needs to be inserted at the front
        self.pattern_label_has_correction = self.pattern_label_has_correction_missing + self.pattern_label_has_correction

        # for dropping double braces added to the sentence if it's regenerated in the LLM response
        if self.highlight_span:
            if self.highlight_span == 'braces':
                # some span is huge, e.g.
                #   `I'm looking for a critically acclaimed film with the {{highest number of nominations for Best Picture}}.`
                #   `The {{United Nations International Children's Emergency Fund}} (UNICEF) is a United Nations agency responsible for ...`
                #   `... the official publication of the {{National Association for the Advancement of Colored People (NAACP)}}.`
                self.pattern_emph = [re.compile(rf'{{{{(?P<et>[^}}]{{1,70}})}}}}', re.IGNORECASE)]
            else:
                assert self.highlight_span == 'brackets'
                self.pattern_emph = [re.compile(rf'\[(?P<et>[^]]{{1,70}})]', re.IGNORECASE)]
        else:
            self.pattern_emph = None

    def get_pattern_label_no_correction(self, entity_type: Union[str, List[str]] = None) -> patterns.Patterns:
        # An edge case where the choice desc is also provided, e.g.
        #   `'(A). Is a named entity of type movie title'`
        # An edge case where LLM didn't follow the template, e.g.
        #   `(C). The span is not a named entity, it is a general reference to a year.'`
        #   `(A). This is a named entity of type movie review.`
        #   `(C). The span is not a named entity, it is a vague description.`
        #   `(A) - Song`
        #   `(A). Named entity of type person`
        #   `(A). Correct Entity Annotation.`
        #   `Label: (A). The correct entity annotation.`
        #   `(A). Organization`
        #   `(C). The span is a named entity but the category is not organization`
        #   `Label: (C). Not a Named Entity (organization)`
        #   `(A). Correct Entity Annotation. Hours`
        #   `(C). Correct Type. The entity type is Amenity.`
        #   `(A). Correct Entity Annotation. (Viewers' Rating)`
        #   `(A). Correct Entity Annotation. Named Entity of type Amenity.`
        #   `(A). The span is a named entity of type amenity (Upscale)`
        #   `(A). The correct entity type is Amenity.`
        if isinstance(entity_type, str):
            entity_type = [entity_type]
        assert isinstance(entity_type, list)
        ets = []
        for et in entity_type:  # name of the entity type in prompt
            ets.append(self.et2info(entity_type=et)['name'])
        entity_type += ets
        pat_et = patterns.options2re_options(options=entity_type)
        return [
            re.compile(rf'^{self._pat_til_choice}( )?(?P<choice_desc>({self._sp_pref}|This )?Is a named entity of type (?P<entity_type>{pat_et}))?\.?$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice}( )?(?P<choice_desc>({self._sp_pref}(is )?Not a named entity( of type {pat_et})?)(, it is a general reference to a (?P<entity_type>{pat_et}))?)?\.?$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} Not a named entity \((?P<entity_type>{pat_et})\)$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice}( )?(?P<choice_desc>{self._sp_pref}is not a named entity, it is a vague description\.)$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice}( )?(?P<choice_desc>named entity of type (?P<entity_type>{pat_et}))?\.?$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} (- )?(?P<entity_type>{pat_et})$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} the Correct Entity type is (?P<entity_type>{pat_et})\.$', re.IGNORECASE),

            re.compile(rf'^{self._pat_til_choice} (the )?Correct Entity Annotation\.?( \(?(?P<entity_type>{pat_et})\)?)?$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} Correct Entity Annotation\. Named entity of type (?P<entity_type>{pat_et})\.$', re.IGNORECASE),
            re.compile(rf'^{self._pat_til_choice} {self._sp_pref}is a named entity of type (?P<entity_type>{pat_et}) \((?P<span>.+)\)$', re.IGNORECASE),
        ]

    def get_prompt(self, instruction_args: Dict[str, Any] = None, **kwargs):
        # if not self.batched:
        #     assert instruction_args.get('n_classify', 1) == 1  # sanity check
        #
        #     if self.dataset_name == 'mit-movie':
        #         if self.zero_shot:
        #             raise NotImplementedError
        #         else:
        #             raise NotImplementedError
        #     else:
        #         raise NotImplementedError
        # else:
        # TODO: for now, if not batched, just use the same prompt
        return super().get_prompt(sample2str_args=dict(highlight_span=self.highlight_span), instruction_args=instruction_args, **kwargs)

    def get_instruction(self, n_classify: int = 20, entity_type: str = None, triples_dir_name: str = None) -> str:
        # Here are 5 spoken queries to a movie dialogue system. Your task is to analyze each query and classify whether the span of text is a named entity of type Title.
        # A named movie title entity must be the name of a movie.
        # In particular, for each query with a span of text, please analyze and classify the span of text in context of the query into one of the following categories:
        # -   (A). Is a named entity of type movie title,
        # -   (B). Contains a named movie title entity but the span boundary is not precise,
        # -   (C). Is a named entity but category is not movie title,
        # -   (D). Not a named entity.
        if self.dataset_name == 'conll2003-no-misc':
            if n_classify == 1:
                ret = 'Here is a sentence from a news article.'
            else:
                ret = f'Here are {n_classify} sentences from news articles.'
            x_nm, x_nm_pl = 'the sentence', 'each sentence'
        elif self.dataset_name == 'wiki-gold-no-misc':
            if n_classify == 1:
                ret = 'Here is a sentence from a Wikipedia article.'
            else:
                ret = f'Here are {n_classify} sentences from Wikipedia articles.'
            x_nm, x_nm_pl = 'the sentence', 'each sentence'
        elif self.dataset_name in ['mit-movie', 'mit-restaurant']:
            kd = 'movies' if self.dataset_name == 'mit-movie' else 'restaurants'
            if n_classify == 1:
                ret = f'Here is a spoken query to a dialogue system about {kd}.'
            else:
                ret = f'Here are {n_classify} spoken queries to a dialogue system about {kd}.'
            x_nm, x_nm_pl = 'the query', 'each query'
        else:
            raise NotImplementedError
        act = 'classify'
        if self.cot:
            x = x_nm if n_classify == 1 else x_nm_pl
            act = f'analyze {x} and classify'
        elm_nm, elm_nm_pl, nm_type = 'named entity', 'named entities', 'entity type'
        # elm_nm, elm_nm_pl, nm_type = 'term', 'terms', 'term category'
        ret = f'{ret} Your task is to {act} whether the span of text is a {elm_nm} of type {entity_type}.\n'

        # self.et2info = Type2EntityCorrectionInfo(dataset_name=self.dataset_name, triples_dir_name=triples_dir_name)
        d = self.et2info(entity_type=entity_type)
        # defn, nm, eg = d['defn'], d['name'], d.get('examples')
        # if isinstance(defn, dict):
        #     defn = defn[elm_nm]
        # assert isinstance(defn, str)  # sanity check
        # if eg is None:
        #     ret += f'{defn}.\n'
        # else:
        #     n_eg = len(eg)
        #     assert n_eg > 0  # sanity check
        #     eg = [enclose_in_quote(x) for x in eg]
        #     if n_eg == 1:
        #         eg = eg[0]
        #     else:
        #         eg = ', '.join(eg[:-1]) + f' and {eg[-1]}'
        #     ret += f'{defn}, e.g. {eg}.\n'

        # no internal representation of `examples`, just merge them into the manually-written defn.; cos certain classes are harder
        defn, nm, nm_full = d['defn'], d['name'], d['name_full']
        if isinstance(defn, dict):
            defn = defn[elm_nm]
        assert isinstance(defn, str)  # sanity check
        # defn_at_end = False
        defn_at_end = True
        if not defn_at_end:
            ret += f'{defn}\n'

        act = 'analyze and classify' if self.cot else 'classify'
        x = 'for the given query' if n_classify == 1 else 'for each query'
        x_s = 'span of text'
        if self.highlight_span:
            if self.highlight_span == 'braces':
                punc = 'double braces'
            else:
                assert self.highlight_span == 'brackets'
                punc = 'brackets'
            x_s = f'{x_s} enclosed in {punc}'
        ret += (f'In particular, {x} with a {x_s}, '
                # f'please {act} the span of text in context of the query into one of the following categories:\n')
                # stress on classifying only the enclosed span
                f'please {act} the enclosed span of text in context of the query into one of the following categories:\n')

        options = [f'- {x}' for x in entity_type2entity_correction_options(entity_type=nm, entity_type_full=nm_full, element_name=elm_nm)]
        ret += '\n'.join(options) + '\n'

        # If the label is (B), you should provide the correct span boundary.
        # If the label is (C), you should provide the correct entity type. Pick an entity type from the following list:
        # [Viewers' Rating, Year, Genre, Director, MPAA Rating, Plot, Actor, Trailer, Song, Review, Character, other].
        ret += 'If the label is (B), you should provide the correct span boundary.\n'
        if elm_nm == 'named entity':
            ret += f'If the label is (C), you should provide the correct entity type. Pick an entity type from the following list:\n'
        else:
            assert elm_nm == 'term'
            ret += f'If the label is (C), you should provide the correct term category. Pick a term category from the following list:\n'
        ets = self.entity_types.copy()
        ets.remove(entity_type)
        ets.append(ENTITY_TYPE_OTHER)
        ret += f'{pl.nc(ets)}.\n'

        if defn_at_end:
            ret += '\n'
            ret += f'{defn}\n'
            # ret += f'\nEntity Annotation Schema: {defn}\n'  # try to be more explicit
            # ret += f'You should refer to the annotation schema above to determine the correct label.\n'
        return ret

    def load_triples(self, triples_dir_name: str = None) -> List[Dict[str, Any]]:
        return load_triples_w_logprob(dataset_name=self.dataset_name, dir_name=triples_dir_name)

    def get_uncertain_triples(
            self, triples_dir_name: str = None, logprob_thresh: float = 2e-2, top_n: Union[int, float] = 0.2,
            group_by_type: bool = True, **kwargs
    ) -> UncertainTriplesOutput:
        """
        Load (sentence, entity span, entity type) triples and select samples w/ low log prob

        :param triples_dir_name: dataset directory name containing the triples w/ logprob file
        :param logprob_thresh: logprob upperbound for selecting uncertain samples
        :param top_n: If an int, select top n samples; if a float, select top n% samples
        :param group_by_type: If true, sort and select top samples for each entity type separately
        :return dict of (entity type => uncertain samples)
        """
        # select uncertain triples capped by count & logprob, try
        #   1> rank solely by log prob
        #   2> first stratify by entity type, then rank by log prob
        triples = self.load_triples(triples_dir_name=triples_dir_name)
        return log_n_select_uncertain_triples(
            triples=triples, logprob_thresh=logprob_thresh, top_n=top_n, group_by_type=group_by_type, entity_types=self.entity_types,
            logger=self.logger, ec=self.ec, **kwargs
        )

    def get_et2info(self, triples_dir_name: str = None, correction_config: str = None):
        if correction_config is not None:
            self.et2info = Type2EntityCorrectionInfo.from_json(dataset_name=self.dataset_name, config=correction_config)
        else:
            self.et2info = Type2EntityCorrectionInfo(dataset_name=self.dataset_name, triples_dir_name=triples_dir_name)
        return self.et2info

    def write_completions(
            self, n_correct: int = 5, triples_dir_name: str = None, correction_config: str = None, output_dir_nm: str = None,
            shuffle_samples: Union[bool, int] = None, subset_entity_types: List[str] = None, uncertain_triple_args: Dict[str, Any] = None,
            generator: Union[random.Random, int] = None, **kwargs
    ):
        out_dir_dnm = output_dir_nm
        if subset_entity_types is not None:
            assert len(subset_entity_types) > 0
            assert (et in self.entity_types for et in subset_entity_types)  # sanity check

        output_path = dataset_name2data_dir(**self.dir_args, output_dir=f'{self.sample_type}-Res', output_postfix=out_dir_dnm).path
        add_file_handler(logger=self.logger, file_path=os_join(output_path, 'write-completion.log'))

        uncertain_args = dict(triples_dir_name=triples_dir_name, shuffle=shuffle_samples, **(uncertain_triple_args or dict()))
        et2triples = self.get_uncertain_triples(**uncertain_args).entity_type2triples

        prompts, fnms = [], []
        dir_nm2prompts, et2n_demo = dict(), dict()
        # self.et2info = Type2EntityCorrectionInfo(dataset_name=self.dataset_name, triples_dir_name=triples_dir_name)
        self.get_et2info(triples_dir_name=triples_dir_name, correction_config=correction_config)
        gen_ = get_random_generator(generator=generator)
        for et, triples_ in et2triples.items():
            if subset_entity_types is not None and et not in subset_entity_types:
                continue
            # if DEBUG:
            #     triples_ = triples_[:25]
            demos = self.et2info(entity_type=et)['demos']
            et2n_demo[et] = len(demos)

            samples_group = group_n(triples_, n=n_correct)
            instr_args = dict(entity_type=et, triples_dir_name=triples_dir_name)
            ppt_args = dict(demo_examples=demos, instruction_args=instr_args, generator=gen_)
            dir_nm2prompts[et] = prompts_ = [self.get_prompt(samples=samples, **ppt_args) for samples in samples_group]
            prompts += prompts_
            # print_prompts(prompt=prompts_[:5])
            # raise NotImplementedError

            os.makedirs(os_join(output_path, et), exist_ok=True)  # write completions to folders dedicated to each entity type
            fnms += [os_join(et, f'completion-{i+1}.txt') for i in range(len(prompts_))]
        # raise NotImplementedError
        if self.ec.have_edge_case:  # expect to see edge cases in highlighting the span to annotate when multi-occurrences of the span found
            self.logger.info(self.ec.summary())

        d_log = {
            'dataset-name': self.dataset_name, '#correct': n_correct, '#demo': et2n_demo, 'subset-entity-types': subset_entity_types,
            'triplet-samples-dir-name': triples_dir_name, 'shuffle-samples': shuffle_samples, 'output-path': output_path
        }
        # sic(prompts, len(prompts))
        # raise NotImplementedError

        ret = completions.write_completions(
            output_path=output_path, logger=self.logger, add_fl_writer=False,
            completion_type=self.sample_type, init_log=d_log, prompts=prompts, completion_fnms=fnms, dir_name2prompts=dir_nm2prompts,
            save_all_prompts=True, **kwargs
        )
        drop_file_handler(logger=self.logger)
        return ret

    def sanitize_entity_type(self, entity_type: str = None) -> str:
        """
        If the LLM-corrected entity type differs in casing, consider as the original case
        """
        for et in self.entity_types:
            if et.lower() == entity_type.lower():
                return et
        return entity_type  # no match found, use the original LLM correction

    def process_completions(
            self, dataset_dir_name: str = None, triples_dir_name: str = None, correction_config: str = None,
            completions_dir_name: completions.CompletionDirectoryDict = None,
            output_dir_name: str = None, expected_samples_per_completion: int = None,
            shuffle_samples: Union[bool, int] = None, logprobs: bool = False,
            uncertain_triple_args: Dict[str, Any] = None, override_w_manual: Union[bool, str] = False,
            subset_entity_types: List[str] = None, lowercase: bool = None
    ) -> dataset.NerProcessOutput:
        """
        Given the original NER dataset directory, containing processed NER samples & triplet logprob file,
            and the completions directory, containing corrections on challenging triplets,
                process the corrections to override the previously processed NER samples

        TODO: after processed all corrections, for triplets that changed, select similar & un-corrected samples to re-correct
        """
        from stefutil import sic

        if subset_entity_types is not None:
            assert len(subset_entity_types) > 0 and (et in self.entity_types for et in subset_entity_types)  # sanity check
        d_out = dataset_name2data_dir(
            **self.dir_args, output_dir=f'{self.processed_type}-Dataset', output_postfix=output_dir_name, timestamp='short-date')
        output_path, base_path = d_out.path, d_out.base_path
        init_log = {
            'class-name': self.__class__.__qualname__, 'metadata': self.meta(),
            'dataset-dir-name': dataset_dir_name, 'triplet-samples-dir-name': triples_dir_name,
            'completions-dir-name': completions_dir_name, 'output-dir-name': output_dir_name, 'output-path': output_path,
            'expected-samples-per-completion': expected_samples_per_completion,
            'generated-samples-dir-name': triples_dir_name, 'shuffle-samples': shuffle_samples, 'lowercase': lowercase
        }
        n_expect = expected_samples_per_completion

        uncertain_args = dict(triples_dir_name=triples_dir_name, shuffle=shuffle_samples, **(uncertain_triple_args or dict()))
        add_file_handler(logger=self.logger, file_path=os_join(output_path, f'process-{self.sample_type.lower()}.log'))
        # so that uncertain triples will also be logged below
        crt_out = self.get_uncertain_triples(**uncertain_args)
        et2triples, d_log_count = crt_out.entity_type2triples, crt_out.d_log
        n_total, n_ch, et2n_ch = (
            d_log_count[k] for k in ('#total-triples', '#challenging-triples-kept', '#challenging-triples-kept-by-type'))
        d_log_count = {'#total-triples': n_total, '#triples-analyzed': n_ch, '#triples-analyzed-by-type': et2n_ch}

        t = Timer()
        base_path = os_join(base_path, completions_dir_name)
        completions.process_completions_init(
            completion_base_path=base_path, output_path=output_path, init_log=init_log, logger=self.logger, add_fl_writer=False)

        et2ret = defaultdict(list)
        d_n_cpl = dict()
        n_et = len(et2triples)
        # self.et2info = Type2EntityCorrectionInfo(dataset_name=self.dataset_name, triples_dir_name=triples_dir_name)
        self.get_et2info(triples_dir_name=triples_dir_name, correction_config=correction_config)
        for i, (et, triples) in enumerate(et2triples.items()):
            if subset_entity_types is not None and et not in subset_entity_types:
                continue
            i: int
            tqdm_type = f'{pl.i(et)} ({pl.i(i+1)}/{pl.i(n_et)}) Correction'
            completions.log_prompt_eg(dir_name=et, base_path=base_path, logger=self.logger)
            self.logger.info(f'Processing entity type directory {pl.i(et)}')
            it = completions.iter_completions(
                dir_name=et, base_path=base_path, completion_type=tqdm_type, logger=self.logger, logprobs=logprobs)
            d_n_cpl[et] = n_cpl = len(it.filepaths)

            ets_other = self.entity_types.copy()  # other relevant entity types for the dataset
            ets_other.remove(et)
            pattern_label_no_correction = self.get_pattern_label_no_correction(entity_type=et)

            d_et = self.et2info(entity_type=et)
            et_nm, et_eqv = d_et['name'], d_et.get('equivalents', [])
            # pool of entity type spans that are considered the same as in prompt
            ets_same = [et.lower(), et_nm.lower()] + [x.lower() for x in et_eqv]

            def span_is_current_entity_type(s_: str = None) -> bool:
                # checks whether the span can be considered as the current entity type
                return s_.lower() in ets_same

            for i_cpl, c in enumerate(it.iter):
                is_last_group = i_cpl == n_cpl - 1
                cpl, fnm, p_fnm = c.content, c.filename, c.pretty_filename

                i_s, i_e = i_cpl * n_expect, (i_cpl + 1) * n_expect  # get triples corrected in this completion
                triples_ = triples[i_s:i_e]
                n_expect_curr = len(triples_)
                assert n_expect_curr <= expected_samples_per_completion  # the last group may have fewer samples
                if self.cot:
                    raise NotImplementedError
                crt_out = self._split_from_sentence_samples(
                    completion=cpl, is_last_group=is_last_group, filename=p_fnm,
                    pattern_w_sentence=self.pattern_sample_search, pattern_wo_sentence=self.pattern_sample_search_edge,
                    edge_split_args=dict(silent=True)  # for the last group edge case, see below
                )
                crt_out, sent_found = crt_out.split_output, crt_out.sentence_found
                if crt_out.success:
                    if len(crt_out.samples) != n_expect_curr:
                        sic(cpl)
                        sic(self.pattern_sample_search, self.pattern_sample_search_edge)
                        sic(len(crt_out.samples), n_expect_curr)
                        sic(crt_out.samples)
                    assert len(crt_out.samples) == n_expect_curr  # sanity check generated samples match # in prompt
                    ms = crt_out.matches
                    has_span = True
                else:  # must be the last prompt w/ only 1 sample to correct
                    n = 1
                    if not (is_last_group and n_expect_curr == 1):
                        # LLM didn't generate the span for general samples, e.g.
                        #   `1. Label: (C). The span is a named entity but category is not mpaa rating. Entity type: Viewers' Rating
                        #   2. Label: (C). The span is a named entity but category is not mpaa rating. Entity type: Viewers' Rating
                        #   3. Label: (C). The span is a named entity but category is not mpaa rating. Entity type: Viewers' Rating`
                        d_log = dict(completion=cpl, filename=p_fnm, n_expect=n_expect_curr)
                        msg = f"Edge Case: LLM didn't generate the span for general batched samples w/ {pl.i(d_log)}"
                        self.ec(msg=msg, kind='span-not-re-generated', args=d_log)
                        n = len(triples_)
                    ms = patterns.find_non_overlap_matches(pattern=self.pattern_label_search, text=f'{cpl}\n', return_matches=True)
                    assert len(ms) == n
                    has_span = False

                # s2lp = None
                # if logprobs:
                #     s2lp = logprob.Span2LogProb(
                #         logprobs=c.logprobs, completion=cpl, sample_type='correction-label', ec=self.ec, logger=self.logger)
                samples_out = []
                for mch, triple in zip(ms, triples_):
                    sample_str = mch.group()
                    sent, span, et_ = triple['sentence'], triple['span'], triple['entity_type']
                    assert et == et_  # sanity check entity type matches
                    if has_span:
                        span_gen, label_str = mch.group('span', 'label')
                    else:
                        span_gen, label_str = span, mch.group('label')
                    label_str = label_str.strip()
                    if sent_found:
                        sent_gen = mch.group('sentence')
                        if self.highlight_span:
                            sent_gen = edit.drop_brackets_in_text(
                                text=sent_gen, pattern_emph=self.pattern_emph, ec=self.ec, sample_kind='sentence')
                        drop = self.check_sentence_diff(sentence_in_prompt=sent, sentence_in_response=sent_gen)
                        if drop:
                            sic(sent, sent_gen, label_str, sample_str, self.highlight_span)
                        assert not drop  # sanity check generated sentence matches sentence in prompt

                    if logprobs:
                        # the span may appear multiple times in LLM response since the output can be quite simple,
                        #   so need to specify which one to extract logprob
                        pass

                    span_gen = edit.drop_enclosing_quotes(span_gen.strip())
                    # generated span may not match span in prompt; but at least, sanity check generated span is at least in sentence
                    if span_gen.lower() not in sent.lower():
                        # span generated not even in sentence, ignore and use the span in prompt
                        d_log = dict(sentence=sent, span_expected=span, span_generated=span_gen, filename=p_fnm)
                        msg = f'Edge Case: LLM generated span not in sentence w/ {pl.i(d_log, indent=1)}'
                        self.ec(msg=msg, kind='generated-span-not-in-sentence', args=d_log)
                        span_gen = span

                    # get the generated label & potential correction
                    mch_nc = patterns.match_row(text=label_str, pattern=pattern_label_no_correction, accept_no_match=True, log=False)
                    mch_crt = patterns.match_row(text=label_str, pattern=self.pattern_label_has_correction, accept_no_match=True, log=False, verbose=True)

                    # if fnm.endswith('-36'):
                    #     sic(sample_str, label_str)
                    #     raise NotImplementedError
                    nc, cr = mch_nc is not None, mch_crt is not None
                    if nc and cr:
                        # matched both, break tie by the choice letter LLM generated
                        choice1, choice2 = mch_crt.group('choice'), mch_nc.group('choice')
                        assert choice1 == choice2  # sanity check
                        correction = mch_crt.groupdict().get('correction')
                        assert correction is not None and span_is_current_entity_type(correction)
                        label = CorrectionLabel.choice_to_label(choice=choice1)
                        # note that to match the no correction pattern,
                        #   the generated entity type must be the same as the one in the current prompt
                        # If the generated choice letter is A, meaning correct, assume is my processing edge case and move on, e.g.
                        #   `Label: (A). The span is a named entity of type movie genre.`
                        # If the generated choice letter is C, meaning wrong type, this is an LLM edge case, will be handled later
                        #   `Label: (C). The span is a named entity of type viewers\' rating.`
                        # if label != LABEL_CORRECT:
                        #     sic(mch.group(), label_str, mch_nc, mch_crt)
                        if label == LABEL_CORRECT:
                            mch_crt = None
                            cr = False
                        else:
                            assert label in [LABEL_WRONG_TYPE, LABEL_WRONG_BOUNDARY]  # in rare cases, also wrong boundary
                            assert span_is_current_entity_type(correction)
                            mch_nc, nc = None, False
                    if not ((nc or cr) and not (nc and cr)):
                        sic(mch.group(), label_str, mch_nc, mch_crt)
                        # sic(pattern_label_no_correction, self.pattern_label_has_correction)
                    assert (nc or cr) and not (nc and cr)  # sanity check only one match
                    has_correction = cr

                    if not has_correction:
                        choice, correction_type, correction = mch_nc.group('choice'), None, None
                    else:
                        choice = mch_crt.group('choice')
                        gd = mch_crt.groupdict()
                        correction_type, correction = gd.get('correction_type'), gd.get('correction')
                        if correction_type is not None:
                            correction_type = correction_type.strip()
                        if correction is not None:
                            correction = edit.drop_enclosing_quotes(correction.strip())
                    label = CorrectionLabel.choice_to_label(choice=choice)

                    # if sent.startswith('Princess Diana was a member of the'):
                    #     sic(sample_str, label_str)
                    #     sic(mch_nc, mch_crt, correction)
                    #     raise NotImplementedError
                    # if label == LABEL_WRONG_TYPE and nc:
                    #     # LLM declares wrong type, but matches the pattern w/ no correction
                    #     # since pattern dynamically matches the expected type, so effectively LLM corrects the span to the same entity type
                    #     raise NotImplementedError
                    if not has_correction:  # sanity check one-to-one match between LLM-generated choice & pattern matched
                        d_log = dict(sentence=sent, span=span, entity_type=et, sample=mch.group(), label=mch_nc.group())
                        if label == LABEL_WRONG_TYPE:
                            # the LLM gets confused on the choice letters vs the meaning, e.g.
                            #   `(C). The span is not a named entity of type movie genre.`
                            #       should actually be `D` for not a named entity
                            d_log['correct_letter'] = '(D)'
                            label = LABEL_NOT_NAMED_ENTITY
                        elif label == LABEL_WRONG_BOUNDARY:
                            # e.g.
                            #   `3. Text Span: "Steve McQueen"
                            #   Label: (B)`  <= didn't provide any correction, so can't correct anything, declare as correct
                            d_log['correct_letter'] = '(A)'
                            label = LABEL_CORRECT
                        if label in [LABEL_WRONG_BOUNDARY, LABEL_WRONG_TYPE]:
                            msg = f'Edge Case: LLM confused choice letter when declaring is/not a named entity w/ {pl.i(d_log)}'
                            self.ec(msg=msg, kind='confused-choice-letter', args=d_log)
                        # otherwise, it's normal to not have corrections

                        group_dict = mch_nc.groupdict()
                        if 'span' in group_dict:
                            # edge case: LLM provided the span again in parentheses, e.g.
                            #   `(A). The span is a named entity of type amenity (Upscale)`
                            span_gen_ = group_dict['span']
                            # sanity check match the span in prompt
                            assert span.lower() == span_gen_.lower()
                            # sic(span_gen_)
                            # raise NotImplementedError
                    else:
                        if label == LABEL_CORRECT:
                            # edge case: LLM declares correct but generates some weird stuff in parentheses, e.g.
                            #   `(A). The span is a named entity of type amenity (Buffet).`
                            group_dict = mch_crt.groupdict()
                            correction_type, correction = group_dict.get('correction_type'), group_dict.get('correction')
                            assert correction_type is not None and correction is not None
                            assert correction_type in self.correction_types_type

                            # sanity check indeed the correction contains a parenthesis
                            mch_pair = re.match(pattern=r'^(?P<gen_et>.+?) \((?P<correction>.+?)\)$', string=correction)
                            if mch_pair is None:
                                sic(label_str, correction)
                            assert mch_pair is not None
                            gen_et, correction = mch_pair.group('gen_et'), mch_pair.group('correction')
                            assert span_is_current_entity_type(gen_et)  # sanity check LLM indeed declares correct type

                            # sanity check on the weird stuff in parentheses
                            span_gen_, correction_ = span_gen.lower(), correction.lower()
                            # for (`buffet`, `buffets`), (`delivery`, `delivery`)
                            assert span_gen_ == correction_ or span_gen_ == f'{correction_}s'

                            d_log = dict(sentence=sent, span=span, entity_type=et, span_generated=span_gen, correction=correction)
                            msg = f'Edge Case: LLM declared correct but generated correction, modified as no correction w/ {pl.i(d_log)}'
                            self.ec(msg=msg, kind='declare-correct-but-generated-correction', args=d_log)

                            label, has_correction = LABEL_CORRECT, False
                            mch_nc = mch_crt  # for entity type sanity check below
                        else:
                            assert label in [LABEL_WRONG_BOUNDARY, LABEL_WRONG_TYPE]
                    if has_correction:
                        d_log = dict(sentence=sent, span=span, entity_type=et, sample=mch.group(), filename=p_fnm)
                        if correction is None:
                            assert correction_type is not None
                            # since correction is missing, can't process the correction further; declare as no correction
                            assert label in [LABEL_WRONG_BOUNDARY, LABEL_WRONG_TYPE]
                            if label == LABEL_WRONG_BOUNDARY:
                                assert correction_type.lower() in self.correction_types_span
                                kd = 'wrong-span-no-correction'
                            else:
                                assert correction_type.lower() in self.correction_types_type
                                kd = 'wrong-type-no-correction'
                            label, has_correction = LABEL_CORRECT, False
                            mch_nc = mch_crt  # for entity type sanity check below
                            msg = f'Edge Case: LLM declared wrong {correction_type} but provided no correction w/ {pl.i(d_log)}'
                            self.ec(msg=f'{msg}, modified as correct', kind=kd, args=d_log)
                        elif correction_type is None:
                            assert label in [LABEL_WRONG_TYPE, LABEL_WRONG_BOUNDARY]
                            correction_type = 'entity type'
                            # correction_type = 'entity type' if label == LABEL_WRONG_TYPE else 'span boundary'
                            d_log['correction'] = correction
                            msg = (f"Edge Case: LLM didn't follow the correction template, "
                                   f"no type of correction provided w/ {pl.i(d_log, indent=1)}")
                            self.ec(msg=msg, kind='correction-type-missing', args=d_log)
                            # raise NotImplementedError

                    # handle generation edge cases
                    if span_gen.lower() != span.lower():  # LLM wants to correct the boundary, but even corrected the span given
                        d_log = dict(sentence=sent, span_expected=span, span_generated=span_gen, entity_type=et)

                        if label == LABEL_WRONG_BOUNDARY:
                            if not (correction == span_gen or correction == span or span_gen in span):
                                sic(correction, span, span_gen, mch.group())
                            if correction == span_gen or correction == span or span_gen in span:
                                if correction == span_gen:
                                    msg = f'Edge Case: LLM modified the input span in prompt w/ {pl.i(d_log)}, modified as wrong span boundary'
                                    self.ec(msg=msg, kind='corrected-input-span', args=d_log)
                                elif correction == span:  # effectively no modification, LLM just modified the candidate span
                                    msg = (f'Edge Case: LLM modified the candidate span in prompt w/ {pl.i(d_log)} '
                                           f'and corrected to original span in prompt')
                                    self.ec(msg=f'{msg}, modified as no correction', kind='modified-input-span-&-corrected-to-same', args=d_log)
                                    label, has_correction = LABEL_CORRECT, False
                                    mch_nc = mch_crt  # for entity type sanity check below
                                else:  # i.e. `correction != span`
                                    assert self.dataset_name == 'mit-movie' and et == 'Genre'
                                    # note already checked all 3 strings are different
                                    assert span_gen in span and correction in span_gen
                                    d_log['correction'] = correction
                                    # span in prompt => span in response => span correction,
                                    # if they are all one by one smaller than each other, allow this edge case
                                    msg = f'Edge Case: LLM modified the input span in prompt w/ {pl.i(d_log)} and provided a substring as span correction'
                                    self.ec(msg=msg, kind='modified-input-span-&-corrected-as-substring', args=d_log)
                            else:
                                assert not check.have_word_overlap(span1=span, span2=span_gen)
                                assert not check.have_word_overlap(span1=span, span2=correction)

                                # re-generated span & correction all completely different;
                                # just ignore the correction in this case, i.e. treat as no change
                                msg = (f'Edge Case: LLM declared wrong span boundary but '
                                       f'annotated a completely different span boundary w/ {pl.i(d_log)}')
                                msg = f'{msg}, modified as no correction'
                                d_log['correction'] = correction
                                self.ec(msg=msg, kind='declare-wrong-span-but-on-different-span', args=d_log)
                                label, has_correction = LABEL_CORRECT, False
                                mch_nc = mch_crt  # for entity type sanity check below
                        else:
                            if label == LABEL_CORRECT:  # An even worse case: LLM corrects the span but choice option is wrong
                                msg = f'Edge Case: LLM declared correct but provided a different span boundary w/ {pl.i(d_log)}'
                                self.ec(msg=msg, kind='declare-yes-but-corrected-input-span', args=d_log)

                                # in this case, no correction is provided, we will feed the span given in the response as the correction
                                label, has_correction = LABEL_WRONG_BOUNDARY, True
                                correction_type, correction = 'span boundary', span_gen
                            elif label == LABEL_WRONG_TYPE:
                                # e.g.
                                #   `3. Text Span: "Gone with the Wind"
                                #       Label: (C). The correct entity type is Title.`
                                if check.have_word_overlap(span1=span, span2=span_gen):
                                    if span_is_current_entity_type(correction):
                                        # in this case, LLM declares wrong entity type but provided the same type as given,
                                        # LLM also provides a different span, so we consider it as actually corrected the span
                                        label = LABEL_WRONG_BOUNDARY
                                        correction_type, correction = 'span boundary', span_gen
                                        msg = f'Edge Case: LLM declared wrong entity type but provided the same type and different span w/ {pl.i(d_log)}'
                                        self.ec(msg=msg, kind='declare-wrong-type-but-corrected-input-span', args=d_log)
                                    else:
                                        # LLM classified a different span as a different entity type, even though they overlap...
                                        # many options to handle this:
                                        #   1> correct span only, 2> correct type only, 3> correct both, 4> ignore
                                        # to be safe, ignore this correction
                                        d_log['correction'] = correction
                                        msg = f'Edge Case: LLM declared wrong entity type but provided a different span w/ {pl.i(d_log)}'
                                        msg = f'{msg}, modified as no correction'
                                        self.ec(msg=msg, kind='declare-wrong-type-and-corrected-span', args=d_log, failed=True)
                                        label, has_correction = LABEL_CORRECT, False
                                        mch_nc = mch_crt  # for entity type sanity check below
                                else:
                                    # completely different spans; just ignore the correction in this case, i.e. treat as no change
                                    msg = f'Edge Case: LLM declared wrong entity type but provided a different span boundary w/ {pl.i(d_log)}'
                                    msg = f'{msg}, modified as not a named entity'
                                    d_log['correction'] = correction
                                    self.ec(msg=msg, kind='declare-wrong-type-but-on-different-span', args=d_log)
                                    label, has_correction = LABEL_NOT_NAMED_ENTITY, False
                            else:
                                assert label == LABEL_NOT_NAMED_ENTITY
                                # if not have_word_overlap(span1=span, span2=span_gen):
                                #     sic(mch.group(), label_str, mch_nc, mch_crt, span, span_gen)
                                if check.have_word_overlap(span1=span, span2=span_gen):
                                    # in this case, since it's an overlap, still treats it as a valid non-entity correction
                                    # and omit the different span generated
                                    msg = f'Edge Case: LLM declared not a named entity but provided a different span boundary w/ {pl.i(d_log)}'
                                    self.ec(msg=msg, kind='declare-none-entity-but-corrected-input-span', args=d_log)
                                else:
                                    # 1st instance of this case:
                                    #   `2. Query: "Who {{directed}} the movie Coco?"`
                                    #   `2. Text Span: "the movie Coco"
                                    #   `Label: (D)`
                                    # in this case, the correct label is actually non-entity,
                                    #   but cannot trust this for other potential cases
                                    # completely different spans; just ignore the correction in this case, i.e. treat as no change
                                    msg = f'Edge Case: LLM declared not a named entity on a completely different span boundary w/ {pl.i(d_log)}'
                                    msg = f'{msg}, modified as not a named entity'
                                    self.ec(msg=msg, kind='declare-none-entity-on-different-span', args=d_log)

                    d_log = dict(sentence=sent, span=span, entity_type=et)
                    if label == LABEL_WRONG_BOUNDARY and correction_type in ['entity type']:
                        # LLM confused the choice option w/ the description, e.g.
                        #   Letter B corresponds should be correcting the span; in this case, B but should've been C
                        #   `1. Query: "Can you recommend an {{Authentic dialogue}}-driven movie from the 90s"
                        #   Text Span: "Authentic dialogue"
                        #   Label: (B). The correct entity type is Movie Review.`
                        # correct the label to be wrong type since LLM already provided an entity type correction

                        d_log.update(sample=mch.group(), label=mch_crt.group(), correct_letter='(C)')
                        msg = f'Edge Case: LLM confused choice letter when declaring wrong type w/ {pl.i(d_log)}'
                        self.ec(msg=msg, kind='confused-choice-letter', args=d_log)
                        label = LABEL_WRONG_TYPE

                    if label == LABEL_CORRECT:
                        et_gen = mch_nc.groupdict().get('entity_type')
                        if et_gen is not None:
                            if not span_is_current_entity_type(et_gen):
                                sic(label, et_gen)
                            assert span_is_current_entity_type(et_gen)  # if provided the choice description, should match the entity type
                    elif label == LABEL_WRONG_BOUNDARY:
                        if correction_type.lower() not in self.correction_types_span:
                            sic(correction_type, mch.group(), mch_nc, mch_crt)
                            sic(mch_crt.group(), correction_type, correction)
                        assert correction_type.lower() in self.correction_types_span  # sanity check
                        d_log['corrected_span'] = correct_span = edit.drop_enclosing_quotes(correction)
                        if correct_span.lower() == span.lower():  # ignore corrections that potentially just correct the case
                            msg = f'Edge Case: LLM declared wrong span boundary but provided the same span w/ {pl.i(d_log)}'
                            self.ec(msg=msg, kind='wrong-span-gen-same', args=d_log)
                            label, has_correction = LABEL_CORRECT, False
                        elif f'{span}.' == correct_span:
                            # Edge case: only difference is a trailing period added
                            # sanity check due to the original sentence ends in a period and span is at the end of the sentence
                            assert sent[-1] == '.' and span[-1] != '.'
                            assert sent[-len(span)-1:] == f'{span}.'
                            msg = f'Edge Case: LLM declared wrong span boundary but provided the same span w/ trailing period w/ {pl.i(d_log)}'
                            self.ec(msg=msg, kind='wrong-span-end-period', args=d_log)
                            label, has_correction = LABEL_CORRECT, False
                        else:
                            if not check.have_word_overlap(span1=span, span2=correct_span):
                                # no overlap between the span, this is not span refinement, more like classifying another entity
                                msg = (f'Edge Case: LLM declared wrong span boundary '
                                       f'but provided a completely different span w/ {pl.i(d_log)}')
                                msg = f'{msg}, modified as not a named entity'
                                self.ec(msg=msg, kind='corrected-span-too-different', args=d_log)

                                # Seen 2 instances of this on Character: e.g.
                                #   `main character => The Lion King`
                                #   `lead role => Mahershala Ali`
                                # assume LLM intend to say the span is not a named entity, and declare as such
                                label, has_correction = LABEL_NOT_NAMED_ENTITY, False

                        # sanity check the correct span is still in the sentence
                        if label == LABEL_WRONG_BOUNDARY:
                            if correct_span in sent or correct_span.lower() in sent.lower():
                                # multi_occur = False
                                if correct_span in sent:
                                    multi_occur = sent.count(correct_span) > 1
                                else:
                                    multi_occur = sent.lower().count(correct_span.lower()) > 1
                                # may occur multiple times, e.g.
                                #   X: `Who is the director of the {{action movie}} acclaimed for its thrilling plot and intense action sequences
                                #   Text Span: `action movie`
                                #   Corrected Text Span: `action`
                                if multi_occur:
                                    # should not be a problem, as long as original span only occurred once in the sentence,
                                    #   cos can find the right entity to replace
                                    assert span.count(correct_span) == 1
                                    msg = f"Edge Case: LLM-corrected span occurred multiple times in the sentence w/ {pl.i(d_log)}"
                                    self.ec(msg=msg, kind='corrected-span-multi-occur', args=d_log)
                            else:
                                msg = f'Edge Case: LLM declared wrong span boundary but provided a span not in the sentence w/ {pl.i(d_log)}'
                                self.ec(msg=msg, kind='corrected-span-not-in-sentence', args=d_log)
                                # in just case, can't use the LLM correction, modify to no correction
                                label, has_correction = LABEL_CORRECT, False

                    elif label == LABEL_WRONG_TYPE:
                        if correction_type.lower() not in self.correction_types_type:
                            sic(correction_type, mch.group(), mch_nc, mch_crt)
                            sic(mch_crt.group(), fnm)
                        assert correction_type.lower() in self.correction_types_type  # sanity check

                        correction = edit.drop_enclosing_brackets(s=correction)
                        correction = self.sanitize_entity_type(entity_type=correction)

                        # sanity check actually corrected to a different entity type
                        if span_is_current_entity_type(correction):
                            msg = f'Edge Case: LLM declared wrong entity type but provided the same entity type w/ {pl.i(d_log)}'
                            self.ec(msg=msg, kind='wrong-type-gen-same', args=d_log)
                            # in this case, just drop the correction
                            label, has_correction = LABEL_CORRECT, False
                            # the LLM provided correction can be considered the entity type, but not exactly,
                            #   e.g. `Popularity Level` for `Viewers' Rating`
                            #       so modify it
                            correction = et

                        # the corrected entity type may not be within the list of entity types given, e.g. `Award` instead of `other`
                        elif correction not in ets_other and correction.lower() != ENTITY_TYPE_OTHER:
                            d_log['corrected_entity_type'] = correction
                            msg = f'Edge Case: LLM corrected to unknown entity type w/ {pl.i(d_log)}'
                            self.ec(msg=msg, kind='type-correct-unknown', args=d_log)

                    if not has_correction:
                        assert label in [LABEL_CORRECT, LABEL_NOT_NAMED_ENTITY]
                    else:
                        assert label in [LABEL_WRONG_BOUNDARY, LABEL_WRONG_TYPE]
                    #     raise NotImplementedError

                    samples_out.append(CorrectionSample(
                        sentence=sent, span=span, entity_type=et, correction_label=label, correction=correction,
                        span_index=triple.get('index'), span_index_super=triple.get('index_super')))
                et2ret[et] += samples_out
            if len(et2ret[et]) != len(triples):
                sic(len(et2ret[et]), len(triples))
            assert len(et2ret[et]) == len(triples)  # sanity check all corrections should be successfully processed
        self.logger.info(self.ec.summary())
        self.ec.clear()

        log_correction_samples(entity_type2corrections=et2ret, logger=self.logger, ec=self.ec)

        if override_w_manual:
            et2ret = override_w_manual_corrections(
                entity_type2corrections=et2ret, dataset_name=self.dataset_name, logger=self.logger, manual_edit_dir_name=override_w_manual)

        d_log_count['#completions'] = d_n_cpl
        d_write = {'dataset-name': self.dataset_name, 'triples-dir-name': triples_dir_name, 'completions-dir-name': completions_dir_name}
        # TODO: add file-save after the corrections are validated, i.e. corrections that don't results in NER sample overlap?
        log_n_save_corrections(entity_type2corrections=et2ret, output_path=output_path, timer=t, d_log_count=d_log_count, logger=self.logger, **d_write)

        # if True:
        #     raise NotImplementedError

        # override the original NER samples w/ the processed corrections
        data_dir_nm = dataset_dir_name or triples_dir_name
        # first, load the samples
        ner_samples = dataset.dataset_dir_name2ner_samples(dataset_name=self.dataset_name, dataset_dir_name=data_dir_nm).samples
        corrections: List[CorrectionSample] = sum(et2ret.values(), start=[])
        assert len(corrections) == len(set(corrections))  # sanity check corrections are all unique

        # match processed triples to the corresponding sample
        sample_idx2corrections = defaultdict(list)
        for crt in corrections:
            ner_samples_match = [(i, crt.match_ner_sample(sample=s)) for i, s in enumerate(ner_samples)]
            ner_samples_match = [(i, m) for (i, m) in ner_samples_match if m.is_match]
            assert len(ner_samples_match) == 1  # sanity check each correction matches exactly 1 NER sample

            i_sample, mch = ner_samples_match[0]
            ner_sample = ner_samples[i_sample]
            mch_e_idx = mch.matched_entity_index
            assert mch_e_idx is not None and crt.span == ner_sample.entity_names[mch_e_idx]  # sanity check span matches

            if crt.correction_label != LABEL_CORRECT:  # correction needed
                sample_idx2corrections[i_sample].append(crt)
        idxs = sorted(sample_idx2corrections.keys())  # re-insert the dict to keep the same iteration order as the original samples
        sample_idx2corrections = {i: sample_idx2corrections[i] for i in idxs}

        samples_corrected = []
        entities_corrected: Dict[CorrectionLabel, List[Tuple[str, str]]] = defaultdict(list)
        it = tqdm(sample_idx2corrections.items(), desc='Processing corrections', unit='sample', total=len(sample_idx2corrections))
        for i_ner, crts in it:
            ner_sample = ner_samples[i_ner]
            sent, enms_ori, ets_ori = ner_sample.sentence, ner_sample.entity_names, ner_sample.entity_types
            crt_out = ner_sample2corrected_sample(sample=ner_sample, corrections=crts, allowed_entity_types=self.entity_types)
            enms, ets = crt_out.entity_names, crt_out.entity_types
            d_log = dict(
                sentence=sent, entity_names_ori=enms_ori, entity_types_ori=ets_ori,
                entity_names_corrected=enms, entity_types_corrected=ets)

            # sanity check after correction, sample is still valid
            assert check.entities_in_sentence(sentence=sent, entity_names=enms, ignore_case=True).all_found

            # sanity check no additional punctuations added to the entities
            out = edit.drop_entities_enclosing_puncs(entity_names=enms, dataset_name=self.dataset_name, drop='both', ec=self.ec, d_log=d_log)
            if len(out.entity_names_modified) != 0:
                sic(enms, out)
            assert len(out.entity_names_modified) == 0

            # to filter out edge case: overlapping due to no ordering only
            #   X: Who directed the film The Matrix and its sequel, The Matrix Reloaded, featuring Morpheus?
            #   Entities: [The Matrix, The Matrix Reloaded, Morpheus]
            ovl_args = dict(sentence=sent, entity_names=enms, ignore_case=True)
            if check.entities_overlapping(**ovl_args, search_in_order=False).overlap and \
                    check.entities_overlapping(**ovl_args, search_in_order=True).overlap:
                crt_boundary_change = [crt for crt in crts if crt.correction_label == LABEL_WRONG_BOUNDARY]
                n_boundary_change = len(crt_boundary_change)
                if n_boundary_change not in [1, 2]:
                    sic(sent, enms_ori, ets_ori)
                    sic(crts, enms, ets, n_boundary_change)
                assert n_boundary_change in [1, 2]

                msg = f'Edge Case: LLM-corrected entity span boundaries caused entity overlap'
                all_correction_dropped = False
                if n_boundary_change == 1:
                    # just 1 wrong boundary, can always resolve by just dropping, since the original sample is valid
                    # a single span boundary correction caused the entities to overlap, drop that correction
                    crt = crt_boundary_change[0]
                    # d_args = dict(original_span=crt.span, corrected_span=crt.correction)
                    spans_w_crts = [(crt.span, crt.correction)]

                    all_correction_dropped = len(crts) == 1
                    if not all_correction_dropped:
                        # re-correct the ner sample cos there are remaining corrections
                        crts_ = deepcopy(crts)
                        crts_.remove(crt)
                        crt_out = ner_sample2corrected_sample(sample=ner_sample, corrections=crts_, allowed_entity_types=self.entity_types)
                        enms, ets = crt_out.entity_names, crt_out.entity_types
                        d_log.update(entity_names_corrected_final=enms, entity_types_corrected_final=ets)
                else:
                    assert n_boundary_change == 2  # sanity check both corrections are modifying span boundaries

                    def correction_expands_boundary(crt_: CorrectionSample) -> bool:
                        # if crt_.correction is None or crt_.span == crt_.correction:
                        #     sic(sample, enms_ori, ets_ori)
                        #     sic(enms, ets, crts, crt_)
                        assert crt_.correction is not None and crt_.span is not None and crt_.correction != crt_.span  # sanity check
                        return crt_.correction > crt_.span
                    crt1, crt2 = crt_boundary_change
                    ce1, ce2 = correction_expands_boundary(crt1), correction_expands_boundary(crt2)
                    if ce1 is True and ce2 is True:
                        # edge case: LLM corrects the same multi-occurring span twice, instead of correcting 2 different spans
                        #   e.g. X: `I'm craving a dish with spicy flavors, can you recommend a restaurant that serves spicy food?`
                        #       Correction twice: `spicy` => `spicy flavors`
                        # sanity check this is indeed the case
                        c1_, c2_ = asdict(crt1), asdict(crt2)
                        c1_.pop('span_index')
                        c2_.pop('span_index')
                        if c1_ != c2_:
                            # a more severe edge case: 2 different spans corrected, and any one of them already causes overlap
                            #   e.g. X: `What is the plot of the greatest dreams-related movie of all time?`
                            #       Original entities: `greatest`, `Dreams`
                            #       Corrected entities: `greatest dreams-related movie`, `Dreams-related movie`
                            # sic(sent, enms_ori, ets_ori)
                            # sic(c1_, c2_)
                            # in this case, just drop one arbitrarily;
                            # later code will see if the remaining one still causes overlap, and if so, drop that one too
                            if random.random() < 0.5:
                                ce2 = False
                            else:
                                ce1 = False
                        else:
                            span, corrected_span = crt1.span, crt1.correction
                            assert sent.count(span) == 2 and sent.count(corrected_span) == 1

                            # one correction is modifying the wrong span, check which one it is, and drop that one
                            ms_ = patterns.find_match(text=sent, keyword=span)
                            assert len(ms_) == 2  # sanity check
                            # sic(crt1, crt2)
                            # sic(ms_)
                            span_starts = [m.start('keyword') for m in ms_]
                            # check which span is the wrong one
                            n_char_crt = len(corrected_span)
                            corrected_correct_span = [sent[i:i+n_char_crt] == corrected_span for i in span_starts]
                            assert corrected_correct_span.count(True) == 1  # sanity check
                            idx_correct = corrected_correct_span.index(True)
                            if idx_correct == 0:  # drop the wrong correction
                                ce1 = False  # i.e., drop crt2
                            else:
                                assert idx_correct == 1  # sanity check
                                ce2 = False  # i.e., drop crt1
                    # sanity check the culprit is a single correction, so then just drop that one
                    if not ((ce1 or ce2) and not (ce1 and ce2)):
                        sic(crt1, crt2, ce1, ce2)
                    assert (ce1 or ce2) and not (ce1 and ce2)  # sanity check only 1 correction expands the boundary

                    # sanity check using the other span correction doesn't cause overlap
                    crts_ = deepcopy(crts)
                    crt, crt_other = (crt1, crt2) if ce1 else (crt2, crt1)
                    crts_.remove(crt)
                    crt_out = ner_sample2corrected_sample(sample=ner_sample, corrections=crts_, allowed_entity_types=self.entity_types)
                    ovl = check.entities_overlapping(
                        sentence=sent, entity_names=crt_out.entity_names, ignore_case=True, search_in_order=False).overlap
                    spans_w_crts = [(crt.span, crt.correction)]
                    if not ovl:
                        # d_args = dict(original_span=crt.span, corrected_span=crt.correction)
                        enms, ets = crt_out.entity_names, crt_out.entity_types
                        d_log.update(entity_names_corrected_final=enms, entity_types_corrected_final=ets)
                    else:
                        # in this case, to resolve the overlap, need to drop both corrections
                        all_correction_dropped = len(crts) == 2
                        spans_w_crts += [(crt_other.span, crt_other.correction)]
                        if not all_correction_dropped:  # re-correct the sample again
                            crts_.remove(crt_other)
                            crt_out = ner_sample2corrected_sample(sample=ner_sample, corrections=crts_, allowed_entity_types=self.entity_types)
                            assert not check.entities_overlapping(
                                sentence=sent, entity_names=crt_out.entity_names, ignore_case=True, search_in_order=False).overlap
                            enms, ets = crt_out.entity_names, crt_out.entity_types
                            d_log.update(entity_names_corrected_final=enms, entity_types_corrected_final=ets)

                    # still some edge case: both span boundaries are larger, and including any would cause overlap, e.g.
                    #   X: `What's the highest rated movie of all time?`
                    #   Entities:
                    #       - highest rated => highest rated movie
                    #       - movie of all time => highest rated movie of all time
                msg = f'{msg}, span(s) causing overlap dropped'
                if all_correction_dropped:
                    msg = f'{msg}, modified as no correction'
                msg = f'{msg} w/ {sdpc(d_log)}'
                self.ec(msg=msg, kind='corrected-entity-span-overlap', args=dict(original_spans_n_corrections=spans_w_crts))
                if all_correction_dropped:  # this sample will not be corrected at all, i.e. same as the original sample
                    continue
            assert not check.entities_overlapping(sentence=sent, entity_names=enms, ignore_case=True, search_in_order=True).overlap

            ic = False if edit.entities_differ_in_case_only(entity_names=enms, sentence=sent, ec=self.ec) else True
            c = check.get_non_overlapping_keyword_counts(sentence=sent, keywords=enms, ignore_case=ic)
            assert all(c.values())  # sanity check no entity is missing

            # sanity check original entity order is preserved
            assert not edit.reorder_entities(sentence=sent, entity_names=enms, entity_types=ets, ignore_case=ic).reordered

            ner_sample = NerReadableExample.from_d(sentence=sent, entity_names=enms, entity_types=ets)
            samples_corrected.append((i_ner, ner_sample))
            for kd, entity_crts in crt_out.correction_maps.items():
                entities_corrected[kd] += entity_crts

        # override the original NER samples and log them
        samples_out = []
        d_log_crt, n_ds, n_dt, n_crt_ = dict(total=len(samples_corrected)), 0, 0, 0
        it_corrected_sample = iter(samples_corrected)
        curr_crt = next(it_corrected_sample, None)
        corrected_log = []
        for i, sample_ori in enumerate(ner_samples):
            idx_crt, sample_crt = curr_crt or (None, None)
            # TODO: refactor w/ `dataset_util.compare_samples`
            if i == idx_crt:  # this sample has been corrected
                assert sample_ori.sentence == sample_crt.sentence and sample_ori != sample_crt  # sanity check
                d = dict(X=sample_ori.sentence)
                d.update({'Y-original': atc(sample_ori), 'Y-corrected': atc(sample_crt)})

                # log the differences
                diff = dataset.analyze_diff(eg1=sample_ori, eg2=sample_crt, logger=self.logger, color=True)
                ds, dt = diff.span_is_different, diff.type_is_different
                assert (ds or dt)
                if ds:
                    n_ds += 1
                    d['diff-span'] = diff.different_span
                if dt:
                    n_dt += 1
                    d['diff-type'] = diff.different_type

                samples_out.append(sample_crt)
                curr_crt = next(it_corrected_sample, None)
                corrected_log.append(d)
                n_crt_ += 1
            else:
                samples_out.append(sample_ori)
        assert n_crt_ == len(samples_corrected)  # sanity check all corrected samples are processed
        d_log_crt.update({'#samples-w/-different-span': n_ds, '#samples-w/-different-type': n_dt})
        d_log_crt = {'#samples-total': len(ner_samples), '#samples-corrected': d_log_crt, 'samples-corrected': corrected_log}
        self.logger.info(f'Processed NER sample corrections w/ {pl.i(d_log_crt, indent=3)}')

        # also log the entity corrections as a rough examination of LLM correction quality
        corrected_entities_log = dict()
        for lb, entity_crts in entities_corrected.items():

            if lb == LABEL_NOT_NAMED_ENTITY:
                assert all(NA == STR_NOT_NAMED_ENTITY for (ori_span, NA) in entity_crts)
                entity_crts = [ori_span for (ori_span, NA) in entity_crts]
            c = Counter(entity_crts)
            if lb != LABEL_NOT_NAMED_ENTITY:
                c_ = dict()
                for k, v in c.most_common():  # sort by count
                    ori, crt = k
                    ori, crt = pl.i(ori, c='y'), pl.i(crt, c='g')  # colorize the changes
                    c_[f'{ori} => {crt}'] = v
            else:
                c_ = dict(c.most_common())
            corrected_entities_log[lb.to_str()] = c_
        self.logger.info(f'Processed entity corrections w/ {pl.i(corrected_entities_log, indent=2)}')

        n_cpl = sum(d_log_count['#completions'].values())
        n_extract = d_log_count['#correction-extracted']['total']
        d_log_count = {
            '#total-triples': d_log_count['#total-triples'], '#completions': n_cpl,
            '#triples-analyzed': d_log_count['#triples-analyzed'], '#correction-extracted': n_extract, '#samples-corrected': n_crt_}
        return dataset.finish_ner_processing(
            samples=samples_out, logger=self.logger, dataset_name=self.dataset_name, entity_types=self.entity_types,
            dedup=True, lowercase=lowercase, output_path=output_path, d_log=d_log_count, ec=self.ec
        )
