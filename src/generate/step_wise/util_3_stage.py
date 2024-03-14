"""
For each NER demo sample, convert to some binary type classifications demos
    Also, get some manually selected spans as negative samples

Each (sentence, span) sample has corresponding manually-written chain-of-thought reasoning
    For both positive and negative samples

Intended for 3-stage generation to boost annotation accuracy
"""

import random
import re
from typing import List, Union
from dataclasses import dataclass

from stefutil import *
from src.util import *
from src.util.ner_example import *
from src.data_util import *


__all__ = [
    'MANUAL_DEMO_SAMPLES',
    'TypeClsTuplesOutput',
    'NOT_ENTITY_TYPE', 'OTHER_ENTITY_TYPE', 'ANY_ENTITY_TYPE',
    'Type2GenType', 'Sentence2NegativeSpans', 'Sentence2SpansReasoning', 'Sample2TypeClsTuples'
]


# type for randomly sampling an entity class for the negative samples
TYPE_RANDOM_SAMPLE = '__random-sample__'
# use multi-class entity type labels instead
NOT_ENTITY_TYPE = '__not-entity__'
OTHER_ENTITY_TYPE = '__other-entity__'
ANY_ENTITY_TYPE = '__any-entity__'  # placeholder for chain-of-thought reasoning indexing


# Add additional manually-written demo samples representative of wrong CLS labels in Step 3
MANUAL_DEMO_SAMPLES = {
    'mit-movie': [
        NerReadableExample.from_d(
            # sentence="Show me viewers' ratings and plot summaries for good funny movies from female filmmakers.",
            sentence="show me viewers' rating and plot summary for a good funny movie from a female filmmaker.",
            entity_names=['good', 'funny'],
            entity_types=["Viewers' Rating", 'Genre']
        )
    ]
}

# for each sentence, each negative sample is a span; for step 2: span generation in 3-stage gen
sentence2negative_spans = {
    'conll2003-no-misc': {
        'Saudi Arabia executes Pakistani man.': ['Pakistani', 'Pakistani man'],
        'BASEBALL - DODGERS WIN FIFTH STRAIGHT.': [],
        'A.de Silva not out 49': [],
        "With 2.5 percent gross domestic product growth expected for 1996, new job growth should slowly lower the unemployment rate over the rest of the year.":
            # ['2.5 percent', '1996'],  # too far from the 3 relevant entity types, so not used
            []
    },
    'mit-movie': {
        'what type of movie genre is the perfect weapon': [],
        'how many movies came out in 2004': [],
        'could you show me some part of the new indiana jones movie': [],
        'show me a movie with the song a whole new world': [],
        'what movie is considered the funniest of all time': [],
        'when did mark joffe direct the bounty hunter film that is rated pg': ['direct'],
        'is cary grant in any historical films that are a must see': ['historical film']
    }
}


NO_NEG = {b: [] for b in [True, False]}
# for each sentence, each negative sample is a (span, type) tuple
sample2negative_pairs = {
    'conll2003-no-misc': {
        'Saudi Arabia executes Pakistani man.': {
            False: [('Pakistani', 'location'), ('Pakistani man', 'person')],  # for binary cls
            True: [('Pakistani', NOT_ENTITY_TYPE), ('Pakistani man', NOT_ENTITY_TYPE)]  # for multi-class cls
        },
        'BASEBALL - DODGERS WIN FIFTH STRAIGHT.': NO_NEG,
        'A.de Silva not out 49': NO_NEG,
        "With 2.5 percent gross domestic product growth expected for 1996, new job growth should slowly lower the unemployment rate over the rest of the year.": {
            False: [('2.5 percent', TYPE_RANDOM_SAMPLE), ('1996', TYPE_RANDOM_SAMPLE)],
            True: [('2.5 percent', OTHER_ENTITY_TYPE), ('1996', OTHER_ENTITY_TYPE)]
        }
    },
    'mit-movie': {
        'what type of movie genre is the perfect weapon': NO_NEG,
        'how many movies came out in 2004': NO_NEG,
        'could you show me some part of the new indiana jones movie': NO_NEG,
        'show me a movie with the song a whole new world': NO_NEG,
        'what movie is considered the funniest of all time': NO_NEG,
        'when did mark joffe direct the bounty hunter film that is rated pg': {
            False: [('direct', TYPE_RANDOM_SAMPLE)],
            True: [('direct', OTHER_ENTITY_TYPE)]
        },
        'is cary grant in any historical films that are a must see': {
            False: [('historical film', TYPE_RANDOM_SAMPLE)],
            True: [('historical film', NOT_ENTITY_TYPE)]
        },
        "show me viewers' rating and plot summary for a good funny movie from a female filmmaker.": {
            False: [
                ("viewers' rating", TYPE_RANDOM_SAMPLE), ('plot summary', TYPE_RANDOM_SAMPLE),
                ('female', TYPE_RANDOM_SAMPLE), ('female director', TYPE_RANDOM_SAMPLE)
            ],
            True: [
                ("viewers' rating", NOT_ENTITY_TYPE), ('plot summary', NOT_ENTITY_TYPE),
                ('female', NOT_ENTITY_TYPE), ('female filmmaker', NOT_ENTITY_TYPE)
            ]
        }
    }
}


# for each sentence in a sample, the reasoning for the list of potential spans
# Intended for step 2: generate spans
#   can contain negative spans, will be filtered out in step 3
# an initial written version, pretty casual, but seems to work
sentence2spans_reasoning = {
    'mit-movie': {
        'what type of movie genre is the perfect weapon':
            '"the perfect weapon" is a movie name.',
        'how many movies came out in 2004':
            # '"2004" is a time period.',
            '"2004" refers to a time period.',
        'could you show me some part of the new indiana jones movie':
            '"some part" indicates a trailer. "indiana jones" refers to a person.',
        'show me a movie with the song a whole new world':
            '"a whole new world" is a song name.',
        'what movie is considered the funniest of all time':
            '"funniest of all time" describes a movie.',
        'when did mark joffe direct the bounty hunter film that is rated pg':
            # '"mark joffe" is a person\'s name. "bounty hunter" describes a film. "pg" is a rating.',
            # '"mark joffe" is a person\'s name. "bounty hunter" describes a film. "pg" is a movie assessment.',
            '"mark joffe" is a person\'s name. "bounty hunter" describes a film element. "pg" is a movie rating.',
        'is cary grant in any historical films that are a must see':
            # '"cary grant" is the name of a person. "historical" and "must see" describes films.',
            # '"cary grant" is the name of a person. "historical" and "must see" describe movies/films.',
            # '"cary grant" is the name of a person. "historical" describe movies. "must see" describe films.',
            # '"cary grant" is the name of a person. "historical" is a genre. "must see" describes film.',
            # '"cary grant" is the name of a person. "historical" is a genre. "must see" describes a film.',
            # '"cary grant" is the name of a person. "historical" is a genre that don\'t contain a trailing "movie". "must see" describes a film.',
            '"cary grant" is the name of a person. "historical" defines a genre. "must see" describes a film.',
    }
}

# the reasoning for each (sentence, span in the sentence, potential type label)
# Intended for step 3: generate type labels
sample_triple2cls_reasoning = {
    'conll2003-no-misc': {
        ('Saudi Arabia executes Pakistani man.', 'Saudi Arabia', 'location'):
            # '"Saudi Arabia" is a general country name. The name of a country is a location entity.',
            '"Saudi Arabia" is the name of a specific country. The name of a country is a location entity.',
        ('Saudi Arabia executes Pakistani man.', 'Pakistani', 'location'):
            # '"Pakistan" refers to a specific country. In contrast, "Pakistani" is a adjective describing "man", so it doesn\'t refer to a specific location.',
            # '"Pakistani" is a adjective describing where the "man" is from. The span "Pakistani" in itself is a nationality, not a country name, so it\'s not a named entity.',
            '"Pakistani" is a adjective describing where the "man" is from. The span "Pakistani" in itself is a nationality, not a country name. A adjective describing a nationality is not a named entity.',
        ('Saudi Arabia executes Pakistani man.', 'Pakistani man', 'person'):
            # '"Pakistani man" refers to a specific person, but it is not the name of a person, so it\'s not a named person entity.',
            # '"Pakistani man" refers to someone. However, the span "Pakistani man" is not the individual\'s name, so it\'s not a named entity.',
            # '"Pakistani man" refers to someone but does not provide the individual\'s name, so it\'s not a named entity.',
            '"Pakistani man" refers to someone but does not provide the individual\'s name. A reference to a person not by the name is not a named entity.',

        ('BASEBALL - DODGERS WIN FIFTH STRAIGHT.', 'DODGERS', 'organization'):
            # '"WIN FIFTH STRAIGHT" likely describes the record of a baseball team, so "DODGERS" should be the name of a sports team. '
            # 'Additionally, "DODGERS" is a known baseball team name. The name of a sports team is an organization entity.',
            '"WIN FIFTH STRAIGHT" describes the record of a baseball team, so "DODGERS" refers to a sports team by its name. '
            # 'Additionally, "DODGERS" is a known baseball team name. The name of a sports team is an organization entity.',
            'Additionally, "DODGERS" is a known baseball team name. A sports team is an organization.',

        ('A.de Silva not out 49', 'A.de Silva', 'person'):
            # '"A.de Silva" appears to be an individual\'s name. '
            # 'Additionally, "not out 49" seems to describe a player\'s score/status in a sports game. Thus, "A.de Silva" is the name of a player. '
            # 'The name of a person is a person entity.',
            # '"de Silva" is a common last name, "A." is a first name initial, so "A.de Silva" should be an individual\'s name. '
            '"de Silva" is a first name initial followed by a common last name, so "A.de Silva" should be an individual\'s name. '
            'Additionally, "not out 49" describes a player\'s score in a sports game. Based on context, "A.de Silva" is the name of a player. '
            'A player is a person. A specific person\'s name is a person entity.',
        ('A.de Silva not out 49', '49', 'other'): """"49" is a number, so it's an other entity.""",

        ('With 2.5 percent gross domestic product growth expected for 1996, new job growth should slowly lower the unemployment rate over the rest of the year.',
         '2.5 percent', ANY_ENTITY_TYPE):
            # '"2.5 percent" is a percentage.',
            # '"2.5 percent" is a percentage. A named percentage entity is not a named entity of types location, person, or organization.',
            # '"2.5 percent" is a percentage. A percentage entity doesn\'t fall under the named entity types location, person, or organization.',
            # '"2.5 percent" is a percentage. A percentage is not a person, location, or organization.',
            '"2.5 percent" is a percentage and a named entity. However, a percentage is not a person, location, or organization.',
        ('With 2.5 percent gross domestic product growth expected for 1996, new job growth should slowly lower the unemployment rate over the rest of the year.',
         '1996', ANY_ENTITY_TYPE):
            # '"1996" is a year.'
            # '"1996" is a year. A named year entity is not a named entity of types location, person, or organization.',
            # '"1996" is a year. A year entity doesn\'t fall under the named entity types location, person, or organization.',
            '"1996" is a year and a named entity. However, a year is not a person, location, or organization.',
    },


    'mit-movie': {
        ('what type of movie genre is the perfect weapon', 'the perfect weapon', 'Title'):
            # 'Based on context, "the perfect weapon" refers to a movie name. "The perfect weapon" is a specific title of a movie.',
            # '"the perfect weapon" refers to a movie name. Thus, "The perfect weapon" is a specific title of a movie.',
            # '"the perfect weapon" is a concrete movie name. That is, "The perfect weapon" is a specific title of a movie.',
            '"the perfect weapon" is a concrete movie name.',

        ('how many movies came out in 2004', '2004', 'Year'):
            # '"2004" is a year. It specifies a particular time duration when the movies were released.',
            # '"2004" is a year. It specifies a particular time duration when the movies were released. A time duration of movie release counts as a Year entity.',
            '"2004" is a year. It specifies a particular year when the movies were released. A year or years of movie release counts as a Year entity.',

        ('could you show me some part of the new indiana jones movie', 'some part', 'Trailer'):
            # '"some part" indicates showing a segment of a movie. Referring to a segment or clip from a movie counts as a Trailer entity.',
            # '"some part" indicates showing a segment of a movie. General reference to a segment or clip from a movie counts as a Trailer entity.',
            # '"some part" refers to a segment of a movie. General request to a segment or clip from a movie counts as a Trailer entity.',
            '"some part" indicates a segment of a movie. General request to a segment or clip from a movie counts as a Trailer entity.',
        ('could you show me some part of the new indiana jones movie', 'indiana jones', 'Character'):
            # '"Indiana Jones" is a well-known character. He\'s the protagonist in the "Indiana Jones" franchise first appearing in the film "Raiders of the Lost Ark".',
            '"Indiana Jones" is a well-known character. He\'s the protagonist in the "Indiana Jones" franchise',

        ('show me a movie with the song a whole new world', 'a whole new world', 'Song'):
            # 'Based on context, "A whole new world" is the name of a song. A song featured in a movie counts as a Song entity.',
            # '"A whole new world" is the name of a song. A specific song name featured in a movie is a Song entity.',
            '"A whole new world" is the name of a song. A song name featured in a movie is a named Song entity.',

        ('what movie is considered the funniest of all time', 'funniest of all time', 'Review'):
            # '"funniest of all time" is a detailed critique or opinion about a movie. A specific and detailed comment from professional critics or the general audience for a movie counts as a Review entity.',
            # '"funniest of all time" is a detailed critique or opinion about a movie. A detailed movie comment from viewers or critics is a Review entity.',
            '"funniest of all time" is a detailed critique or opinion about a movie. A detailed movie comment is a Review entity.',

        ('when did mark joffe direct the bounty hunter film that is rated pg', 'mark joffe', 'Director'):
            # '"Mark Joffe" is a person\'s name. Based on context, "Mark Joffe" directed a film. Thus, the span "Mark Joffe" is the name of a director.',
            # '"Mark Joffe" is a person\'s name. Based on context, "Mark Joffe" directed a film. Thus, the span "Mark Joffe" is the name of a director. A directory\'s name is a Director entity.',
            # '"Mark Joffe" is a person\'s name. Based on context, "Mark Joffe" directed a film. Thus, the span "Mark Joffe" is a director\'s name. Only a director\'s name is a Director entity.',
            # '"Mark Joffe" is a person\'s name and he directed a film. Thus, "Mark Joffe" is a director\'s name. A specific director\'s name is a Director entity.',
            # '"Mark Joffe" directed a film and is a person\'s name. Thus, "Mark Joffe" is a director\'s name. A specific director\'s name is a Director entity.',
            '"Mark Joffe" directed a film and is a person\'s name. Thus, "Mark Joffe" is a director\'s name. A director\'s full name is a Director entity.',
        ('when did mark joffe direct the bounty hunter film that is rated pg', 'direct', ANY_ENTITY_TYPE):
            # '"direct" is a verb describing the action of the director. It is not a named entity.',
            # '"direct" is a verb describing the action of the director. It is the action of a director.',
            '"direct" is a verb describing the action of the director.',
        ('when did mark joffe direct the bounty hunter film that is rated pg', 'bounty hunter', 'Plot'):
            # 'Based on context, "the bounty hunter film" indicates "bounty hunter" is not a movie name. Instead, "bounty hunter" describes a plot element for movies. A movie theme or storyline counts as a Plot entity.',
            # 'Based on context, "the bounty hunter film" indicates "bounty hunter" is not a movie name. Instead, "bounty hunter" is a specific plot element for movies. A movie theme or storyline counts as a Plot entity.',
            # 'The span "the bounty hunter film" indicates "bounty hunter" is not a movie name. Thus, "bounty hunter" is a specific plot element for movies. A specific movie theme or storyline counts as a Plot entity.',
            # '"bounty hunter" is a specific plot element for movies. A named movie theme or storyline counts as a named Plot entity.',
            '"bounty hunter" is a particular plot element for movies. A named movie theme or storyline counts as a named Plot entity.',
            # '"bounty hunter" is a concrete element in movie plots. A named movie theme or storyline counts as a named Plot entity.',
        ('when did mark joffe direct the bounty hunter film that is rated pg', 'pg', 'MPAA Rating'):
            # '"pg" stands for  "Parental Guidance Suggested." This rating is a part of the Motion Picture Association of America (MPAA) film rating system.',
            '"pg" stands for "Parental Guidance Suggested" and is a rating in the MPAA film rating system.',

        ('is cary grant in any historical films that are a must see', 'cary grant', 'Actor'):
            # '"Cary Grant" is a person\'s name. By context, "Cary Grant" is in films, making the span an actor’s name.',
            # '"Cary Grant" is a person\'s name. By context, "Cary Grant" appears in films, making the span an actor\'s name.',
            # '"Cary Grant" is a person\'s name and he is in films. Thus, "Cary Grant" is an actor\'s name. A specific actor\'s name is a named Actor entity.',
            '"Cary Grant" appears in films so he is an actor. "Cary Grant" is also a person\'s name. Thus, "Cary Grant" is an actor\'s name. A specific actor\'s name is a named Actor entity.',
        ('is cary grant in any historical films that are a must see', 'historical', 'Genre'):
            # 'Based on context, "historical" refers to the category or style of film.',
            # '"historical" refers to the category or style of film. A specific movie category or style is a Genre entity.',
            '"historical" is a specific category or style of film. A specific movie category or style is a Genre entity.',
        ('is cary grant in any historical films that are a must see', 'historical film', ANY_ENTITY_TYPE):
            '"historical" is suffice to serve as a Genre keyword. The addition of film is redundant and the entire span "historical film" is not a named entity.',
        ('is cary grant in any historical films that are a must see', 'must see', 'Viewers\' Rating'):
            # 'Based on context, "must see" is a general assessment of a movie by its audience. It indicates popularity or recommendation level.',
            # 'Based on context, "must see" is a specific assessment of a movie by its audience. It indicates general popularity or recommendation level.',
            # '"must see" is a specific assessment of a movie by its audience. It indicates general popularity or recommendation level. A specific movie assessment is a Viewers\' Rating entity.',
            # '"must see" is a specific assessment of a movie by its audience. It indicates broad popularity or recommendation level. A specific movie assessment is a Viewers\' Rating entity.',
            # '"must see" indicates a specific level of popularity or recommendation. "must see" is a specific assessment of a movie by its audience. A specific movie assessment is a Viewers\' Rating entity.',
            # '"must see" indicates a specific level of popularity or recommendation. The span "must see" is a particular viewers\' assessment of a movie. A concrete instance of a viewer assessment is a Viewers\' Rating entity.',
            '"must see" is a quick assessment of a movie by its audience. A concise movie assessment counts as a Viewers\' Rating entity.',

        ("show me viewers' rating and plot summary for a good funny movie from a female filmmaker.", "viewers' rating", 'Viewers\' Rating'):
            # 'The span "viewers\' rating" is part of the request. It is not a specific instance of a rating or a movie assessment. A viewers\' rating entity has to be a specific rating or assessment.',
            # 'The span "viewers\' rating" merely request information. It does not include a specific rating or a movie assessment.',
            # 'The span "viewers\' rating" does not include a specific rating or a movie assessment. It merely requests information.',
            # '"viewers\' rating" is not an actual movie assessment from viewers. "viewers\' rating" merely requests rating information. A flexible request for a movie assessment doesn\'t count as a Viewers\' Rating entity.',
            # 'The span "viewers\' rating" in itself is not a concrete movie assessment from viewers. "viewers\' rating" merely requests rating information. A flexible request for a movie assessment doesn\'t count as a Viewers\' Rating entity.',
            # '"viewers\' rating" is not an actual movie rating or recommendation level. "viewers\' rating" merely requests rating information. A flexible request for a rating is not a named Viewers\' Rating entity.',
            # '"viewers\' rating" does not contain an actual movie rating or recommendation level. "viewers\' rating" merely requests rating information. A flexible request for a rating is not a named Viewers\' Rating entity.',
            '"viewers\' rating" does not contain an actual movie rating or recommendation level.',
            # '"viewers\' rating" simply mentions ratings. It doesn\'t include an actual movie rating or recommendation level. A general mention of rating is not a named Viewers\' Rating entity.',
            # '"viewers\' rating" is not an actual movie rating or recommendation level. "viewers\' rating" simply mentions ratings. A general mention of rating is not a named Viewers\' Rating entity.',
        ("show me viewers' rating and plot summary for a good funny movie from a female filmmaker.", "plot summary", 'Plot'):
            # 'The span "plot summary" is a requested category of information. It is not a specific theme or element related to a movie.',
            # 'The span "plot summary" is a requested category of information. It does not refer to a specific movie theme or element. A general request for a movie storyline is not a Plot entity.',
            # 'The span "plot summary" does not refer to a specific movie theme or element. "plot summary" simply requests storyline information. A general request for a movie storyline is not a Plot entity.',
            # '"plot summary" simply requests storyline information. "plot summary" is not an actual movie theme or plot element. A flexible request for a movie plot doesn\'t count as a Plot entity.',
            # '"plot summary" simply requests storyline information. "plot summary" is not an actual movie theme or plot element. ',
            '"plot summary" simply refers to plot. "plot summary" is not an actual movie theme or plot element.',
            # '"plot summary" simply refers to plot. "plot summary" is not an actual movie theme.',
            # '"plot summary" is a flexible request for storyline information. "plot summary" doesn\'t contain actual movie themes or plot elements. A general mention of plot is not a named Plot entity.',
            # '"plot summary" is simply mentions storyline. "plot summary" doesn\'t contain specific named movie themes or plot elements. A general mention of plot is not a named Plot entity.',
        ("show me viewers' rating and plot summary for a good funny movie from a female filmmaker.", "good", "Viewers\' Rating"):
            '"good" is an adjective that describes "movie". It implies a high quality movie perceived by viewers.',
        ("show me viewers' rating and plot summary for a good funny movie from a female filmmaker.", "funny", "Genre"):
            '"movie" is right after "funny", so the span "funny" directly specifies a movie characteristic. It implies a movie in the comedy genre.',
        ("show me viewers' rating and plot summary for a good funny movie from a female filmmaker.", "female", "Director"):
            '"female" is a adjective describing a characteristic of "filmmaker". It is a preference, not a female filmmaker\'s  name.',
        ("show me viewers' rating and plot summary for a good funny movie from a female filmmaker.", "female filmmaker", "Director"):
            '"female filmmaker" is a category of directors. It is not the name of a specific director.'
    }
}


@dataclass
class TupleGroupOutput:
    entity_name: str = None
    entity_type: str = None
    is_typed_entity: bool = None
    reason: str = None


@dataclass
class TypeClsTuplesOutput:
    sentence: str
    tuples: List[TupleGroupOutput]


class Type2GenType:
    """
    Convert the internal entity type to the entity type used for generation
        Used for multi-class entity type classification
    """
    # Incorrect label formatting from LLM, instead of `a named person entity`, e.g.
    #   In CoNLL-2003
    #       `a person entity`, `named person entity`, `named entity of other type`
    #   In MIT-Movie
    #       `specific MPAA Rating entity`, `not a named entity related to movies`, `named Actor entities`
    pattern_wrong_generated_entity_type = re.compile(r'^((a|an) )?(named |specific )?(?P<entity_type>.+) (entity|entities)$')

    # Incorrect label formatting from LLM for `not a named entity`

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.entity_types = sconfig(f'datasets.{self.dataset_name}.readable-entity-types').copy()

        self.entity_types_all = self.entity_types + [OTHER_ENTITY_TYPE, NOT_ENTITY_TYPE]

        self.t2gt = None
        self.t2gt = {et: self.__call__(entity_type=et, output_kind='demo') for et in self.entity_types_all}
        gts = list(self.t2gt.values())
        assert len(gts) == len(set(gts))  # sanity check all values are unique

        self.gt2t = {v: k for k, v in self.t2gt.items()}

    @property
    def gen_entity_types(self):
        return [self.__call__(entity_type=et, output_kind='name') for et in (self.entity_types + [OTHER_ENTITY_TYPE, NOT_ENTITY_TYPE])]

    def __call__(self, entity_type: str, output_kind: str = 'name'):
        """
        :param entity_type: Internal entity type to map, e.g. `location`
        :param output_kind: One of [`name`, `demo`]
            If `name`, used for entity types in task instruction, e.g. `named location entity`
            If `demo`: used in demos, e.g. `a named location entity`
        :return:
        """
        if output_kind == 'demo' and self.t2gt is not None:
            return self.t2gt[entity_type]

        if entity_type in self.entity_types:
            # return f'is named entity and class is {et}'  # too long, and no sign that it improves CLS acc
            if output_kind == 'name':
                return f'named {entity_type} entity'
            else:
                return f'a named {entity_type} entity'
                # return f'named {entity_type} entity'  # try using the same one
        elif entity_type == OTHER_ENTITY_TYPE:
            # return 'is named entity and class is other'
            if output_kind == 'name':
                return f'named entity of other type'
            else:
                return f'a named entity of other type'
                # return f'named entity of other type'
        else:
            assert entity_type == NOT_ENTITY_TYPE
            return f'not a named entity'

    def decode(self, generated_entity_type: str, ec: prettier.EdgeCases = None, generated_sample: str = None):
        """
        Convert the generated entity type back to the internal entity type
        """
        if generated_entity_type in self.gt2t:
            return self.gt2t[generated_entity_type]
        else:
            # LLM generation didn't follow the expected format
            if generated_entity_type in ['named entity of other type']:
                entity_type = OTHER_ENTITY_TYPE
            elif generated_entity_type in [
                'not a specific named entity', 'not a specific entity type', 'not a named entity related to movies',
                'not a named entity in the context of the query', 'not specify a named entity', 'not refer to a specific entity',
                'not a specific movie title or a named entity', 'not a movie title or a named entity'
            ]:
                entity_type = NOT_ENTITY_TYPE
            elif generated_entity_type == 'a named entity':
                if self.dataset_name == 'mit-movie' and generated_sample.endswith("doesn't count as a named entity.\n"):
                    entity_type = NOT_ENTITY_TYPE
                else:
                    d_log = dict(generated=generated_entity_type, generated_sample=generated_sample)
                    raise ValueError(pl.fmt(d_log))
            else:
                m = self.pattern_wrong_generated_entity_type.match(generated_entity_type)
                if m is None:
                    sic(self.pattern_wrong_generated_entity_type, generated_entity_type)
                assert m is not None
                entity_type = m.group('entity_type')

                if entity_type not in self.entity_types:
                    if self.dataset_name == 'mit-movie' and entity_type in ['Award', 'Singer', 'Time Period', 'Reviewer']:
                        # if ec:
                        #     d_log = dict(span=entity_type, entity_type=OTHER_ENTITY_TYPE)
                        #     msg = f'LLM generated label not in allowed entity types w/ {pl.i(d_log)}'
                        #     ec(msg=msg, kind='wrong-label-format', args=d_log)
                        entity_type = OTHER_ENTITY_TYPE
                    else:
                        d_log = dict(
                            generated=generated_entity_type, extracted=entity_type, allowed_types=self.entity_types,
                            generated_sample=generated_sample)
                        raise ValueError(pl.fmt(d_log))
                # if entity_type not in self.entity_types:
                #     sic(entity_type, self.entity_types)
                # assert entity_type in self.entity_types
            if ec:
                d_log = dict(span=generated_entity_type, entity_type=entity_type)
                msg = f'LLM generated label not in expected format w/ {pl.i(d_log)}'
                ec(msg=msg, kind='wrong-label-format', args=d_log)
            return entity_type


class Sentence2NegativeSpans:
    def __init__(self, dataset_name: str = 'conll2003-no-misc'):
        ca(dataset_name=dataset_name)
        if dataset_name not in ['conll2003-no-misc', 'mit-movie']:
            raise NotImplementedError
        self.dataset_name = dataset_name
        self.d = sentence2negative_spans[dataset_name].copy()

        for sent, spans in self.d.items():  # sanity check negative spans appear in the sentence
            for span in spans:
                assert sent.count(span) == 1

    def __call__(self, sentence: Union[str, NerReadableExample]) -> List[str]:
        if isinstance(sentence, NerReadableExample):
            sentence = sentence.sentence
        assert isinstance(sentence, str)
        return self.d[sentence]


class Sentence2SpansReasoning:
    def __init__(self, dataset_name: str = 'mit-movie'):
        ca(dataset_name=dataset_name)
        if dataset_name not in ['mit-movie']:
            raise NotImplementedError
        self.dataset_name = dataset_name
        self.d = sentence2spans_reasoning[dataset_name].copy()

    def __call__(self, sentence: Union[str, NerReadableExample]) -> str:
        if isinstance(sentence, NerReadableExample):
            sentence = sentence.sentence
        assert isinstance(sentence, str)
        return self.d[sentence]


class Sample2TypeClsTuples:
    def __init__(self, dataset_name: str = 'conll2003-no-misc', multi_class_classify: bool = False, cot: bool = False):
        ca(dataset_name=dataset_name)
        if dataset_name not in ['conll2003-no-misc', 'mit-movie']:
            raise NotImplementedError
        self.dataset_name = dataset_name
        self.d = sample2negative_pairs[dataset_name].copy()
        self.ets = sconfig(f'datasets.{self.dataset_name}.readable-entity-types').copy()

        self.multi_class_classify = multi_class_classify

        self.cot = cot
        self.d_reason = sample_triple2cls_reasoning[self.dataset_name]

    def __call__(self, sample: NerReadableExample) -> TypeClsTuplesOutput:
        # if self.dataset_name == 'conll2003-no-misc':
        sent = sample.sentence
        tuples = []

        # get each for the sentence give, get all the positive samples followed by all the negative samples
        it = [(en, et, True) for en, et in zip(sample.entity_names, sample.entity_types)]
        assert sent in self.d
        enms_neg = self.d[sent][bool(self.multi_class_classify)]
        it += [(en, et, False) for en, et in enms_neg]

        for en, et, is_pos in it:  # sanity check: for these demo samples, each span appears once in the sentence
            assert sent.count(en) == 1
        # sort the spans by their appearance in the sentence
        it = sorted(it, key=lambda x: sent.index(x[0]))

        # for en, et in zip(sample.entity_names, sample.entity_types):
        for en, et, is_pos in it:
            if is_pos:  # positive sample
                reason = None
                if self.cot:
                    reason = sample_triple2cls_reasoning[self.dataset_name][(sent, en, et)]
                tuples.append(TupleGroupOutput(entity_name=en, entity_type=et, is_typed_entity=True, reason=reason))
            else:  # negative sample
                # for (en, et) in enms_neg:
                if not self.multi_class_classify:
                    if et == TYPE_RANDOM_SAMPLE:
                        et = random.choice(self.ets)
                reason = None
                if self.cot:
                    if not self.multi_class_classify:
                        key, key_fallback = (sent, en, et), (sent, en, ANY_ENTITY_TYPE)
                        reason = self.d_reason.get(key, self.d_reason.get(key_fallback))
                    else:
                        # search for all key matches for the (sentence, span) pair
                        keys = [k for k in self.d_reason.keys() if k[:2] == (sent, en)]
                        assert len(keys) == 1  # there should always be exactly one match
                        key = keys[0]
                        reason = self.d_reason[key]

                    assert reason is not None  # there must be a cot reasoning available
                tuples.append(TupleGroupOutput(entity_name=en, entity_type=et, is_typed_entity=False, reason=reason))
        return TypeClsTuplesOutput(sentence=sent, tuples=tuples)
        # else:
        #     raise NotImplementedError


if __name__ == '__main__':
    sic.output_width = 256

    def check_call():
        # cot_ = False
        cot_ = True
        s2t = Sample2TypeClsTuples(dataset_name='conll2003-no-misc', cot=cot_)
        sample = NerReadableExample(
            sentence='Saudi Arabia executes Pakistani man.',
            entity_names=('Saudi Arabia',),
            entity_types=('location',)
        )
        out = s2t(sample=sample)
        sic(out)
    check_call()
