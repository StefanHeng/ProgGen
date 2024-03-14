"""
Provide reasoning for the demo-samples

Intended for chain-of-thought reasoning to boost annotation accuracy
"""

from typing import Union

from stefutil import *
from src.util.ner_example import *


__all__ = ['Sample2Reasoning']


sample2reason_dict = {
#     'conll2003-no-misc': {
#         'Mouscron 4 2 2 0 7 4 8': """Given the context, the series of numbers likely represent sports statistics, so "Mouscron" appears to be the name of a sports team. Thus, it's an organization.""",
#         'Saudi Arabia executes Pakistani man.': """"Saudi Arabia" is a known country, thus it's a location.
# "Pakistani man" is a generic reference to an individual based on nationality and not a specific person's name.""",
#         'BASEBALL-DODGERS WIN FIFTH STRAIGHT.': """Given the context, "BASEBALL" and "WIN FIFTH STRAIGHT" describe the record of a baseball team. "DODGERS" is also a known baseball team, which qualifies it as an organization.""",
#         'In a lengthy debate on Burundi before the U.N. Security Council, Ambassador Nsanze Terence said the new military government took over to stabilise the country and wanted negotiations under former Tanzanian President Julius Nyrere.': """“Burundi” appears after “on” and "Burundi" is a country, so it’s a location.
# "U.N. Security Council" is a collective body under the United Nations, so it's an organization.
# "Nsanze Terence" is mentioned with the title "Ambassador", signaling a person.
# "Julius Nyrere" is mentioned with the title "President", signaling a person.
# "Ambassador" and "President" are titles and not part of a commonly recognized name.""",
#         "Crohn's is an inflammation of the bowel that can sometimes require surgery.": """"Crohn's" suggests a medical condition named after an individual, making “Crohn” a person entity.""",
#         'A.de Silva not out 49':  """Given the context, "not out 49" seems to describe a player’s score/status (likely in cricket). Thus, "A.de Silva" is a person.""",
#         'Interacciones ups Mexico GDP forecast, lowers peso.': """"Interacciones" seems to be a company making an economic forecast, hence an organization.
# "Mexico" is a known country and thus a location.""",
#         'Oil product inventories held in independent tankage in the Amsterdam-Rotterdam-Antwerp area were at the following levels, with week-ago and year-ago levels, industry sources said.': """"Amsterdam-Rotterdam-Antwerp" appears between “in the” and “area”. It refers to a recognized region encompassing three cities in Europe. Hence, it's a location.""",
#         'At Colchester: Gloucestershire 280 and 27-4.': """Given the context, “280” and “27-4” seem to be scores for a sports event.
# "Colchester" appears after "at" and is a town. It’s likely the location where the event takes place.
# The sports statistics describe "Gloucestershire." It seems to be a sports team. Thus, it's an organization.""",
#         "+5 Mark O'Meara through 15": """+5" and "through 15" seem to refer to scores or statuses.
# "Mark O'Meara" is a name, likely of a sports player given the context, and thus is a person."""
#     }
    'conll2003-no-misc': {
        'Mouscron 4 2 2 0 7 4 8': [
            """The series of numbers likely represent sports statistics, so "Mouscron" appears to be the name of a sports team. Thus, it's an organization."""
        ],
        'Saudi Arabia executes Pakistani man.': [
            """"Saudi Arabia" is a general country name and hence a location.""",
            """"Pakistani man" refers to an individual, but it's not the name of a specific person, so it's not a person entity."""
        ],
        'BASEBALL - DODGERS WIN FIFTH STRAIGHT.': [
            """"WIN FIFTH STRAIGHT" likely describes the record of a baseball team, so "DODGERS" should be the name of a sports team. Thus, it's an organization.""",
            """"DODGERS" is a known baseball team name, which qualifies it as an organization entity."""
        ],
        'In a lengthy debate on Burundi before the U.N. Security Council, Ambassador Nsanze Terence said the new military government took over to stabilise the country and wanted negotiations under former Tanzanian President Julius Nyrere.': [
            """"Burundi" appears after "on", thus it's likely the name of a location.""",
            """"Burundi" is a known country name, so it's a location.""",
            """"U.N. Security Council" is the name of a collective body under the United Nations, so it's an organization entity.""",
            """"Nsanze Terence" is the full name of an individual, thus a person entity.""",
            """"Nsanze Terence" is mentioned with the title "Ambassador", signaling a person entity.""",
            """"Julius Nyrere" is the full name of an individual, thus a person entity.""",
            """"Julius Nyrere" is mentioned with the title "President", signaling a person entity.""",
            """Given the context, "Ambassador" and "President" are titles and not part of a commonly recognized name, so they are not included in the person entities."""
        ],
        "Crohn's is an inflammation of the bowel that can sometimes require surgery.": [
            """"Crohn's" suggests a medical condition named after an individual, making "Crohn" the name of an individual, thus a person entity."""
        ],
        'A.de Silva not out 49': [
            # """"A.de Silva" appears to be an individual's name, hence it's categorized as a person.""",
            """"A.de Silva" is the full name of a specific individual, hence it's categorized as a named person entity."""
            """"not out 49" seems to describe a player's score/status in a sports game. Thus, "A.de Silva" is the name of a player, thus a person entity."""
        ],
        'Interacciones ups Mexico GDP forecast, lowers peso.': [
            """"Interacciones" appears to be the name of a company making an economic forecast, hence an organization.""",
            """"Mexico" is a known country name and hence a location."""
        ],
        'Oil product inventories held in independent tankage in the Amsterdam-Rotterdam-Antwerp area were at the following levels, with week-ago and year-ago levels, industry sources said.': [
            """"Amsterdam-Rotterdam-Antwerp" appears between "in the" and "area", so it's likely the name of a location.""",
            """"Amsterdam-Rotterdam-Antwerp" refers to a recognized region encompassing three cities in Europe. Hence, it's a named location entity."""
        ],
        'At Colchester: Gloucestershire 280 and 27-4.': [
            """"Colchester" appears after "at", so it's likely the name of a location.""",
            """"Colchester" is the name of a town. It's likely the location where the event takes place, making it a location entity.""",
            """Given the context, "280" and "27-4" seem to be the scores for a sports event, so "Gloucestershire" seems to be the name of a sports team. Thus, it's a named organization entity."""
        ],
        "+5 Mark O'Meara through 15": [
            """"Mark O'Meara" is an individual's name and is categorized as a person.""",
            """Given the context, "+5" and "through 15" look like a score update for a player. Thus, "Mark O'Meara" is the name of a person."""
        ],
        "With 2.5 percent gross domestic product growth expected for 1996, new job growth should slowly lower the unemployment rate over the rest of the year.": [
            """"2.5 percent" is a percentage. "1996" is a year. The sentence does not mention specific persons, geographical locations, or named organizations."""
        ]
    }
}


class Sample2Reasoning:
    def __init__(self, dataset_name: str = 'conll2003-no-misc'):
        ca(dataset_name=dataset_name)
        if dataset_name != 'conll2003-no-misc':
            raise NotImplementedError
        self.dataset_name = dataset_name
        self.d = sample2reason_dict[dataset_name].copy()

    def __call__(self, sample: Union[NerReadableExample, str]) -> str:
        if self.dataset_name == 'conll2003-no-misc':
            sent = sample if isinstance(sample, str) else sample.sentence
            assert sent in self.d
            ret = self.d[sent]
            ret = [f'- {row}' for row in ret]  # reformat as markdown bullet points
            ret = '\n'.join(ret)
            return f'Reasoning:\n{ret}'
        else:
            raise NotImplementedError


if __name__ == '__main__':
    s2r = Sample2Reasoning(dataset_name='conll2003-no-misc')
    reason = s2r('At Colchester: Gloucestershire 280 and 27-4.')
    print(reason)


