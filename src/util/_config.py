"""
Writes a project-level config file
"""

import json
from os.path import join as os_join
from typing import Dict, Any

from src.util._paths import BASE_PATH, PROJ_DIR, PKG_NM
from src.util.ner_example import ner_labels2tags


config_dict = {
    'datasets': {
        'conll2003': {
            'x-name': 'Sentence',  # name for prompt prefix (e.g. `Sentence: `)
            'x-name-pl': 'Sentences',  # plural form of `x-name`
            'x-decoded-name': ['Sentence', 'Sentences'],  # prefix output from LLM completions, contains edge cases
            'y-name': 'Named Entities',
            'y-decoded-name': ['Entity Names', 'Entities', 'Named Entities', 'Named Entity'],
            'entity-types': ['PER', 'LOC', 'ORG', 'MISC'],
            'readable-entity-types': ['person', 'location', 'organization', 'miscellaneous']
        },
        'job-desc': {
            'x-name': 'Sentence',
            'x-name-pl': 'Sentences',
            'x-decoded-name': ['Sentence'],
            'y-name': 'Named Entities',
            'y-decoded-name': ['Named Entities'],
            'entity-types': ['Skill', 'Qualification', 'Experience', 'Domain', 'Occupation'],
        },
        'job-stack': {
            'x-name': 'Sentence',
            'x-name-pl': 'Sentences',
            'x-decoded-name': ['Sentence'],
            'y-name': 'Named Entities',
            'y-decoded-name': ['Named Entities'],
            'entity-types': ['Organization', 'Location', 'Profession', 'Contact', 'Name'],
        },
        'mit-movie': {
            'x-name': 'Query',
            'x-name-pl': 'Queries',
            'x-decoded-name': ['Query', 'Queries', 'Spoken Query'],
            # 'y-name': 'Keywords',
            # 'y-decoded-name': ['Keywords'],
            'y-name': 'Named Entities',
            'y-decoded-name': ['Named Entities'],
            # by manual inspection, `RATING` => `MPAA Rating` and `RATINGS_AVERAGE` => `Viewers' Rating`
            'entity-types': [
                'TITLE', 'RATINGS_AVERAGE', 'YEAR', 'GENRE', 'DIRECTOR', 'RATING',
                'PLOT', 'ACTOR', 'TRAILER', 'SONG', 'REVIEW', 'CHARACTER'
            ],
            'readable-entity-types': [
                'Title', "Viewers' Rating", 'Year', 'Genre', 'Director', 'MPAA Rating',
                'Plot', 'Actor', 'Trailer', 'Song', 'Review', 'Character'
            ]
        },
        'mit-restaurant': {
            'x-name': 'Query',
            'x-name-pl': 'Queries',
            'x-decoded-name': ['Query', 'Queries', 'Spoken Query'],
            'y-name': 'Named Entities',
            'y-decoded-name': ['Named Entities'],
            # TODO: `Goal` not found but mentioned in paper??
            'entity-types': ['Restaurant_Name', 'Amenity', 'Cuisine', 'Dish', 'Hours', 'Location', 'Price', 'Rating'],
            'readable-entity-types': [
                'Restaurant Name', 'Amenity', 'Cuisine', 'Dish', 'Hours', 'Location', 'Price', 'Rating'

                # try lower-casing
                # 'restaurant name', 'amenity', 'cuisine', 'dish', 'hours', 'location', 'price', 'rating'
            ]
        },
        'ncbi-disease': {
            'x-name': 'Sentence',
            'x-name-pl': 'Sentences',
            'x-decoded-name': ['Sentence'],
            'y-name': 'Diseases',
            'y-decoded-name': ['Diseases'],
            'entity-types': ['Disease'],
            'readable-entity-types': ['disease']
        }
    },
    'sub-directory-names': {
        'step-wise': 'step-wise',
        'attr-prompt': 'attr-prompt',
        'diverse-context': 'diverse-context',
        'diverse-entity': 'diverse-entity',
        'diversity': 'diversity-config'
    }
}


d_dsets: Dict[str, Any] = config_dict['datasets']
d_conll = d_dsets['conll2003']
# `conll2003-no-misc` is `conll2003` w/ `MISC` dropped
d_conll_no_misc = d_conll.copy()
for k in ['entity-types', 'readable-entity-types']:
    d_conll_no_misc[k] = d_conll_no_misc[k][:-1]  # drop `MISC`
d_dsets['conll2003-no-misc'] = d_conll_no_misc

# the config for `wiki-gold` is the same as that of `conll2003`
d_dsets['wiki-gold'] = d_conll.copy()
d_dsets['wiki-gold-no-misc'] = d_conll_no_misc.copy()

for dnm, d in d_dsets.items():
    readable = d.get('readable-entity-types')
    if readable:
        d['label2readable-label'] = dict(zip(d['entity-types'], readable))
        d['readable-ner-tags'] = ner_labels2tags(entity_types=readable)
for dnm in ['job-desc', 'job-stack']:
    d_dsets[dnm]['readable-entity-types'] = d_dsets[dnm]['entity-types']


if __name__ == '__main__':
    from stefutil import sic

    def run():
        sic.output_width = 256

        fl_nm = 'config.json'
        sic(config_dict)
        open(fl_nm, 'a').close()  # Create file in OS
        with open(os_join(BASE_PATH, PROJ_DIR, PKG_NM, 'util', fl_nm), 'w') as f:
            json.dump(config_dict, f, indent=4)
    run()
