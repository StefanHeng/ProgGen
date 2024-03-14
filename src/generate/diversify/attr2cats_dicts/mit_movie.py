from src.generate.diversify.util import *


__all__ = ['attribute2categories_dict']


attribute2categories_dict = {
    ENTITY_KEY: dict(
        short='ent',
        seed_category=None,
        categories='150-type-ppt',
        # categories='200-per3',
        presets={  # for ~400 unique named entities for each entity type
            '100': '23-11-12_Entity-Dataset',
            'new-gpt-100': {
                ENTITY_TYPE_DEFAULT: '23-11-21_Entity-Dataset',
                # ('Plot', 'Trailer', 'Review'): '23-11-21_Entity-Dataset_{ets=[Plot,Trailer,Review],#demo=1,demo%=0.4}'
            },
            '100-ppt-v1': '23-12-27_Entity-Dataset',
            '100-ppt-v1-mv': '23-12-27_Entity-Dataset_moved',
            '100-ppt-v2': '23-12-27_Entity-Dataset_ppt-v2',
            '150-ppt-v3': '23-12-27_Entity-Dataset_ppt-v3',
            '200-ppt-v4': '23-12-27_Entity-Dataset_ppt-v4',
            '100-type-ppt': '24-01-16_Entity-Dataset_type-ppt-search',
            '150-type-ppt': '24-01-17_Entity-Dataset_type-ppt-search-10',
            '200-per': '24-02-06_Entity-Dataset_larger-pool',
            '220-per': '24-02-06_Entity-Dataset_larger-pool-15',
            '200-per2': '24-02-06_Entity-Dataset_larger-pool2',
            '200-per3': '24-02-06_Entity-Dataset_larger-pool3'
        }
    ),
    ENTITY_KEY_SEEDED: dict(
        short='ent-s',
        seed_category='query-category',
        categories='new10=>100-type-ppt',
        presets_dependency={
            '15=>30': '15',
            'new10=>new30': 'new10',
            'new10=>new40': 'new10',
            'new10=>40-ppt-v3': 'new10'
        },
        presets={
            '15=>30': '23-11-12_Entity-Dataset_seeded',
            'new10=>new30': '23-11-21_Entity-Dataset_{seed=query}_30',
            'new10=>new40': '23-11-21_Entity-Dataset_{seed=query}_40',
            'new10=>100-ppt-v3': '23-12-27_Entity-Dataset_{seed=query}',
            'new10=>100-type-ppt': '24-02-06_Entity-Dataset_{seed=query}',
        }
    ),
    'query-category': dict(
        short='cat',
        name='Query Category',
        attribute_names=['query category'],
        # desc=None,
        # examples=[
        #     'movie trivia', 'comparative', 'open-ended', 'specific request', 'interactive', 'cross-genre',
        #     'movie recommendations', 'plot summaries', 'actor information', 'Easter eggs', 'interesting behind-the-scenes facts'
        # ],
        # desc='The type of information being sought',
        # examples=[
        #     'movie recommendations',
        #     # 'specific movie information',
        #     'movie information',
        #     'movie reviews and ratings',
        #     'movie showtimes and locations',
        #     # 'basic information',
        #     'plot and content',
        #     # 'availability',
        #     # 'movie series or franchise'
        # ],
        # TODO: providing descriptions and examples seemed to make lower-quality categories
        desc=None,
        examples=None,
        categories='new10',
        presets={
            # '15': '23-10-18_Processed-Categories_{attrs=[query-category]}'
            'new10': '23-11-21_Category-Dataset_{attr=[query-category]}'
        },
    ),
    'user-persona': dict(
        short='per',
        attribute_names=['user persona', 'user personality'],
        desc='Distinct preferences and ways of phrasing queries',
        examples=[
            'film enthusiasts', 'casual viewers', 'critics', 'parents looking for family-friendly options'
        ],
        categories='20',
        presets={
            '20': '23-10-18_Processed-Categories_{attrs=[user-persona]}'
        },
    ),
    'demographic': dict(
        short='dem',
        name='User Demographic',
        attribute_names=['user demographic'],
        desc="Different age groups and genders",
        examples=[
            'young', 'older',
            # 'gender-neutral',
            'age-friendly', 'family-friendly',
            # 'accessibility'
        ],
        categories='new15',
        presets={
            'new15': '23-11-21_Category-Dataset_{attr=[demographic]}'
        },
    ),
    'preference': dict(
        short='pref',
        attribute_names=['preference', 'user preference'],
        desc=None,
        examples=[
            'preference for humor', 'interested in history',
            'language', 'movie ratings', 'personal experience'
        ],
        categories='10',
        presets={
            '10': '23-10-18_Processed-Categories_{attrs=[preference]}'
        },
    ),
    'query-complexity': dict(
        short='len',
        attribute_names=['query complexity', 'length', 'level of detail'],
        desc=None,
        examples=['straightforward', 'complex', 'detailed', 'simple', 'in-depth', 'short', 'quick', 'intricate'],
        categories='10',
        presets={
            '10': '23-10-18_Processed-Categories_{attrs=[query-complexity]}'
        },
    ),
    'time-period': dict(
        short='tm',
        attribute_names=['time period', 'time range', 'time frame'],
        desc='related to specific time periods',
        examples=[
            'recent releases', 'all-time classics', 'specific decades', 'classic movies from the 1950s', 'upcoming films',
            'cult classics from the 1980s'
        ],
        categories='15',
        presets={
            '15': '23-10-18_Processed-Categories_{attrs=[time-period]}'
        },
    ),
    'culture': dict(
        short='cul',
        attribute_names=['cultural reference', 'current event', 'trending topic', 'trend', 'culture'],
        # desc='reference popular culture or current events',
        # examples=[
        #     # 'space exploration for World Space Week', 'film festivals', 'emerging movie trends', 'award shows', 'movie premieres'
        #     # 'film festivals', 'emerging movie trends', 'award shows', 'movie premieres'
        #     'film festivals', 'award shows', 'movie premieres'
        # ],
        # desc='Include culturally specific or regionally popular movies',
        desc=None,
        examples=[
            ['Hollywood', 'Bollywood'], ['East Asian cinema', 'French movies']
        ],
        categories='new20',
        presets={
            '15': '23-10-18_Processed-Categories_{attrs=[culture]}',
            'new20': '23-11-21_Category-Dataset'
        },
    ),
    'emotion': dict(
        short='emo',
        name='User Emotion',
        attribute_names=['emotion', 'sentiment', 'mood'],
        desc=None,
        # examples=['adventurous', 'happy', 'sad', 'excited', 'nostalgic'],
        examples=['curious', 'frustrated'],
        categories='new20',
        presets={
            '15': '23-10-18_Processed-Categories_{attrs=[emotion]}',
            'new20': '23-11-21_Category-Dataset'
        },
    ),
    'language': dict(
        short='lang',
        name='Query Language',
        attribute_names=['language', 'language style', 'query style'],
        desc=None,
        # examples=['formal', 'informal', 'casual', 'slang', 'technical'],
        examples=['formal', 'informal', 'casual', 'slang', 'technical', 'straightforward', 'indirect', 'vague', 'colloquial'],
        categories='new20',
        presets={
            '8': '23-10-18_Processed-Categories_{attrs=[language]}',
            'new20': '23-11-21_Category-Dataset'
        },
    )
}


for attr, d in attribute2categories_dict.items():
    if attr not in [ENTITY_KEY, ENTITY_KEY_SEEDED]:
        d.update(kind='categorical', seed_category=None, from_file=True)


if __name__ == '__main__':
    from stefutil import sic
    sic(attribute2categories_dict)
