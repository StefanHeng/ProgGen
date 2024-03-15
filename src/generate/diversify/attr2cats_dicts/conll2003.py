from src.generate.diversify.util import ENTITY_KEY, ENTITY_KEY_SEEDED


__all__ = ['attribute2categories_dict']


attribute2categories_dict = {
    ENTITY_KEY: dict(
        short='ent',
        seed_category=None,
        categories='100-domain-2-dedup',
        # categories='300-75',
        # categories='150-50',
        presets={  # for ~400 unique named entities for each entity type
            # '400': '23-11-05_Processed-Entity-Data',
            '100': '23-11-19_Entity-Dataset',
            '150-domain-2': '23-12-28_Entity-Dataset_domain-2',
            '100-domain-2-dedup': '23-12-28_Entity-Dataset_domain-2-dedup',
            '300-75': '24-02-07_Entity-Dataset_75',
            '150-50': '24-02-07_Entity-Dataset_50'
        }
    ),
    ENTITY_KEY_SEEDED: dict(
        short='ent-s',
        seed_category='news-category',
        categories='new30=>40-domain-2',  # best-performing
        presets_dependency={
            '30=>200': '30',
            '30=>50': '30',
            '30=>50-split': '30',
            '30=>30': '30',
            'new30=>40': 'new30',
            'new30=>40-domain-2': 'new30'
        },
        presets={
            '30=>200': '23-11-09_Entity-Dataset_seeded',
            '30=>50': '23-11-10_Entity-Dataset_seeded_50',
            '30=>50-split': '23-11-10_Entity-Dataset_{drop-[,]=T,split-()=T}_seeded_50',
            '30=>30': '23-11-10_Entity-Dataset_seeded_30',
            'new30=>40': '23-11-19_Entity-Dataset_{seed=news}',
            'new30=>40-domain-2': '23-12-28_Entity-Dataset_{seed=news}_domain-2'
        }
    ),
    'news-category': dict(
        short='cat',
        name='News Topic',
        attribute_names=['news category'],
        desc=None,
        examples=[
            'politics', 'economics', 'technology', 'health', 'environment', 'culture', 'sports', 'human interest stories',
            'international events', 'local stories', 'science', 'breaking news', 'business'
        ],
        kind='categorical',
        seed_category=None,
        categories='new30',
        from_file=True,
        presets={
            'new30': '23-11-19_Category-Dataset',
            'dup90': [  # 3 completion runs from ChatGPT
                'Africa',
                'Agriculture',
                'Agriculture and Farming',
                'Arts & Culture',
                'Arts and Literature',
                'Asia-Pacific',
                'Automotive',
                'Automotive',
                'Automotive Industry',
                'Aviation',
                'Aviation',
                'Aviation and Aerospace',
                'Business',
                'Business',
                'Business and Finance',
                'Climate Change',
                'Climate Change',
                'Climate Change',
                'Crime',
                'Crime',
                'Economics',
                'Economy',
                'Education',
                'Education',
                'Education',
                'Energy',
                'Energy',
                'Energy and Environment',
                'Entertainment',
                'Entertainment',
                'Entertainment and Culture',
                'Environment',
                'Environment',
                'Fashion and Lifestyle',
                'Finance',
                'Finance',
                'Food and Beverage',
                'Global Markets',
                'Health',
                'Health',
                'Health and Wellness',
                'Healthcare and Medicine',
                'Human Rights',
                'Human Rights',
                'Human Rights',
                'Immigration',
                'Immigration',
                'Immigration and Refugees',
                'International News',
                'International Politics',
                'International Relations',
                'Investigations',
                'Law',
                'Law and Justice',
                'Legal Issues and Courts',
                'Lifestyle',
                'Lifestyle',
                'Markets',
                'Markets',
                'Middle East',
                # 'National Politics (coverage specific to various countries)',
                'National Politics',  # manual edit, drop the part in parentheses
                'Opinion',
                'Politics',
                'Politics',
                'Real Estate',
                'Real Estate',
                'Real Estate and Housing',
                'Religion',
                'Retail',
                'Science',
                'Science',
                'Science and Research',
                'Sports',
                'Sports',
                'Sports',
                'Stock Markets',
                'Technology',
                'Technology',
                'Technology Companies and Startups',
                'Technology and Innovation',
                'Terrorism and Security',
                'Trade and Economics',
                'Travel',
                'Travel',
                'Travel and Tourism',
                'Weather',
                'Weather and Natural Disasters',
                'World Affairs',
                'World News',
                'World News'
            ],
            '30': [
                'Agriculture and Farming',
                'Arts and Literature',
                'Automotive Industry',
                'Aviation and Aerospace',
                'Business and Finance',
                'Climate Change',
                'Education',
                'Energy and Environment',
                'Entertainment and Culture',
                'Fashion and Lifestyle',
                'Food and Beverage',
                'Global Markets',
                'Health and Wellness',
                'Healthcare and Medicine',
                'Human Rights',
                'Immigration and Refugees',
                'International Politics',
                'Legal Issues and Courts',
                # 'National Politics (coverage specific to various countries)',
                'National Politics',  # manual edit, drop the part in parentheses
                'Real Estate and Housing',
                'Science and Research',
                'Sports',
                'Stock Markets',
                'Technology Companies and Startups',
                'Technology and Innovation',
                'Terrorism and Security',
                'Trade and Economics',
                'Travel and Tourism',
                'Weather and Natural Disasters',
                'World News'
            ],
            '9': [
                'Business and Finance',
                'Entertainment',
                'Environment',
                'Health',
                'International News',
                'Politics',
                'Science',
                'Sports',
                'Technology'
            ],
        }
    ),
    'length': dict(
        short='len',
        attribute_names=['length'],
        kind='categorical',
        seed_category=None,
        categories='3',
        from_file=False,
        presets={
            '3': [
                'between 2 words to 15 words',
                'between 15 words to 30 words',
                'between 35 words to 50 words'
            ]
        }
    ),
    'sub-category': dict(
        short='sub',
        attribute_names=['sub-category'],
        kind='categorical',
        seed_category='news-category',
        categories='30=>10',
        from_file=True,
        presets_dependency={
            '30=>10': '30',
            '9=>10': '9',
            '9=>20': '9'
        },
        presets={
            '30=>10': '23-10-10_Processed-Categories_{attrs=[sub-category]}_prompt1',
            '9=>10': '23-10-11_Processed-Categories_{attrs=[sub-category],#list=10,dep={sub<=9}}',
            '9=>20': '23-10-11_Processed-Categories_{attrs=[sub-category],#list=20,dep={sub<=9}}'
        }
    ),
    'subtopic': dict(
        attribute_names=['subtopic', 'theme'],
        kind='categorical',
        seed_category='sub-category',
        from_file=True,
        categories=[]  # TODO
    ),
    'perspective': dict(
        short='per',
        attribute_names=['perspective', 'view point'],
        kind='categorical',
        seed_category='news-category',
        categories='30=>5',
        from_file=True,
        presets_dependency={
            '30=>5': '30',
        },
        presets={
            '30=>5': '23-10-13_Processed-Categories_{attrs=[perspective],#list=5,dep={per<=9}}_cat30-edit',
        }
    ),
    'location': dict(
        short='loc',
        attribute_names=['location', 'geographic region'],
        kind='categorical',
        seed_category=None,
        categories='30',
        from_file=False,
        presets={
            '80': [  # 80 categories
                'Africa',
                'Egypt',
                'Hong Kong',
                'United States',
                'United Kingdom',
                'Russia',
                'China',
                'India',
                'Brazil',
                'Japan',
                'Australia',
                'Canada',
                'Mexico',
                'France',
                'Germany',
                'Spain',
                'Italy',
                'South Africa',
                'Nigeria',
                'Kenya',
                'Saudi Arabia',
                'United Arab Emirates',
                'Israel',
                'Turkey',
                'Iran',
                'Iraq',
                'Afghanistan',
                'Pakistan',
                'South Korea',
                'North Korea',
                'Indonesia',
                'Thailand',
                'Vietnam',
                'Philippines',
                'Malaysia',
                'Singapore',
                'Argentina',
                'Chile',
                'Peru',
                'Colombia',
                'Venezuela',
                'Bolivia',
                'Ecuador',
                'Guatemala',
                'Cuba',
                'Haiti',
                'Dominican Republic',
                'Jamaica',
                'Puerto Rico',
                'Venezuela',
                'Ukraine',
                'Belarus',
                'Poland',
                'Hungary',
                'Czech Republic',
                'Greece',
                'Portugal',
                'Netherlands',
                'Belgium',
                'Switzerland',
                'Sweden',
                'Norway',
                'Denmark',
                'Finland',
                'Iceland',
                'New Zealand',
                'Papua New Guinea',
                'Fiji',
                'Solomon Islands',
                'Tuvalu',
                'Samoa',
                'Tonga',
                'Kiribati',
                'Marshall Islands',
                'Micronesia',
                'Palau',
                'Northern Mariana Islands',
                'Guam',
                'American Samoa',
                'Cook Islands'
            ],
            '30': [  # 30 categories
                'United States',
                'United Kingdom',
                'China',
                'Russia',
                'India',
                'Brazil',
                'Germany',
                'France',
                'Japan',
                'Canada',
                'Australia',
                'South Africa',
                'Mexico',
                'South Korea',
                'Saudi Arabia',
                'Israel',
                'Turkey',
                'Argentina',
                'Nigeria',
                'Egypt',
                'Iran',
                'Ukraine',
                'Venezuela',
                'Indonesia',
                'Kenya',
                'Chile',
                'Sweden',
                'Singapore',
                'New Zealand',
                'United Arab Emirates'
            ],
            'cont6': [  # continents dropping `Antarctica`, taken from AttrPrompt repo
                'Europe',
                'Asia',
                'Africa',
                'North America',
                'South America',
                'Oceania'
            ]
        }
    ),
    'angle': dict(
        attribute_names=['angle', 'target reader', 'target audience'],
        kind='categorical',
        categories=['global', 'local']  # TODO: not a good one intuitively?
    ),
    'writing-style': dict(
        short='style',
        name='Writing Style',
        attribute_names=['writing style'],
        desc=None,
        examples=[
            ['straightforward', 'factual', 'clear', 'concise', 'objective', 'narrative', 'analytical'],
            ['in-depth analysis', 'opinion pieces', 'feature stories', 'op-eds']
        ],
        categories='new8',
        seed_category=None,
        from_file=True,
        presets={
            'new8': '23-11-19_Category-Dataset',
            'dup20': [  # 3 completion runs from ChatGPT
                'News Reporting',
                'Feature Articles',
                'Opinion Pieces',
                'Data-Driven Analysis',
                'Interviews',
                'Hard News Reporting',
                'Feature Articles',
                'Opinion and Editorial Pieces',
                'Business and Financial Reporting',
                'Human Interest and Lifestyle Stories',
                'In-Depth and Analytical',
                'Narrative and Human-Interest',
                'Breaking News and Alerts',
                'Investigative Journalism',
                'Opinion and Editorial',
                'Feature Stories',
                'Data-Driven Journalism',
                'Interviews and Profiles',
                'Visual Journalism',
                'Explainer and How-To Articles'
            ],
            '5': [  # taken from AttrPrompt repo
                'Investigative journalism: '
                'This style involves in-depth research and analysis of a specific topic, '
                'often involving uncovering hidden information or exposing wrongdoing',
                'Op-Eds: '
                'This style involves the expression of opinions and ideas on current events, '
                'often written by experts or influential individuals in their respective fields',
                'Feature writing: '
                'This style involves in-depth storytelling, '
                'often highlighting individuals or events that have a significant impact on society',
                'News analysis: '
                'This style involves examining and interpreting current events and news stories, '
                'often focusing on the broader context and implications of a particular event or trend',
                'Profiles and interviews: '
                'This style involves highlighting specific individuals, '
                'whether they be celebrities, politicians, or ordinary people, through interviews or in-depth reporting'
            ]
        }
    ),
    'culture': dict(
        short='cul',
        attribute_names=['culture', 'cultural background', 'cultural perspective', 'social', 'tradition'],
        kind='categorical',
        seed_category='news-category',
        categories='30=>5',
        from_file=True,
        presets_dependency={
            '30=>5': '30',
        },
        presets={
            '30=>5': '23-10-13_Processed-Categories_{attrs=[culture],#list=5,dep={cul<=9}}_cat30-edit'
        }
    ),
    'objectivity': dict(
        attribute_names=['objectivity', 'balance'],
        kind='categorical',
        categories=['objective', 'subjective']  # TODO: add 'balanced', 'neutral', 'biased'?
    ),
    'tone': dict(
        attribute_names=['tone', 'sentiment'],
        kind='categorical',
        categories=['positive', 'critical']  # TODO: add 'negative', 'neutral'?
    ),
    'timeline': dict(
        attribute_names=['timeline'],
        kind='binary',
        instruction=[
            'Please include historical context',
            'Please include historical milestone'
        ]
    ),
    'data': dict(
        # attribute_names=['data', 'statistics'],
        kind='binary',
        instruction=[
            'Please include data',
            'Please include statistics'
        ]
    ),
    'expert-comment': dict(
        # attribute_names=['expert comment', 'expert opinion'],
        kind='binary',
        instruction=[
            'Please include expert commentary',
            'Please include expert opinion',
            'Please include expert insight'
        ]
    ),
    'accessibility': dict(
        kind='binary',
        instruction=[
            'Please write in a way that is easily understandable by a broad audience'
        ]
    )
}
