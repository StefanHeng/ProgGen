from src.generate.diversify.util import *


__all__ = ['attribute2categories_dict']


attribute2categories_dict = {
    ENTITY_KEY: dict(
        short='ent',
        seed_category=None,
        categories='100-ppt-v3',
        presets={
            '100': '23-11-14_Entity-Dataset',
            '100+demo': '23-11-14_Entity-Dataset_{#demo=1,demo%=0.4}',
            '100-domain-twice': '23-12-28_Entity-Dataset_domain-twice',
            '100-ppt-v3': '23-12-28_Entity-Dataset_et-ppt-v3'
        }
    ),
    ENTITY_KEY_SEEDED: dict(
        short='ent-s',
        seed_category='job-category',
        categories='new20=>40-domain-twice',
        presets_dependency={
            '14=>40': '10',
            '14=>40+demo': '10',
            'new20=>40': 'new10',
        },
        presets={
            '14=>40': '23-11-14_Entity-Dataset_{seed=job}',
            '14=>40+demo': '23-11-14_Entity-Dataset_{seed=job,#demo=1,demo%=0.4}',
            'new20=>40+demo': '23-11-26_Entity-Dataset_{seed=job,#demo=1,demo%=0.4}',
            'new20=>40-domain-twice': '23-12-28_Entity-Dataset_{seed=job}_domain-twice',
        }
    ),
    'job-category': dict(
        short='cat',
        attribute_names=['job category'],
        desc=None,
        examples=['software development', 'data science', 'system administration', 'graphic design', 'product management'],
        categories='new10',
        presets={
            '10': '23-11-14_Category-Dataset',
            'new10': '23-11-26_Category-Dataset'
        }
    ),
    'language': dict(
        short='lang',
        attribute_names=['language', 'sentence complexity'],
        desc=None,
        examples=['straightforward', 'concise', 'detailed', 'technical', 'formal', 'casual', 'friendly'],
        categories='new10',
        presets={
            '10': '23-11-14_Category-Dataset',
            'new10': '23-11-26_Category-Dataset'
        }
    ),
    'experience-level': dict(
        short='lvl',
        attribute_names=['experience level'],
        desc=None,
        examples=['entry-level', 'mid-level', 'senior'],
        categories='new10',
        presets={
            '10': '23-11-14_Category-Dataset',
            'new10': '23-11-26_Category-Dataset'
        }
    ),
    'location': dict(
        short='lvl',
        attribute_names=['location'],
        desc=None,
        examples=['onsite', 'remote', 'hybrid'],
        categories='10',
        presets={
            '10': '23-11-14_Category-Dataset'
        }
    ),
    'culture': dict(
        short='cul',
        attribute_names=['culture', 'company culture', 'company value', 'work environment'],
        desc=None,
        examples=['collaborative', 'innovation-focus', 'work-life balance'],
        categories='10',
        presets={
            '10': '23-11-14_Category-Dataset'
        }
    ),
    'tone': dict(
        short='ton',
        attribute_names=['tone'],
        desc=None,
        examples=['friendly', 'professional', 'enthusiastic'],
        categories='new10',
        presets={
            '10': '23-11-14_Category-Dataset',
            'new10': '23-11-26_Category-Dataset'
        }
    )
}


for attr, d in attribute2categories_dict.items():
    if attr not in [ENTITY_KEY, ENTITY_KEY_SEEDED]:
        d.update(kind='categorical', seed_category=None, from_file=True)
