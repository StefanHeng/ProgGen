from typing import Dict, Any

from src.generate.diversify.util import ENTITY_KEY, ENTITY_KEY_SEEDED


__all__ = ['attribute2categories_dict']


attribute2categories_dict = {
    ENTITY_KEY: dict(
        short='ent',
        seed_category=None,
        categories='200',
        presets={
            '200': '23-11-15_Entity-Dataset'
        }
    ),
    ENTITY_KEY_SEEDED: dict(
        short='ent-s',
        seed_category='topic',
        categories='new30=>15',
        presets_dependency={
            '30=>30': '30',
            '30=>15': '30',
            'new30=>15': 'new30'
        },
        presets={
            '30=>30': '23-11-15_Entity-Dataset_{seed=topic}',
            '30=>15': '23-11-17_Entity-Dataset_{seed=topic}',
            'new30=>15': '23-11-27_Entity-Dataset_{seed=topic}'
        }
    ),
    'topic': dict(
        short='cat',
        name='Topic',
        attribute_names=['topic'],
        desc=None,
        examples=['science', 'history', 'culture', 'technology', 'art', 'biography'],
        categories='new30',
        presets={
            '30': '23-11-15_Category-Dataset',
            'new30': '23-11-26_Category-Dataset_{attr=[topic]}'
        }
    ),
    'language': dict(
        short='lang',
        name='Writing Style',
        attribute_names=['language', 'length', 'writing style', 'tone'],
        desc=None,
        examples=['concise', 'detailed', 'formal', 'informal', 'simple', 'technical', 'narrative', 'neutral'],
        categories='new30',
        presets={
            '30': '23-11-15_Category-Dataset',
            'new30': '23-11-26_Category-Dataset'
        }
    ),
}


for attr, d in attribute2categories_dict.items():
    d: Dict[str, Any]
    if attr not in [ENTITY_KEY, ENTITY_KEY_SEEDED]:
        d.update(kind='categorical', seed_category=None, from_file=True)
