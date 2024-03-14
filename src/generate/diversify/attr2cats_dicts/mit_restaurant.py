from src.generate.diversify.util import *

__all__ = ['attribute2categories_dict']


attribute2categories_dict = {
    ENTITY_KEY: dict(
        short='ent',
        seed_category=None,
        categories='200-ppt-v3',
        from_file=True,
        presets={
            '100': {
                ENTITY_TYPE_DEFAULT: '23-11-18_Entity-Dataset',
                # needs to be after to override `default`
                'Price': '23-11-18_Entity-Dataset_{ets=[Price],#demo=1,demo%=0.4}'
            },
            '200-ppt-v3': '23-12-28_Entity-Dataset_ppt-v3',
        }
    ),
    ENTITY_KEY_SEEDED: dict(
        short='ent-s',
        seed_category='meal-category',
        categories='10=>40-ppt-v3',
        from_file=True,
        presets_dependency={
            # '10': [
            #     'fine dining', 'buffet', 'late-night dining', 'dinner', 'café', 'fast food',
            #     'breakfast', 'lunch', 'brunch', 'food truck'
            # ],
            '10=>30': '10',
            '10=>40-ppt-v3': '10'
        },
        presets={
            '10=>30': '23-11-18_Entity-Dataset_{seed=meal}',
            '10=>40-ppt-v3': '23-12-28_Entity-Dataset_{seed=meal}'
        }
    ),
    'meal-category': dict(
        short='cat',
        name='Meal Category',
        attribute_names=['meal type', 'dining style'],
        desc=None,
        examples=[
            'breakfast', 'lunch', 'dinner', 'late-night',
            # 'snacks', 'burgers', 'pizza', 'salads',
            'fast food', 'fine dining', 'café', 'buffet'
        ],
        categories=[
            'fine dining', 'buffet', 'late-night dining', 'dinner', 'café', 'fast food',
            'breakfast', 'lunch', 'brunch', 'food truck'
        ]
    ),
    'location': dict(
        short='loc',
        attribute_names=['location', 'proximity'],
        desc='Specific neighborhood/city, or near a landmark',
        examples=['metropolitan', 'coastal', 'near me'],
        categories='',
        presets={
            '': ''
        }
    ),
    'demographic': dict(
        short='dem',
        name='User Demographic',
        attribute_names=['user demographic'],
        desc='age, gender, occupation, etc.',
        examples=None,
        categories='20',
        presets={
            '20': '23-11-18_Category-Dataset'
        }
    ),
    'ambiance': dict(
        short='amb',
        name='Ambiance',
        attribute_names=['ambiance', 'setting'],
        desc=None,
        examples=['romantic', 'family-friendly', 'outdoor seating', 'casual', 'formal'],
        categories='20',
        presets={
            '20': '23-11-18_Category-Dataset'
        }
    ),
    'price': dict(
        short='pr',
        name='Price Range',
        attribute_names=['price range'],
        desc=None,
        examples=['budget-friendly', 'mid-range', 'fine dining', 'affordable'],
        categories='6',
        presets={
            '20': '23-11-18_Category-Dataset',
            '6': '23-11-18_Category-Dataset_{attr=[price]}'
        }
    ),
    'dietary': dict(
        short='diet',
        name='Dietary Restriction',
        attribute_names=['dietary restriction'],
        desc='Health or ethical dietary preferences',
        examples=['vegan', 'vegetarian', 'gluten-free', 'halal', 'kosher'],
        categories='20',
        presets={
            '20': '23-11-18_Category-Dataset'
        }
    ),
    'special': dict(
        short='sp',
        name='Special Offering',
        attribute_names=['special feature', 'special event', 'special offer'],
        desc=None,
        examples=[
            'live music', 'waterfront views', 'happy hours', 'special discounts', 'outdoor seating',
            # 'pet-friendly', 'family-friendly', 'child-friendly',
            # 'live cooking sessions'
        ],
        categories=[
            "Chef's specials", 'Private Dining', 'Tasting Menus',
            'live music', 'waterfront views', 'happy hours', 'special discounts', 'outdoor seating',
        ]
    ),
    'service': dict(
        short='ser',
        name='Service Mode',
        attribute_names=['service type'],
        desc=None,
        examples=None,
        categories=['takeout', 'delivery', 'dine-in', 'drive-thru'],
    ),
}


for attr, d in attribute2categories_dict.items():
    if attr not in [ENTITY_KEY, ENTITY_KEY_SEEDED]:
        from_file = attr not in ['meal-category', 'special', 'service']
        d.update(kind='categorical', seed_category=None, from_file=from_file)


if __name__ == '__main__':
    from stefutil import sic
    sic(attribute2categories_dict)

