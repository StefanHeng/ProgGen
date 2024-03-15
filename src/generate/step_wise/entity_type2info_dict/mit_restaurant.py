from src.generate.step_wise.entity_type2info_dict.util import CORRECT, WRONG_SPAN, WRONG_TYPE, NA


__all__ = ['entity_type2info_dict']


entity_type2info_dict = {
    '24-02-12_NER-Dataset_{fmt=n-p2,#l=50,lc=T}_add-super-idx': {
        'Restaurant Name': dict(
            defn=(
                # 'A named restaurant name entity must be the name of a restaurant.'
                'A restaurant name entity must be the name of a restaurant. '
                'Reference to a restaurant by cuisine such as "taco truck" should be a named cuisine entity. '
            ),
            name='restaurant name',
            name_full='restaurant name entity',  # override default templating, i.e. `named {entity_type} entity`
            demos=[
                dict(sentence="Can I get a table for two at Ruth's Chris Steak House tonight?", entity_span="Ruth's Chris Steak House", label=CORRECT),
                dict(sentence="What's the average cost of a meal at the steakhouse downtown?", entity_span="steakhouse", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence="Which restaurant serves the best brunch in the city?", entity_span="restaurant", label=NA),
            ]
        ),
        'Amenity': dict(
            defn=(
                'A named amenity entity is a feature or service offered by a restaurant. '
                'Descriptions on the atmosphere and ambiance of a restaurant such as "fancy", "romantic" and "trendy" are also named amenity entities. '
                'Meal times such as "breakfast", "brunch", "lunch" and "dinner" should be named Hours entities. '
                'General types of eateries such as "diner", "cafe" and "fast food" should be named Cuisine entities.'
            ),
            demos=[
                dict(sentence='Is there a place nearby that has a good happy hour deal?', entity_span='happy hour', label=CORRECT),
                dict(sentence='I want to try a new fusion restaurant, any suggestions?', entity_span='new', label=CORRECT),
                # dict(sentence="I'm searching for a buffet restaurant with a variety of options.", entity_span="buffet", label=CORRECT),
                # dict(sentence="Where's the best place to get a big breakfast in this town?", entity_span="breakfast", label=WRONG_TYPE, correct_type='Hours'),
                dict(sentence="Find a place that serves bottomless mimosas for brunch", entity_span="brunch", label=WRONG_TYPE, correct_type='Hours'),
                dict(sentence="Are there any all-day breakfast places around here?", entity_span="all-day breakfast", label=WRONG_TYPE, correct_type='Hours'),
                dict(sentence="Is there a vegan-friendly cafe with gluten-free options?", entity_span="cafe", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence='I want to have a romantic dinner at a French bistro, any romantic ones nearby?', entity_span="French bistro", label=WRONG_TYPE, correct_type='Cuisine'),
            ]
        ),
        'Cuisine': dict(
            defn=(
                'A named cuisine entity is a style or type of cooking. '
                # '"Sushi", "pizza", "tacos" and "burgers" are named cuisine entities. '
                'A named cuisine entity must not have trailing "place" or "restaurant". '
                'Starting Cuisine adjectives such as "authentic" and "classic" should be a part of the Cuisine entity. '
                'Trailing "food" should not be a part of a Cuisine entity if not necessary, such as "vegan" as opposed to "vegan food". '
                # this doesn't help at all, and makes things worse
                # '\n'
                # 'Restaurant name entities must be the name of a restaurant.'
                # 'You should not correct the entity type to a restaurant name entity unless the span is the specific name of a restaurant. '
                # 'A restaurant name entity must be a unique name that distinguishes one restaurant from another. '
                # 'Otherwise, you should never correct the entity type to a restaurant name entity.'
                # 'You should not correct the entity type to restaurant name entity unless '
                # 'For example, "In-N-Out Burger" is a restaurant name entity, but just "burger" is a named Cuisine entity.'
            ),
            demos=[
                # dict(sentence="I'm looking for a pizza place in the area", entity_span='pizza place', label=WRONG_SPAN, correct_span='pizza'),
                dict(sentence="Can you recommend a dessert place that's open late?", entity_span="dessert place", label=WRONG_SPAN, correct_span='dessert'),
                # dict(sentence="What is the best Mexican restaurant in this area?", entity_span="Mexican", label=CORRECT),
                dict(sentence='What time does the Italian restaurant on Main Street close tonight?', entity_span="Italian restaurant", label=WRONG_SPAN, correct_span='Italian'),
                dict(sentence='I want to try some authentic Mexican food, any recommendations?', entity_span="authentic Mexican food", label=WRONG_SPAN, correct_span='authentic Mexican'),
                dict(sentence='Find me a burger joint with a drive-thru.', entity_span="burger joint", label=CORRECT),
                dict(sentence='I need to make a reservation at the French bistro', entity_span="French bistro", label=CORRECT),
            ]
        ),
        'Dish': dict(
            defn=(
                'A named dish entity is a specific food item or meal. '
                'Be sure to distinguish a named dish entity from a named cuisine entity based on the context. '
                'Reference to a broader category of food should be a named Cuisine entity. '
                # 'General types of food such as "breakfast", "lunch" and "dinner" should be named Hours entities. '
                # 'General types of eateries such as "diner", "cafe" and "fast food" should be named Cuisine entities.'
            ),
            demos=[
                dict(sentence='What is the best burger joint in town?', entity_span="burger", label=WRONG_TYPE, correct_type='Cuisine'),
                # dict(sentence="I'm craving some authentic [sushi], where should I go?", entity_span="sushi", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence='Recommend a place to get a good burger.', entity_span="burger", label=CORRECT),
                dict(sentence="I want to try a new dish, what's popular at the Thai restaurant?", entity_span="new dish", label=NA),
                dict(sentence='Where can I find a late-night diner that serves breakfast? ', entity_span="breakfast", label=WRONG_TYPE, correct_type='Hours'),
            ]
        ),
        'Hours': dict(
            defn=(
                'Any temporal reference to a specific time of day or day of the week is a named Hours entity. '
                'A named hours entity must not contain irrelevant terms such as "diner" or "reservation". '
                'Days of the week such as "Friday" and "Wednesday" are named Hours entities. '
                'Meal times such as "breakfast", "brunch", "lunch" and "dinner" are also named Hours entities.'
            ),
            demos=[
                dict(sentence='I need to find a 24-hour diner for late-night cravings.', entity_span='24-hour diner', label=WRONG_SPAN, correct_span='24-hour'),
                dict(sentence='I need to book a table at Olive Garden for 7 pm tonight.', entity_span='7 pm tonight', label=CORRECT),
                dict(sentence="I'm looking for a quick lunch spot with vegetarian options", entity_span="lunch", label=CORRECT),
                dict(sentence='Is there a place around here that serves late-night tacos?', entity_span="late-night", label=CORRECT),
                dict(sentence='Call Olive Garden and make a [reservation for 7:00 pm].', entity_span='reservation for 7:00 pm', label=WRONG_SPAN, correct_span='7:00 pm'),
            ]
        ),
        'Location': dict(
            defn=(
                'A named location entity refers to the geographical location or relative proximity of the restaurant. '
                'Proximity adjectives such as "nearest" and "closest" are also named location entities. '
                'A named location entity should include all relevant terms such as "in this city" as opposed to "city". '
                # 'General mentions such as "place", "spot" and "address" are not named entities.'
                # 'General mentions such as "place", "spot" and "address" are not relevant named entities.'
            ),
            demos=[
                dict(sentence="What's the best place for a Sunday brunch in this city?", entity_span='city', label=WRONG_SPAN, correct_span='in this city'),
                dict(sentence="I'm craving some ice cream. Any good dessert places near me?", entity_span='near me', label=CORRECT),
                dict(sentence='What are the hours for the nearest sushi bar?', entity_span='nearest', label=CORRECT),
                dict(sentence="I'm looking for a cheap place to eat near downtown", entity_span='downtown', label=WRONG_SPAN, correct_span='near downtown'),
                dict(sentence='Are there any halal food trucks in the downtown area?', entity_span='downtown area', label=WRONG_SPAN, correct_span='in the downtown area'),
                dict(sentence='Can you help me locate a place that serves gluten-free options?', entity_span='place', label=NA),
                # dict(sentence='Can you help me locate a place that serves gluten-free options?', entity_span='place', label=WRONG_TYPE, correct_type='other'),
                # dict(sentence='I need to know the address of the nearest Starbucks', entity_span='address', label=NA),
                # dict(sentence='I need to know the address of the nearest Starbucks', entity_span='address', label=WRONG_TYPE, correct_type='other'),
            ]
        ),
        'Price': dict(
            defn=(
                'A named price entity is a specific monetary value or range such as "under $10 per person". '
                'Cost-related adjectives such as "cheap", "affordable", "reasonable" and "upscale" are also named price entities. '
                'Named price entities should not have trailing terms such as "prices" and "eats". '
                'General mentions of price such as "price range" and "prices" are not named entities.'
            ),
            demos=[
                dict(sentence='Any all-you-can-eat buffets in the downtown area?', entity_span='all-you-can-eat', label=WRONG_TYPE, correct_type='Amenity'),
                dict(sentence="What's the price range for the steakhouse downtown", entity_span="price range", label=NA),
                dict(sentence="How much does the average meal cost at Olive Garden", entity_span="average meal cost", label=NA),
                dict(sentence='Where can I get a good slice of pizza for under $5?', entity_span='under $5', label=CORRECT),
                # dict(sentence='Is there a 24-hour diner in the area with affordable prices? ', entity_span='affordable prices', label=WRONG_SPAN, correct_span='affordable'),
                dict(sentence='Is there a steakhouse nearby with a [reasonable price]?', entity_span='reasonable price', label=WRONG_SPAN, correct_span='reasonable'),
                # dict(sentence="Where's the cheapest place for a quick bite to eat?", entity_span="cheapest", label=CORRECT),
                dict(sentence="I'm looking for a place that offers a prix fixe menu.", entity_span="prix fixe", label=WRONG_TYPE, correct_type='Amenity'),
            ]
        ),
        'Rating': dict(
            defn=(
                'A named rating entity is a specific rating score such as "5 star". '
                'Qualitative descriptions such as "good", "best", "delicious" and "popular" are also named rating entities. '
                'You should capture the complete descriptive phrase such as "best reviewed" as opposed to "best".'
            ),
            demos=[
                dict(sentence='I want to try some Caribbean cuisine, where can I find a good place?', entity_span='good', label=CORRECT),
                dict(sentence='recommend a family-friendly Italian place with a kids menu', entity_span='family-friendly', label=WRONG_TYPE, correct_type='Amenity'),
                dict(sentence='Find me a steakhouse with a 5-star rating', entity_span='5-star', label=WRONG_SPAN, correct_span='5 star rating'),
                dict(sentence='Find me a restaurant with a 5-star rating for dinner tonight.', entity_span='5-star', label=WRONG_SPAN, correct_span='5 star rating'),
                dict(sentence='What is the rating of the nearest Italian restaurant?', entity_span='rating', label=NA),
                dict(sentence='Can you recommend a restaurant with a high Zagat rating?', entity_span='high', label=WRONG_SPAN, correct_span='high Zagat rating'),
            ]
        )
    },
    '24-02-15_NER-Dataset_{fmt=n-p2,#l=3,dc=T,lc=T}_add-super-idx': {
        'Restaurant Name': dict(
            defn=(
                # 'A named restaurant name entity must be the name of a restaurant.'
                'A restaurant name entity must be the name of a restaurant.\n'
                # 'Reference to a restaurant by cuisine such as "taco truck" should be a named cuisine entity. '
                'Ensure to distinguish restaurant name entities from a named cuisine entities. '
                'A named cuisine entity is a style or type of cooking, such as "diner", "burger", "food truck" and "café".'
            ),
            name='restaurant name',
            name_full='restaurant name entity',  # override default templating, i.e. `named {entity_type} entity`
            demos=[
                dict(sentence="Can I get a table for two at Ruth's Chris Steak House tonight?", entity_span="Ruth's Chris Steak House", label=CORRECT),
                dict(sentence="What's the average cost of a meal at the steakhouse downtown?", entity_span="steakhouse", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence="Which restaurant serves the best brunch in the city?", entity_span="restaurant", label=NA),
                # dict(sentence="I'm craving some street food, where can I find a food truck with a diverse menu?", entity_span="food truck", label=WRONG_TYPE, correct_type='Cuisine'),
            ]
        ),
        'Amenity': dict(
            defn=(
                'A named amenity entity is a feature or service offered by a restaurant. '
                'Descriptions on the atmosphere and ambiance of a restaurant such as "fancy", "romantic" and "trendy" are also named amenity entities.\n'
                'Ensure to distinguish named amenity entities from named hours entities and named cuisine entities. '
                'Meal times such as "breakfast", "brunch", "lunch" and "dinner" should be named Hours entities. '
                # 'General types of eateries such as "diner", "cafe" and "fast food" should be named Cuisine entities.'
                'A named cuisine entity is a style or type of cooking, such as "diner", "burger", "food truck" and "café".'
            ),
            demos=[
                dict(sentence='Is there a place nearby that has a good happy hour deal?', entity_span='happy hour', label=CORRECT),
                # dict(sentence='I want to try a new fusion restaurant, any suggestions?', entity_span='new', label=CORRECT),
                # dict(sentence="I'm searching for a buffet restaurant with a variety of options.", entity_span="buffet", label=CORRECT),
                dict(sentence="Where's the best place to get a big breakfast in this town?", entity_span="breakfast", label=WRONG_TYPE, correct_type='Hours'),
                # dict(sentence="Find a place that serves bottomless mimosas for brunch", entity_span="brunch", label=WRONG_TYPE, correct_type='Hours'),
                # dict(sentence="Are there any all-day breakfast places around here?", entity_span="all-day breakfast", label=WRONG_TYPE, correct_type='Hours'),
                dict(sentence="Is there a vegan-friendly cafe with gluten-free options?", entity_span="cafe", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence='Can you recommend a food truck with Korean BBQ?', entity_span="food truck", label=WRONG_TYPE, correct_type='Cuisine'),
                # dict(sentence='I want to have a romantic dinner at a French bistro, any romantic ones nearby?', entity_span="French bistro", label=WRONG_TYPE, correct_type='Cuisine'),
            ]
        ),
        'Cuisine': dict(
            defn=(
                # 'A named cuisine entity is a style or type of cooking. '
                'A named cuisine entity is a style or type of cooking, such as "diner", "burger", "food truck" and "café". '
                # '"Sushi", "pizza", "tacos" and "burgers" are named cuisine entities. '
                # 'A named cuisine entity must not have trailing "place" or "restaurant". '
                'Starting Cuisine adjectives such as "authentic" and "classic" should be a part of the Cuisine entity.\n'
                # 'Trailing "food" should not be a part of a Cuisine entity if not necessary, such as "vegan" as opposed to "vegan food". '
                'You should drop trailing "food", "place", "restaurant" words from named cuisine entities if not necessary.\n'
                'Ensure to distinguish named amenity entities from named hours entities. '
                # 'Ensure to distinguish named amenity entities from named hours entities and named dish entities. '
                'Meal times such as "breakfast", "brunch", "lunch" and "dinner" should be named Hours entities. '
                # 'A named dish entity must be a specific food item or meal.'  # doesn't help
                # 'A named amenity entity must be feature or service offered by a restaurant. '
            ),
            demos=[
                # dict(sentence="I'm looking for a pizza place in the area", entity_span='pizza place', label=WRONG_SPAN, correct_span='pizza'),
                # dict(sentence="Can you recommend a dessert place that's open late?", entity_span="dessert place", label=WRONG_SPAN, correct_span='dessert'),
                # dict(sentence="What is the best Mexican restaurant in this area?", entity_span="Mexican", label=CORRECT),
                dict(sentence='What time does the Italian restaurant on Main Street close tonight?', entity_span="Italian restaurant", label=WRONG_SPAN, correct_span='Italian'),
                dict(sentence='I want to try some authentic Mexican food, any recommendations?', entity_span="authentic Mexican food", label=WRONG_SPAN, correct_span='authentic Mexican'),
                dict(sentence='Find me a burger joint with a drive-thru.', entity_span="burger joint", label=CORRECT),
                # dict(sentence='I need to make a reservation at the French bistro', entity_span="French bistro", label=CORRECT),
                dict(sentence='show me a budget-friendly café with vegan options', entity_span="vegan options", label=WRONG_SPAN, correct_span='vegan'),
                dict(sentence='Is there a 24-hour restaurant nearby that offers a breakfast buffet?', entity_span="breakfast", label=WRONG_TYPE, correct_type='Hours'),
            ]
        ),
        'Dish': dict(
            defn=(
                'A named dish entity is a specific food item or meal.\n'
                'Ensure to distinguish a named dish entity from a named cuisine entity based on the context. '
                'Reference to a broader category of food should be a named Cuisine entity.'
                # 'General types of food such as "breakfast", "lunch" and "dinner" should be named Hours entities. '
                # 'General types of eateries such as "diner", "cafe" and "fast food" should be named Cuisine entities.'
            ),
            demos=[
                dict(sentence='What is the best burger joint in town?', entity_span="burger", label=WRONG_TYPE, correct_type='Cuisine'),
                # dict(sentence="I'm craving some authentic [sushi], where should I go?", entity_span="sushi", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence='Recommend a place to get a good burger.', entity_span="burger", label=CORRECT),
                dict(sentence="I want to try a new dish, what's popular at the Thai restaurant?", entity_span="new dish", label=NA),
                dict(sentence='Where can I find a late-night diner that serves breakfast? ', entity_span="breakfast", label=WRONG_TYPE, correct_type='Hours'),
            ]
        ),
        'Hours': dict(
            defn=(
                'Any temporal reference to a specific time of day or day of the week is a named Hours entity.\n'
                'A named hours entity must not contain irrelevant terms such as "diner" or "reservation".\n'
                'Days of the week such as "Friday" and "Wednesday" are named Hours entities. '
                'Meal times such as "breakfast", "brunch", "lunch" and "dinner" are also named Hours entities.\n'
                'Ambiguous identifiers such as "hours-of-operation" and "operating hours" are not named entities.'
            ),
            equivalents=['time'],
            demos=[
                dict(sentence='I need to find a 24-hour diner for late-night cravings.', entity_span='24-hour diner', label=WRONG_SPAN, correct_span='24-hour'),
                dict(sentence='I need to book a table at Olive Garden for 7 pm tonight.', entity_span='7 pm tonight', label=CORRECT),
                dict(sentence="I'm looking for a quick lunch spot with vegetarian options", entity_span="lunch", label=CORRECT),
                dict(sentence='Is there a place around here that serves late-night tacos?', entity_span="late-night", label=CORRECT),
                dict(sentence='Call Olive Garden and make a reservation for 7:00 pm.', entity_span='reservation for 7:00 pm', label=WRONG_SPAN, correct_span='7:00 pm'),
                dict(sentence='What are the operating hours of the nearest burger joint?', entity_span='operating hours', label=NA),
            ]
        ),
        'Location': dict(
            defn=(
                'A named location entity refers to the geographical location or relative proximity of the restaurant. '
                'Proximity adjectives such as "nearest" and "closest" are also named location entities. '
                'A named location entity should include all relevant terms such as "in this city" as opposed to "city". '
                # 'General mentions such as "place", "spot" and "address" are not named entities.'
                # 'General mentions such as "place", "spot" and "address" are not relevant named entities.'
            ),
            demos=[
                dict(sentence="What's the best place for a Sunday brunch in this city?", entity_span='city', label=WRONG_SPAN, correct_span='in this city'),
                dict(sentence="I'm craving some ice cream. Any good dessert places near me?", entity_span='near me', label=CORRECT),
                dict(sentence='What are the hours for the nearest sushi bar?', entity_span='nearest', label=CORRECT),
                dict(sentence="I'm looking for a cheap place to eat near downtown", entity_span='downtown', label=WRONG_SPAN, correct_span='near downtown'),
                dict(sentence='Are there any halal food trucks in the downtown area?', entity_span='downtown area', label=WRONG_SPAN, correct_span='in the downtown area'),
                dict(sentence='Can you help me locate a place that serves gluten-free options?', entity_span='place', label=NA),
                # dict(sentence='Can you help me locate a place that serves gluten-free options?', entity_span='place', label=WRONG_TYPE, correct_type='other'),
                # dict(sentence='I need to know the address of the nearest Starbucks', entity_span='address', label=NA),
                # dict(sentence='I need to know the address of the nearest Starbucks', entity_span='address', label=WRONG_TYPE, correct_type='other'),
            ]
        ),
        'Price': dict(
            defn=(
                'A named price entity is a specific monetary value or range such as "under $10 per person". '
                'Cost-related adjectives such as "cheap", "affordable", "reasonable" and "upscale" are also named price entities.\n'
                'Named price entities should not have trailing terms such as "prices" and "eats".\n'
                'General mentions of price such as "price range" and "prices" are not named entities.'
            ),
            demos=[
                dict(sentence='Any all-you-can-eat buffets in the downtown area?', entity_span='all-you-can-eat', label=WRONG_TYPE, correct_type='Amenity'),
                dict(sentence="What's the price range for the steakhouse downtown", entity_span="price range", label=NA),
                dict(sentence="How much does the average meal cost at Olive Garden", entity_span="average meal cost", label=NA),
                dict(sentence='Where can I get a good slice of pizza for under $5?', entity_span='under $5', label=CORRECT),
                # dict(sentence='Is there a 24-hour diner in the area with affordable prices? ', entity_span='affordable prices', label=WRONG_SPAN, correct_span='affordable'),
                dict(sentence='Is there a steakhouse nearby with a reasonable price?', entity_span='reasonable price', label=WRONG_SPAN, correct_span='reasonable'),
                # dict(sentence="Where's the cheapest place for a quick bite to eat?", entity_span="cheapest", label=CORRECT),
                dict(sentence="I'm looking for a place that offers a prix fixe menu.", entity_span="prix fixe", label=WRONG_TYPE, correct_type='Amenity'),
            ]
        ),
        'Rating': dict(
            defn=(
                'A named rating entity is a specific rating score such as "5 star". '
                'Qualitative descriptions such as "good", "best", "delicious" and "popular" are also named rating entities.\n'
                'You should capture the complete descriptive phrase such as "best reviewed" as opposed to "best".'
            ),
            demos=[
                dict(sentence='I want to try some Caribbean cuisine, where can I find a good place?', entity_span='good', label=CORRECT),
                dict(sentence='recommend a family-friendly Italian place with a kids menu', entity_span='family-friendly', label=WRONG_TYPE, correct_type='Amenity'),
                dict(sentence='Find me a steakhouse with a 5-star rating', entity_span='5-star', label=WRONG_SPAN, correct_span='5 star rating'),
                dict(sentence='Find me a restaurant with a 5-star rating for dinner tonight.', entity_span='5-star', label=WRONG_SPAN, correct_span='5 star rating'),
                dict(sentence='What is the rating of the nearest Italian restaurant?', entity_span='rating', label=NA),
                dict(sentence='Can you recommend a restaurant with a high Zagat rating?', entity_span='high', label=WRONG_SPAN, correct_span='high Zagat rating'),
            ]
        )
    },
    '24-02-12_NER-Dataset_{fmt=n-p2,#l=3,de=T,lc=T}_add-super-idx': {
        'Restaurant Name': dict(
            defn=(
                'A restaurant name entity must be the name of a restaurant.\n'
                'Ensure to distinguish restaurant name entities from a named cuisine entities. '
                'A named cuisine entity is a style or type of cooking, such as "diner", "burger", "food truck" and "café".'
            ),
            name='restaurant name',
            name_full='restaurant name entity',  # override default templating, i.e. `named {entity_type} entity`
            demos=[
                dict(sentence="Can I get a table for two at Ruth's Chris Steak House tonight?", entity_span="Ruth's Chris Steak House", label=CORRECT),
                dict(sentence="What's the average cost of a meal at the steakhouse downtown?", entity_span="steakhouse", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence="Which restaurant serves the best brunch in the city?", entity_span="restaurant", label=NA),
            ]
        ),
        'Amenity': dict(
            defn=(
                'A named amenity entity is a feature or service offered by a restaurant. '
                'Descriptions on the atmosphere and ambiance of a restaurant such as "fancy", "romantic" and "trendy" are also named amenity entities.\n'
                'Ensure to distinguish named amenity entities from named hours entities and named cuisine entities. '
                'Meal times such as "breakfast", "brunch", "lunch" and "dinner" should be named Hours entities. '
                'A named cuisine entity is a style or type of cooking, such as "diner", "burger", "food truck" and "café".\n'
                'Review platforms such as "Yelp", "Time Out" and "Foodies\' Choice" are not relevant named entities. '
                'Vague restaurant identifiers such as "restaurant" are also not named entities.'
            ),
            demos=[
                dict(sentence='Is there a place nearby that has a good happy hour deal?', entity_span='happy hour', label=CORRECT),
                dict(sentence="Where's the best place to get a big breakfast in this town?", entity_span="breakfast", label=WRONG_TYPE, correct_type='Hours'),
                dict(sentence="Is there a vegan-friendly cafe with gluten-free options?", entity_span="cafe", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence='Can you recommend a food truck with Korean BBQ?', entity_span="food truck", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence="What is the Foodies' Choice for Italian cuisine in the city center?", entity_span="Foodies' Choice", label=WRONG_TYPE, correct_type='other'),
                dict(sentence='Can you recommend a restaurant with great calamari?', entity_span='restaurant', label=NA),
            ]
        ),
        'Cuisine': dict(
            defn=(
                'A named cuisine entity is a style or type of cooking, such as "diner", "burger", "food truck" and "café". '
                'Starting Cuisine adjectives such as "authentic" and "classic" should be a part of the Cuisine entity.\n'
                'You should drop trailing "food", "place", "restaurant" words from named cuisine entities if not necessary.\n'
                'Ensure to distinguish named amenity entities from named hours entities. '
                'Meal times such as "breakfast", "brunch", "lunch" and "dinner" should be named Hours entities. '
            ),
            demos=[
                dict(sentence='What time does the Italian restaurant on Main Street close tonight?', entity_span="Italian restaurant", label=WRONG_SPAN, correct_span='Italian'),
                dict(sentence='I want to try some authentic Mexican food, any recommendations?', entity_span="authentic Mexican food", label=WRONG_SPAN, correct_span='authentic Mexican'),
                dict(sentence='Find me a burger joint with a drive-thru.', entity_span="burger joint", label=CORRECT),
                dict(sentence='show me a budget-friendly café with vegan options', entity_span="vegan options", label=WRONG_SPAN, correct_span='vegan'),
                dict(sentence='Is there a 24-hour restaurant nearby that offers a breakfast buffet?', entity_span="breakfast", label=WRONG_TYPE, correct_type='Hours'),
            ]
        ),
        'Dish': dict(
            defn=(
                'A named dish entity is a specific food item or meal.\n'
                'Ensure to distinguish a named dish entity from a named cuisine entity based on the context. '
                'Reference to a broader category of food should be a named Cuisine entity.\n'
                # 'A named cuisine entity is a style or type of cooking, such as "diner", "burger", "food truck" and "café".\n'
                'Generic dish descriptors such as "new dish" and "dishes" are not named entities.'
            ),
            demos=[
                dict(sentence='What is the best burger joint in town?', entity_span="burger", label=WRONG_TYPE, correct_type='Cuisine'),
                # dict(sentence="I'm craving some authentic sushi, where should I go?", entity_span="sushi", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence='Recommend a place to get a good burger.', entity_span="burger", label=CORRECT),
                dict(sentence="I want to try a new dish, what's popular at the Thai restaurant?", entity_span="new dish", label=NA),
                dict(sentence='Where can I find a late-night diner that serves breakfast? ', entity_span="breakfast", label=WRONG_TYPE, correct_type='Hours'),
            ]
        ),
        'Hours': dict(
            defn=(
                'Any temporal reference to a specific time of day or day of the week is a named Hours entity.\n'
                # 'A named Hours entity must be a temporal reference to a specific time of day or day of the week.\n'
                # 'A named Hours entity must be a specific time of day or day of the week.\n'
                'A named Hours entity must not contain irrelevant terms such as "diner" or "reservation".\n'
                'Days of the week such as "Friday" and "Wednesday" are named Hours entities. '
                'Meal times such as "breakfast", "brunch", "lunch" and "dinner" are also named Hours entities.\n'
                'Ambiguous identifiers such as "hours-of-operation", "hours" and "operating hours" are not named entities.'
            ),
            equivalents=['time'],
            demos=[
                dict(sentence='I need to find a 24-hour diner for late-night cravings.', entity_span='24-hour diner', label=WRONG_SPAN, correct_span='24-hour'),
                dict(sentence='I need to book a table at Olive Garden for 7 pm tonight.', entity_span='7 pm tonight', label=CORRECT),
                dict(sentence="I'm looking for a quick lunch spot with vegetarian options", entity_span="lunch", label=CORRECT),
                dict(sentence='Is there a place around here that serves late-night tacos?', entity_span="late-night", label=CORRECT),
                dict(sentence='Call Olive Garden and make a reservation for 7:00 pm.', entity_span='reservation for 7:00 pm', label=WRONG_SPAN, correct_span='7:00 pm'),
                dict(sentence='What are the operating hours of the nearest burger joint?', entity_span='operating hours', label=NA),
            ]
        ),
        'Location': dict(
            defn=(
                'A named location entity refers to the geographical location or relative proximity of the restaurant. '
                'Proximity adjectives such as "nearest", "nearby" and "closest" are also named location entities.\n'
                'A named location entity should include all relevant terms such as "in this city" as opposed to "city". '
                # 'General mentions such as "place", "spot" and "address" are not named entities.'
                # 'General mentions such as "place", "spot" and "address" are not relevant named entities.'
            ),
            equivalents=['Proximity'],
            demos=[
                dict(sentence="What's the best place for a Sunday brunch in this city?", entity_span='city', label=WRONG_SPAN, correct_span='in this city'),
                dict(sentence="I'm craving some ice cream. Any good dessert places near me?", entity_span='near me', label=CORRECT),
                dict(sentence='What are the hours for the nearest sushi bar?', entity_span='nearest', label=CORRECT),
                dict(sentence="I'm looking for a cheap place to eat near downtown", entity_span='downtown', label=WRONG_SPAN, correct_span='near downtown'),
                # dict(sentence='Are there any halal food trucks in the downtown area?', entity_span='downtown area', label=WRONG_SPAN, correct_span='in the downtown area'),
                dict(sentence='Is there a Western Living-themed restaurant in the area?', entity_span='area', label=WRONG_SPAN, correct_span='in the area'),
                dict(sentence='Can you help me locate a place that serves gluten-free options?', entity_span='place', label=NA),
            ]
        ),
        'Price': dict(
            defn=(
                'A named price entity is a specific monetary value or range such as "under $10 per person". '
                'Cost-related adjectives such as "cheap", "affordable", "reasonable" and "upscale" are also named price entities.\n'
                'Named price entities should not have trailing terms such as "prices" and "eats".\n'
                'General mentions of price such as "price range" and "prices" are not named entities.'
            ),
            demos=[
                dict(sentence='Any all-you-can-eat buffets in the downtown area?', entity_span='all-you-can-eat', label=WRONG_TYPE, correct_type='Amenity'),
                dict(sentence="What's the price range for the steakhouse downtown", entity_span="price range", label=NA),
                dict(sentence="How much does the average meal cost at Olive Garden", entity_span="average meal cost", label=NA),
                dict(sentence='Where can I get a good slice of pizza for under $5?', entity_span='under $5', label=CORRECT),
                dict(sentence='Is there a steakhouse nearby with a reasonable price?', entity_span='reasonable price', label=WRONG_SPAN, correct_span='reasonable'),
                dict(sentence="I'm looking for a place that offers a prix fixe menu.", entity_span="prix fixe", label=WRONG_TYPE, correct_type='Amenity'),
            ]
        ),
        'Rating': dict(
            defn=(
                'A named rating entity is a specific rating score such as "5 star". '
                'Qualitative descriptions such as "good", "best", "high", "delicious" and "popular" are also named rating entities. '
                'Rating accolades such as "high-rated," "best ratings," and "best in town" are also named rating entities.\n'
                'You should capture the complete descriptive phrase such as "best reviewed" as opposed to "best".'
            ),
            equivalents=['quality'],
            demos=[
                dict(sentence='I want to try some Caribbean cuisine, where can I find a good place?', entity_span='good', label=CORRECT),
                # dict(sentence='recommend a family-friendly Italian place with a kids menu', entity_span='family-friendly', label=WRONG_TYPE, correct_type='Amenity'),
                dict(sentence='Find me a steakhouse with a 5-star rating', entity_span='5-star', label=WRONG_SPAN, correct_span='5 star rating'),
                # dict(sentence='Find me a restaurant with a 5-star rating for dinner tonight.', entity_span='5-star', label=WRONG_SPAN, correct_span='5 star rating'),
                dict(sentence='What is the rating of the nearest Italian restaurant?', entity_span='rating', label=NA),
                dict(sentence='Can you recommend a restaurant with a high Zagat rating?', entity_span='high', label=WRONG_SPAN, correct_span='high Zagat rating'),
            ]
        )
    },
    '24-02-21_NER-Dataset_{fmt=n-p2,#l=3,de=s,lc=T}_add-super-idx': {
        'Restaurant Name': dict(
            defn=(
                'A restaurant name entity must be the name of a restaurant.\n'
                'Ensure to distinguish restaurant name entities from a named cuisine entities. '
                'A named cuisine entity is a style or type of cooking, such as "diner", "burger", "food truck" and "café".'
            ),
            name='restaurant name',
            name_full='restaurant name entity',  # override default templating, i.e. `named {entity_type} entity`
            demos=[
                dict(sentence="Can I get a table for two at Ruth's Chris Steak House tonight?", entity_span="Ruth's Chris Steak House", label=CORRECT),
                dict(sentence="What's the average cost of a meal at the steakhouse downtown?", entity_span="steakhouse", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence="Which restaurant serves the best brunch in the city?", entity_span="restaurant", label=NA),
            ]
        ),
        'Amenity': dict(
            defn=(
                'A named amenity entity is a feature or service offered by a restaurant. '
                'Descriptions on the atmosphere and ambiance of a restaurant such as "fancy", "romantic" and "trendy" are also named amenity entities.\n'
                'Ensure to distinguish named amenity entities from named hours entities and named cuisine entities. '
                'Meal times such as "breakfast", "brunch", "lunch" and "dinner" should be named Hours entities. '
                'A named cuisine entity is a style or type of cooking, such as "diner", "burger", "food truck" and "café".\n'
                'Review platforms such as "Yelp", "Time Out" and "Foodies\' Choice" are not relevant named entities. '
                'Vague restaurant identifiers such as "restaurant" are also not named entities.'
            ),
            demos=[
                dict(sentence='Is there a place nearby that has a good happy hour deal?', entity_span='happy hour', label=CORRECT),
                dict(sentence="Where's the best place to get a big breakfast in this town?", entity_span="breakfast", label=WRONG_TYPE, correct_type='Hours'),
                dict(sentence="Is there a vegan-friendly cafe with gluten-free options?", entity_span="cafe", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence='Can you recommend a food truck with Korean BBQ?', entity_span="food truck", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence="What is the Foodies' Choice for Italian cuisine in the city center?", entity_span="Foodies' Choice", label=WRONG_TYPE, correct_type='other'),
                dict(sentence='Can you recommend a restaurant with great calamari?', entity_span='restaurant', label=NA),
            ]
        ),
        'Cuisine': dict(
            defn=(
                'A named cuisine entity is a style or type of cooking, such as "diner", "burger", "food truck", "café" and "vegetarian". '
                'Starting Cuisine adjectives such as "authentic" and "classic" should be a part of the Cuisine entity.\n'
                'You should drop trailing "food", "place", "restaurant" and "options" words from named cuisine entities if not necessary.\n'
                'Ensure to distinguish named amenity entities from named hours entities. '
                'Meal times such as "breakfast", "brunch", "lunch" and "dinner" should be named Hours entities. '
            ),
            demos=[
                dict(sentence='What time does the Italian restaurant on Main Street close tonight?', entity_span="Italian restaurant", label=WRONG_SPAN, correct_span='Italian'),
                dict(sentence='I want to try some authentic Mexican food, any recommendations?', entity_span="authentic Mexican food", label=WRONG_SPAN, correct_span='authentic Mexican'),
                dict(sentence='Find me a burger joint with a drive-thru.', entity_span="burger joint", label=CORRECT),
                dict(sentence='show me a budget-friendly café with vegan options', entity_span="vegan options", label=WRONG_SPAN, correct_span='vegan'),
                dict(sentence='Is there a 24-hour restaurant nearby that offers a breakfast buffet?', entity_span="breakfast", label=WRONG_TYPE, correct_type='Hours'),
            ]
        ),
        'Dish': dict(
            defn=(
                'A named dish entity is a specific food item or meal.\n'
                'Ensure to distinguish a named dish entity from a named cuisine entity based on the context. '
                'Reference to a broader category of food should be a named Cuisine entity.\n'
                # 'A named cuisine entity is a style or type of cooking, such as "diner", "burger", "food truck" and "café".\n'
                'Generic dish descriptors such as "new dish" and "dishes" are not named entities.'
            ),
            demos=[
                dict(sentence='What is the best burger joint in town?', entity_span="burger", label=WRONG_TYPE, correct_type='Cuisine'),
                # dict(sentence="I'm craving some authentic sushi, where should I go?", entity_span="sushi", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence='Recommend a place to get a good burger.', entity_span="burger", label=CORRECT),
                dict(sentence="I want to try a new dish, what's popular at the Thai restaurant?", entity_span="new dish", label=NA),
                dict(sentence='Where can I find a late-night diner that serves breakfast? ', entity_span="breakfast", label=WRONG_TYPE, correct_type='Hours'),
            ]
        ),
        'Hours': dict(
            defn=(
                # 'Any temporal reference to a specific time of day or day of the week is a named Hours entity.\n'
                'Any temporal reference to a specific time of day or day of the week is a named Hours entity. '
                'For example, "10PM", "Friday" and "late-night" are all named Hours entities.\n'
                # 'A named Hours entity must be a temporal reference to a specific time of day or day of the week.\n'
                # 'A named Hours entity must be a specific time of day or day of the week.\n'
                'A named Hours entity must not contain irrelevant terms such as "diner" or "reservation".\n'
                'Days of the week such as "Friday" and "Wednesday" are named Hours entities. '
                'Meal times such as "breakfast", "brunch", "lunch" and "dinner" are also named Hours entities.\n'
                # 'Ambiguous identifiers such as "hours-of-operation", "hours" and "operating hours" are not named entities.'
                'Generic hours descriptors such as "hours-of-operation", "hours", "opening hours" and "operating hours" are not named entities.'
            ),
            equivalents=['time'],
            demos=[
                dict(sentence='I need to find a 24-hour diner for late-night cravings.', entity_span='24-hour diner', label=WRONG_SPAN, correct_span='24-hour'),
                dict(sentence='I need to book a table at Olive Garden for 7 pm tonight.', entity_span='7 pm tonight', label=CORRECT),
                dict(sentence="I'm looking for a quick lunch spot with vegetarian options", entity_span="lunch", label=CORRECT),
                dict(sentence='Is there a place around here that serves late-night tacos?', entity_span="late-night", label=CORRECT),
                dict(sentence='Call Olive Garden and make a reservation for 7:00 pm.', entity_span='reservation for 7:00 pm', label=WRONG_SPAN, correct_span='7:00 pm'),
                dict(sentence='What are the operating hours of the nearest burger joint?', entity_span='operating hours', label=NA),
            ]
        ),
        'Location': dict(
            defn=(
                'A named location entity refers to the geographical location or relative proximity of the restaurant. '
                'Proximity adjectives such as "nearest", "nearby" and "closest" are also named location entities.\n'
                'A named location entity should include all relevant terms such as "in this city" as opposed to "city". '
                # 'General mentions such as "place", "spot" and "address" are not named entities.'
                # 'General mentions such as "place", "spot" and "address" are not relevant named entities.'
            ),
            equivalents=['Proximity'],
            demos=[
                dict(sentence="What's the best place for a Sunday brunch in this city?", entity_span='city', label=WRONG_SPAN, correct_span='in this city'),
                dict(sentence="I'm craving some ice cream. Any good dessert places near me?", entity_span='near me', label=CORRECT),
                dict(sentence='What are the hours for the nearest sushi bar?', entity_span='nearest', label=CORRECT),
                dict(sentence="I'm looking for a cheap place to eat near downtown", entity_span='downtown', label=WRONG_SPAN, correct_span='near downtown'),
                # dict(sentence='Are there any halal food trucks in the downtown area?', entity_span='downtown area', label=WRONG_SPAN, correct_span='in the downtown area'),
                dict(sentence='Is there a Western Living-themed restaurant in the area?', entity_span='area', label=WRONG_SPAN, correct_span='in the area'),
                dict(sentence='Can you help me locate a place that serves gluten-free options?', entity_span='place', label=NA),
            ]
        ),
        'Price': dict(
            defn=(
                'A named price entity is a specific monetary value or range such as "under $10 per person". '
                'Cost-related adjectives such as "cheap", "affordable", "reasonable", "expensive" and "upscale" are also named price entities. '
                # 'Price category descriptors such as " '
                # 'Named price entities should not have trailing terms such as "prices" and "eats".\n'
                # 'You should always drop trailing "prices", "price range" and "eats" words from named price entities.\n'
                'You should drop trailing "prices", "price range" and "eats" words from named price entities if not necessary, such as "moderately" as opposed to "moderately priced".\n'
                'General mentions of price such as "price range" and "prices" are not named entities.'
            ),
            demos=[
                # dict(sentence='Any all-you-can-eat buffets in the downtown area?', entity_span='all-you-can-eat', label=WRONG_TYPE, correct_type='Amenity'),
                # dict(sentence="What's the price range for the steakhouse downtown", entity_span="price range", label=NA),
                # dict(sentence="How much does the average meal cost at Olive Garden", entity_span="average meal cost", label=NA),
                dict(sentence='Where can I get a good slice of pizza for under $5?', entity_span='under $5', label=CORRECT),
                dict(sentence='Can you recommend a budget-friendly Italian restaurant in the area?', entity_span='budget-friendly', label=CORRECT),
                dict(sentence='Is there a steakhouse nearby with a reasonable price?', entity_span='reasonable price', label=WRONG_SPAN, correct_span='reasonable'),
                dict(sentence="I'm looking for a place that offers a prix fixe menu.", entity_span="prix fixe", label=WRONG_TYPE, correct_type='Amenity'),
            ]
        ),
        'Rating': dict(
            defn=(
                'A named rating entity is a specific rating score such as "5 star". '
                'Qualitative descriptions such as "good", "best", "high", "delicious" and "popular" are also named rating entities. '
                'Rating accolades such as "high-rated," "best ratings," and "best in town" are also named rating entities.\n'
                'You should capture the complete descriptive phrase such as "best reviewed" as opposed to "best".'
            ),
            equivalents=['quality'],
            demos=[
                dict(sentence='I want to try some Caribbean cuisine, where can I find a good place?', entity_span='good', label=CORRECT),
                # dict(sentence='recommend a family-friendly Italian place with a kids menu', entity_span='family-friendly', label=WRONG_TYPE, correct_type='Amenity'),
                dict(sentence='Find me a steakhouse with a 5-star rating', entity_span='5-star', label=WRONG_SPAN, correct_span='5 star rating'),
                # dict(sentence='Find me a restaurant with a 5-star rating for dinner tonight.', entity_span='5-star', label=WRONG_SPAN, correct_span='5 star rating'),
                dict(sentence='What is the rating of the nearest Italian restaurant?', entity_span='rating', label=NA),
                dict(sentence='Can you recommend a restaurant with a high Zagat rating?', entity_span='high', label=WRONG_SPAN, correct_span='high Zagat rating'),
            ]
        )
    },
    '24-02-08_NER-Dataset_{fmt=n-p2,#l=3,ap={dc=T,de=s},lc=T}': {
        'Restaurant Name': dict(
            defn=(
                'A restaurant name entity must be the name of a restaurant.\n'
                'Ensure to distinguish restaurant name entities from a named cuisine entities. '
                'A named cuisine entity is a style or type of cooking, such as "diner", "burger", "food truck" and "café".'
            ),
            name='restaurant name',
            name_full='restaurant name entity',  # override default templating, i.e. `named {entity_type} entity`
            demos=[
                dict(sentence="Can I get a table for two at Ruth's Chris Steak House tonight?", entity_span="Ruth's Chris Steak House", label=CORRECT),
                dict(sentence="What's the average cost of a meal at the steakhouse downtown?", entity_span="steakhouse", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence="Which restaurant serves the best brunch in the city?", entity_span="restaurant", label=NA),
            ]
        ),
        'Amenity': dict(
            defn=(
                'A named amenity entity is a feature or service offered by a restaurant. '
                'Descriptions on the atmosphere and ambiance of a restaurant such as "fancy", "romantic" and "trendy" are also named amenity entities.\n'
                'Ensure to distinguish named amenity entities from named hours entities and named cuisine entities. '
                'Meal times such as "breakfast", "brunch", "lunch" and "dinner" should be named Hours entities. '
                'A named cuisine entity is a style or type of cooking, such as "diner", "burger", "food truck" and "café".\n'
                'Review platforms such as "Yelp", "Time Out" and "Foodies\' Choice" are not relevant named entities. '
                'Vague restaurant identifiers such as "restaurant" are also not named entities.'
            ),
            demos=[
                dict(sentence='Is there a place nearby that has a good happy hour deal?', entity_span='happy hour', label=CORRECT),
                dict(sentence="Where's the best place to get a big breakfast in this town?", entity_span="breakfast", label=WRONG_TYPE, correct_type='Hours'),
                dict(sentence="Is there a vegan-friendly cafe with gluten-free options?", entity_span="cafe", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence='Can you recommend a food truck with Korean BBQ?', entity_span="food truck", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence="What is the Foodies' Choice for Italian cuisine in the city center?", entity_span="Foodies' Choice", label=WRONG_TYPE, correct_type='other'),
                dict(sentence='Can you recommend a restaurant with great calamari?', entity_span='restaurant', label=NA),
            ]
        ),
        'Cuisine': dict(
            defn=(
                'A named cuisine entity is a style or type of cooking, such as "diner", "burger", "food truck", "café" and "vegetarian". '
                'Starting Cuisine adjectives such as "authentic" and "classic" should be a part of the Cuisine entity.\n'
                'You should drop trailing "food", "place", "restaurant" and "options" words from named cuisine entities if not necessary.\n'
                'Ensure to distinguish named amenity entities from named hours entities. '
                'Meal times such as "breakfast", "brunch", "lunch" and "dinner" should be named Hours entities. '
            ),
            demos=[
                dict(sentence='What time does the Italian restaurant on Main Street close tonight?', entity_span="Italian restaurant", label=WRONG_SPAN, correct_span='Italian'),
                dict(sentence='I want to try some authentic Mexican food, any recommendations?', entity_span="authentic Mexican food", label=WRONG_SPAN, correct_span='authentic Mexican'),
                dict(sentence='Find me a burger joint with a drive-thru.', entity_span="burger joint", label=CORRECT),
                dict(sentence='show me a budget-friendly café with vegan options', entity_span="vegan options", label=WRONG_SPAN, correct_span='vegan'),
                dict(sentence='Is there a 24-hour restaurant nearby that offers a breakfast buffet?', entity_span="breakfast", label=WRONG_TYPE, correct_type='Hours'),
            ]
        ),
        'Dish': dict(
            defn=(
                'A named dish entity is a specific food item or meal.\n'
                'Ensure to distinguish a named dish entity from a named cuisine entity based on the context. '
                'Reference to a broader category of food should be a named Cuisine entity.\n'
                # 'A named cuisine entity is a style or type of cooking, such as "diner", "burger", "food truck" and "café".\n'
                'Generic dish descriptors such as "new dish" and "dishes" are not named entities.'
            ),
            demos=[
                dict(sentence='What is the best burger joint in town?', entity_span="burger", label=WRONG_TYPE, correct_type='Cuisine'),
                # dict(sentence="I'm craving some authentic sushi, where should I go?", entity_span="sushi", label=WRONG_TYPE, correct_type='Cuisine'),
                dict(sentence='Recommend a place to get a good burger.', entity_span="burger", label=CORRECT),
                dict(sentence="I want to try a new dish, what's popular at the Thai restaurant?", entity_span="new dish", label=NA),
                dict(sentence='Where can I find a late-night diner that serves breakfast? ', entity_span="breakfast", label=WRONG_TYPE, correct_type='Hours'),
            ]
        ),
        'Hours': dict(
            defn=(
                # 'Any temporal reference to a specific time of day or day of the week is a named Hours entity.\n'
                'Any temporal reference to a specific time of day or day of the week is a named Hours entity. '
                'For example, "10PM", "Friday" and "late-night" are all named Hours entities.\n'
                # 'A named Hours entity must be a temporal reference to a specific time of day or day of the week.\n'
                # 'A named Hours entity must be a specific time of day or day of the week.\n'
                'A named Hours entity must not contain irrelevant terms such as "diner" or "reservation".\n'
                'Days of the week such as "Friday" and "Wednesday" are named Hours entities. '
                'Meal times such as "breakfast", "brunch", "lunch" and "dinner" are also named Hours entities.\n'
                # 'Ambiguous identifiers such as "hours-of-operation", "hours" and "operating hours" are not named entities.'
                'Generic hours descriptors such as "hours-of-operation", "hours", "opening hours" and "operating hours" are not named entities.'
            ),
            equivalents=['time', 'meal time'],
            demos=[
                dict(sentence='I need to find a 24-hour diner for late-night cravings.', entity_span='24-hour diner', label=WRONG_SPAN, correct_span='24-hour'),
                dict(sentence='I need to book a table at Olive Garden for 7 pm tonight.', entity_span='7 pm tonight', label=CORRECT),
                dict(sentence="I'm looking for a quick lunch spot with vegetarian options", entity_span="lunch", label=CORRECT),
                dict(sentence='Is there a place around here that serves late-night tacos?', entity_span="late-night", label=CORRECT),
                dict(sentence='Call Olive Garden and make a reservation for 7:00 pm.', entity_span='reservation for 7:00 pm', label=WRONG_SPAN, correct_span='7:00 pm'),
                dict(sentence='What are the operating hours of the nearest burger joint?', entity_span='operating hours', label=NA),
            ]
        ),
        'Location': dict(
            defn=(
                'A named location entity refers to the geographical location or relative proximity of the restaurant. '
                # 'Proximity adjectives such as "nearest", "nearby", "closest" and "local" are also named location entities. '
                'Proximity descriptors such as "nearest", "nearby", "closest" and "local" are also named location entities.\n'
                'Longer proximity descriptors such as "adjacent to a bar" are also named location entities. '
                'A named location entity should include all relevant terms such as "in this city" as opposed to "city".'
                # 'General mentions such as "place", "spot" and "address" are not named entities.'
                # 'General mentions such as "place", "spot" and "address" are not relevant named entities.'
            ),
            equivalents=['Proximity', 'Proximity Descriptor'],
            demos=[
                dict(sentence="What's the best place for a Sunday brunch in this city?", entity_span='city', label=WRONG_SPAN, correct_span='in this city'),
                dict(sentence="I'm craving some ice cream. Any good dessert places near me?", entity_span='near me', label=CORRECT),
                dict(sentence='What are the hours for the nearest sushi bar?', entity_span='nearest', label=CORRECT),
                dict(sentence="I'm looking for a cheap place to eat near downtown", entity_span='downtown', label=WRONG_SPAN, correct_span='near downtown'),
                # dict(sentence='Are there any halal food trucks in the downtown area?', entity_span='downtown area', label=WRONG_SPAN, correct_span='in the downtown area'),
                dict(sentence='Is there a Western Living-themed restaurant in the area?', entity_span='area', label=WRONG_SPAN, correct_span='in the area'),
                dict(sentence='Can you help me locate a place that serves gluten-free options?', entity_span='place', label=NA),
            ]
        ),
        'Price': dict(
            defn=(
                'A named price entity is a specific monetary value or range such as "under $10 per person". '
                'Cost-related adjectives such as "cheap", "affordable", "reasonable", "expensive" and "upscale" are also named price entities. '
                # 'Price category descriptors such as " '
                # 'Named price entities should not have trailing terms such as "prices" and "eats".\n'
                # 'You should always drop trailing "prices", "price range" and "eats" words from named price entities.\n'
                'You should drop trailing "prices", "price range" and "eats" words from named price entities if not necessary, such as "moderately" as opposed to "moderately priced".\n'
                'General mentions of price such as "price range" and "prices" are not named entities.'
            ),
            demos=[
                # dict(sentence='Any all-you-can-eat buffets in the downtown area?', entity_span='all-you-can-eat', label=WRONG_TYPE, correct_type='Amenity'),
                # dict(sentence="What's the price range for the steakhouse downtown", entity_span="price range", label=NA),
                # dict(sentence="How much does the average meal cost at Olive Garden", entity_span="average meal cost", label=NA),
                dict(sentence='Where can I get a good slice of pizza for under $5?', entity_span='under $5', label=CORRECT),
                dict(sentence='Can you recommend a budget-friendly Italian restaurant in the area?', entity_span='budget-friendly', label=CORRECT),
                dict(sentence='Is there a steakhouse nearby with a reasonable price?', entity_span='reasonable price', label=WRONG_SPAN, correct_span='reasonable'),
                dict(sentence="I'm looking for a place that offers a prix fixe menu.", entity_span="prix fixe", label=WRONG_TYPE, correct_type='Amenity'),
            ]
        ),
        'Rating': dict(
            defn=(
                'A named rating entity is a specific rating score such as "5 star". '
                'Qualitative descriptions such as "good", "best", "high", "delicious" and "popular" are also named rating entities. '
                'Rating accolades such as "high-rated," "best ratings," and "best in town" are named rating entities. '
                'Awards such as "Michelin Star" are also named rating entities.\n'
                'You should capture the complete descriptive phrase such as "best reviewed" as opposed to "best". '
                'You should always drop trailing "restaurant" and "place" words from named rating entities.'
            ),
            equivalents=['quality'],
            demos=[
                dict(sentence='I want to try some Caribbean cuisine, where can I find a good place?', entity_span='good', label=CORRECT),
                # dict(sentence='recommend a family-friendly Italian place with a kids menu', entity_span='family-friendly', label=WRONG_TYPE, correct_type='Amenity'),
                dict(sentence='Find me a steakhouse with a 5-star rating', entity_span='5-star', label=WRONG_SPAN, correct_span='5 star rating'),
                # dict(sentence='Find me a restaurant with a 5-star rating for dinner tonight.', entity_span='5-star', label=WRONG_SPAN, correct_span='5 star rating'),
                dict(sentence='What is the rating of the nearest Italian restaurant?', entity_span='rating', label=NA),
                dict(sentence='Can you recommend a restaurant with a high Zagat rating?', entity_span='high', label=WRONG_SPAN, correct_span='high Zagat rating'),
                dict(sentence='which restaurants have a good reputation for their Saturday morning brunch', entity_span='good reputation', label=CORRECT),
            ]
        )
    }
}
