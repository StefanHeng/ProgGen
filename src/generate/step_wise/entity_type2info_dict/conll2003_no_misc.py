from src.generate.step_wise.entity_type2info_dict.util import *


__all__ = ['entity_type2info_dict']


entity_type2info_dict = {
    '24-02-07_NER-Dataset_{fmt=n-p2,#l=50}': {
        'person': dict(
            defn=(
                'A named person entity must be the name of a person. '
                'Only first names, last names, and full names are considered named person entities. '
                'General reference to a person or people such as "actress", "Prime Minister of India" and "CEO of Google" are not named entities. '
                # 'A named person entity should not have starting titles such as "President".'
                'A named person entity should not have any starting titles such as "President" or "Prime Minister".'
            ),
            demos=[
                dict(sentence='Russian President Vladimir Putin meets with Chinese leader Xi Jinping.', entity_span='Vladimir Putin', label=CORRECT),
                # dict(sentence='Former President Obama to release memoir next year.', entity_span='President Obama', label=WRONG_SPAN, correct_span='Obama'),
                dict(sentence='Canadian Prime Minister Justin Trudeau visits Indigenous communities.', entity_span='Prime Minister Justin Trudeau', label=WRONG_SPAN, correct_span='Justin Trudeau'),
                dict(sentence='Newly elected president pledges to address climate change.', entity_span='president', label=NA),
                dict(sentence='CEO of Amazon steps down from role.', entity_span='CEO of Amazon', label=NA),
            ]
        ),
        'location': dict(
            defn=(
                'A named location entity must be the name of a location. '
                'Adjectives like "Chinese" and "Russian" are not named location entities.'
            ),
            demos=[
                dict(sentence='The Prime Minister of Japan meets with world leaders to discuss economic cooperation.', entity_span='Japan', label=CORRECT),
                dict(sentence='Australian wildfires destroy acres of land.', entity_span='Australian', label=NA),
                dict(sentence='German Chancellor Angela Merkel meets with French President Macron.', entity_span='French', label=NA),
            ]
        ),
        'organization': dict(
            defn=(
                'A named organization entity must be the name of an organization. '
                'Adjectives such as "Chinese" and "Russian" are not named organization entities. '
                'Viruses such as "COVID-19" are also not named organization entities.'
            ),
            demos=[
                dict(sentence='Apple unveils the latest iPhone model at a product launch event.', entity_span='Apple', label=CORRECT),
                dict(sentence='CEO of Tesla Elon Musk Denies Securities Fraud Allegations', entity_span='CEO', label=NA),
                dict(sentence='Japanese automaker Toyota recalls millions of vehicles.', entity_span='Japanese', label=NA),
                dict(sentence='Pfizer announces new vaccine efficacy data against COVID-19 variants.', entity_span='COVID-19', label=NA),
            ]
        ),
    },
    '24-02-07_NER-Dataset_{fmt=n-p2,#l=3,dc=T}': {
        'person': dict(
            defn=(
                'A named person entity must be the name of a person. '
                'Only first names, last names, and full names are considered named person entities. '
                'Any general reference to a person or people such as "chef", "CEO" and "woman" is not a named entity. '
                'A named person entity should not have any starting titles such as "President" or "Mayor".'
            ),
            demos=[
                dict(sentence='local hero saves drowning child from lake.', entity_span='local hero', label=NA),
                dict(sentence='Mayor Smith unveils new community center in downtown area.', entity_span='Mayor Smith', label=WRONG_SPAN, correct_span='Smith'),
                dict(sentence='Local high school student wins national science competition.', entity_span='high school student', label=NA),
                dict(sentence='Authorities arrest three suspects in connection with bank robbery.', entity_span='suspects', label=NA)
            ]
        ),
        'location': dict(
            defn=(
                # 'A named entity must refer to an entity by name. '
                'A named location entity must be the name of a location. '
                # 'A named location entity must be the name of a location and clearly identify a specific place. '
                # 'A named location entity refers to a specific place or geographical location that has been assigned a name. '
                # 'Any general reference to a location such as "downtown", "city hall" and "bank" is not a named entity.'
                'Any general reference to a location or locations such as "city hall", "downtown", "major cities" and "bank" is not a named entity. '
                'Specific references to a location or locations not assigned a name such as "major airports" and "arts district" are also not named entities. '
                'Hurricanes and other natural disasters are not named location entities.'
                # 'Adjectives like "Chinese" and "Russian" are also not named location entities.'
            ),
            demos=[
                dict(sentence='The Prime Minister of Japan meets with world leaders to discuss economic cooperation.', entity_span='Japan', label=CORRECT),
                dict(sentence='Local high school marching band to perform at prestigious music festival.', entity_span='high school', label=NA),
                dict(sentence='New restaurant to open in downtown area, bringing jobs and economic growth.', entity_span='downtown area', label=NA),
                dict(sentence="Research reveals alarming levels of plastic pollution in the world's oceans, threatening marine life and ecosystems.", entity_span="world's oceans", label=NA)
            ]
        ),
        'organization': dict(
            defn=(
                'A named organization entity must be the name of an organization. '
                'Any general reference to an organization such as "high school", "city council" and "local community" is not a named entity. '
                'Adjectives such as "Chinese" and "European" are also not named organization entities. '
            ),
            demos=[
                dict(sentence='local charity organization provides free meals to homeless population during holiday season.', entity_span='local charity organization', label=NA),
                dict(sentence='Local non-profit organization provides free meals to homeless veterans.', entity_span='non-profit organization', label=NA),
                dict(sentence='Researchers discover new exoplanet in habitable zone.', entity_span='Researchers', label=NA),
                dict(sentence='Canadian government announces new immigration policies to attract more international students.', entity_span='Canadian', label=NA)
            ]
        ),
    },
    '24-02-07_NER-Dataset_{fmt=n-p2,#l=3,de=T}_ori-et-pool-#et=1.5': {
        'person': dict(
            defn=(
                'A named person entity must be the name of a person. '
                'Only first names, last names, and full names are considered named person entities. '
                # 'Any reference to a person or people not by name such as "mayor", "CEO" and "Prime Minister of Australia" is not a named entity. '
                # 'Any reference to a person or people not by name such as "mayor", "President of Nepal" and "Prime Minister of Australia" is not a named entity. '
                'Any reference to a person or people such as "mayor" and "CEO" is not a named entity. '

                # 'A named person entity should not have any starting titles such as "President" or "Mayor".'
                'A named person entity should not have any starting titles such as "President", "Prince" and "Professor". '
                # 'and titles such as "President of Iceland" and "Prime Minister of Australia" are not relevant named entities.'
                'Titles such as "President of Iceland" and "Prime Minister of Australia" are also not relevant named entities.'
            ),
            demos=[
                # dict(sentence='President Biden announces new infrastructure plan.', entity_span='President Biden', label=WRONG_SPAN, correct_span='Biden'),
                # dict(sentence='First Lady Melania Trump visits International Organization for Migration facility in Guatemala.', entity_span='First Lady Melania Trump', label=WRONG_SPAN, correct_span='Melania Trump'),
                dict(sentence='The CEO of Ford Motor Company predicts a surge in demand for electric cars in the next decade.', entity_span='CEO', label=NA),
                dict(sentence='South African president visits Johannesburg to address issues of poverty and inequality.', entity_span='South African president', label=NA),
                dict(sentence="India's Prime Minister to meet with President of Nepal for bilateral talks.", entity_span='President of Nepal', label=NA),
                # dict(sentence="India's Prime Minister to meet with President of Nepal for bilateral talks.", entity_span='President of Nepal', label=WRONG_SPAN, correct_span='Nepal'),
                # dict(sentence='Mayor Garcetti announces plan to address homelessness crisis in Los Angeles.', entity_span='Mayor Garcetti', label=WRONG_SPAN, correct_span='Garcetti')
                # dict(sentence='President Biden announces new infrastructure plan.', entity_span='President Biden', label=WRONG_SPAN, correct_span='Biden')
                dict(
                    sentence='President Jokowi announces new economic policies to boost growth in Jakarta.',
                    entity_span='President Jokowi', label=WRONG_SPAN, correct_span='Jokowi')
            ]
        ),
        'location': dict(
            defn=(
                'A named location entity must be the name of a location. '
                # 'Any general reference to a location or locations not by name such as "downtown" and "major cities" is not a named entity. '
                # 'Reference to a sports team by country name such as "Barcelona" should be named organization entities. '
                # 'Adjectives like "American" and "Brazilian" are not named location entities. '
                'Adjectives like "American", "Russian" and "Brazilian" are not relevant named entities. '
                # 'Hurricanes and viruses are also not relevant named entities.'
            ),
            demos=[
                # dict(sentence='The European Union member states agree on a new trade deal with South American countries.', entity_span='South American', label=NA),
                dict(sentence='European Union imposes sanctions on Belarusian officials, prompting strong response from Sergey Lavrov.', entity_span='Belarusian', label=NA),
                dict(sentence='The European Union imposes sanctions on Russian officials over human rights abuses.', entity_span='Russian', label=NA),
                # dict(sentence='Real Madrid defeats Barcelona in a thrilling El Clasico match.', entity_span='Barcelona', label=WRONG_TYPE, correct_type='organization'),
                dict(sentence='Sydney native wins gold medal in swimming competition.', entity_span='Sydney', label=CORRECT),
                dict(sentence="London Mayor Sadiq Khan responds to criticism over public transport funding.", entity_span="London", label=CORRECT)
            ]
        ),
        'organization': dict(
            defn=(
                'A named organization entity must be the name of an organization. '
                'Any general reference to an organization or organizations such as "government", "company" and "global technology company" is not a named entity. '
                'Adjectives such as "Chinese", "Australian" and "European" are also not relevant named entities. '
                'Viruses such as "COVID-19" are also not named organization entities. '
                # 'Products such as "iPhone" and "Instagram" are relevant named entities.'
                'Products such as "iPhone" and "Instagram" are not relevant named entities.'
            ),
            demos=[
                dict(sentence='The Indian government announces new policies to address air pollution in Delhi.', entity_span='Indian government', label=NA),
                dict(sentence='Major electronics company headquartered in Seoul sees 15% increase in profits.', entity_span='company', label=NA),
                dict(sentence='Local tech company to open new headquarters in San Francisco.', entity_span='tech company', label=NA),
                dict(sentence='Australian Prime Minister to visit Sydney for climate change summit.', entity_span='Australian', label=NA),
                # dict(sentence='Indian government bans TikTok and 58 other Chinese apps.', entity_span='TikTok', label=CORRECT)
                dict(sentence='The University of Washington football team defeated Stanford in a close game.', entity_span='University of Washington', label=CORRECT)
            ]
        )
    },
    '24-02-08_NER-Dataset_{fmt=n-p2,#l=3,de=s}': {
        'person': dict(
            defn=(
                'A named person entity must be the name of a person. '
                'Only first names, last names, and full names are considered named person entities. '
                'Any general reference to a person or people such as "chef", "CEO" and "Mayor" is not a named entity. '
                'A named person entity should not have any starting titles such as "President" or "Mayor".'
            ),
            demos=[
                dict(sentence='President Biden announces new infrastructure plan in speech to Congress.', entity_span='President Biden', label=WRONG_SPAN, correct_span='Biden'),
                dict(sentence='CEO of Dubai-based company arrested for fraud.', entity_span='CEO', label=NA),
                dict(sentence="China's President Xi Jinping meets with South Korean leader for trade talks.", entity_span='Xi Jinping', label=CORRECT),
                dict(sentence='Former US President Obama awarded The Nobel Foundation Peace Prize.', entity_span='US President Obama', label=WRONG_SPAN, correct_span='Obama')
            ]
        ),
        'location': dict(
            defn=(
                'A named location entity must be the name of a location. '
                # 'Any general reference to a location or locations such as "high school" and "community center" is not a named entity. '
                'Events like "Olympics" and "Fashion Week" are not relevant named entities.'
            ),
            demos=[
                dict(sentence='Celebrities flock to Paris Fashion Week for the latest trends.', entity_span='Paris Fashion Week', label=WRONG_SPAN, correct_span='Paris'),
                dict(sentence='The mayor of Manchester delivers a speech at the town hall.', entity_span='Manchester', label=CORRECT),
                # dict(sentence='German Chancellor Angela Merkel meets with French President Macron.', entity_span='French', label=NA),
            ]
        ),
        'organization': dict(
            defn=(
                'A named organization entity must be the name of an organization. '
                # 'Adjectives such as "Chinese" and "Russian" are not named organization entities. '
                'Adjectives such as "Chinese" and "Russian" are not relevant named entities. '
                'Viruses such as "COVID-19" are also not named organization entities. '
                'Products such as "iPhone" and "Windows" are not relevant named entities.'
            ),
            demos=[
                # dict(sentence='Brazilian President to visit United States next week for trade talks.', entity_span='Brazilian', label=NA),
                dict(sentence='Russian cosmonauts and Chinese astronauts conduct joint space mission.', entity_span='Russian', label=NA),
                dict(sentence='Indian government bans TikTok and 58 other Chinese apps.', entity_span='TikTok', label=CORRECT)
            ]
        )
    },
    # '24-02-08_NER-Dataset_{fmt=n-p2,#l=3,ap={dc=T,de=s}}': {
    '24-02-25_NER-Dataset_{fmt=n-p2,#l=3,ap={dc=T,de=s}}_add-super-idx': {
        'person': dict(
            defn=(
                'A named person entity must be the name of a person. '
                'Only first names, last names, and full names are considered named person entities. '
                'Any general reference to a person or people such as "woman", "CEO" and "high school student" is not a named entity. '
                # 'A named person entity should not have any starting titles such as "President" or "Mayor".'
                'A named person entity should not have any starting titles such as "Professor" and "Dr.".'
            ),
            demos=[
                # dict(sentence='President Biden announces new infrastructure plan in speech to Congress.', entity_span='President Biden', label=WRONG_SPAN, correct_span='Biden'),
                dict(sentence='CEO of Dubai-based company arrested for fraud.', entity_span='CEO', label=NA),
                dict(sentence="China's President Xi Jinping meets with South Korean leader for trade talks.", entity_span='Xi Jinping', label=CORRECT),
                dict(
                    sentence='Professor David Johnson from Buenos Aires, Argentina, receives prestigious award from the National Society for the Gifted and Talented.',
                    entity_span='Professor David Johnson', label=WRONG_SPAN, correct_span='David Johnson'),
                dict(sentence='Indian Prime Minister to visit Berlin next week.', entity_span='Indian Prime Minister', label=NA)
            ]
        ),
        'location': dict(
            defn=(
                'A named location entity must be the name of a location. '
                'Any general reference to a location or locations such as "downtown area" and "community center" is not a named entity. '
                'Events like "Olympics" and "Fashion Week" are not relevant named entities.'
            ),
            demos=[
                dict(sentence='Celebrities flock to Paris Fashion Week for the latest trends.', entity_span='Paris Fashion Week', label=WRONG_SPAN, correct_span='Paris'),
                dict(sentence='The mayor of Manchester delivers a speech at the town hall.', entity_span='Manchester', label=CORRECT),
                # dict(sentence='German Chancellor Angela Merkel meets with French President Macron.', entity_span='French', label=NA),
            ]
        ),
        'organization': dict(
            defn=(
                'A named organization entity must be the name of an organization. '
                'Any general reference to an organization or organizations such as "non-profit organization", "foreign government", "high school" and "school district" is not a named entity. '
                'Adjectives such as "Indian", "European", "Israeli", and "Russian" are also not relevant named entities.'
                # 'Viruses such as "COVID-19" are also not named organization entities. '
                # 'Products such as "iPhone" and "Windows" are not relevant named entities.'
            ),
            demos=[
                dict(sentence='Joe Biden holds talks with European leaders on climate change and global security.', entity_span='European', label=NA),
                # dict(sentence='Joe Biden holds talks with European leaders on climate change and global security.', entity_span='European', label=WRONG_TYPE, correct_type='other'),
                dict(sentence='Environmental group launches campaign to protect Arctic Circle wildlife.', entity_span='Environmental group', label=NA),
                # dict(sentence='Local bakery donates over 500 loaves of bread to Feeding America.', entity_span='Local bakery', label=NA),
                dict(sentence='Taiwanese startup revolutionizes the tech industry in Canada with innovative new app.', entity_span='Taiwanese', label=NA),
                dict(sentence='Renowned Indian classical dancer performs at prestigious cultural event in Mumbai, India.', entity_span='Indian', label=NA),
                # dict(sentence="Elon Musk's SpaceX successfully launches a new batch of Starlink satellites into orbit to expand internet coverage.", entity_span='SpaceX', label=CORRECT)
            ]
        )
    }
}
