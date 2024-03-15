from src.generate.step_wise.entity_type2info_dict.util import CORRECT, WRONG_SPAN, WRONG_TYPE, NA


__all__ = ['entity_type2info_dict']


entity_type2info_dict = {
    '24-02-11_NER-Dataset_{fmt=n-p2,#l=50}_add-super-idx': {
        'person': dict(
            defn=(
                'A named person entity must be the name of a person. '
                'Names of figures in history, literature, art and mythology such as "Mona Lisa" are also named person entities. '
                # 'General reference to a person such as "Pope", "President of Ireland" are not named entities. '
                'General reference to a person such as "Pope", "President of Ireland" are not named entities of any relevant type. '
                'A named person entity should not have any starting titles such as "President". '
                'Other names of work of art such as "The Starry Night" are not relevant named entities.'
            ),
            demos=[
                dict(sentence='The author of the article, Dr. John Smith, is a leading expert in the field of neuroscience.', entity_span='John Smith', label=CORRECT),
                dict(sentence='The Sistine Chapel is a chapel in the Apostolic Palace, the official residence of the Pope, in Vatican City.', entity_span='Pope', label=NA),
                dict(sentence='She served as Secretary of State under President George W. Bush.', entity_span='President George W. Bush', label=WRONG_SPAN, correct_span='George W. Bush'),
                dict(sentence='The Louvre Museum in Paris is home to the famous painting, the Mona Lisa.', entity_span='Mona Lisa', label=CORRECT),
                # dict(sentence='The Kremlin is a historic fortified complex located in Moscow, Russia, and is the official residence of the President of Russia.', entity_span='President of Russia', label=NA),
                dict(sentence='The Queen of England, Elizabeth II, celebrated her diamond jubilee in 2012.', entity_span='Queen of England', label=NA)
            ]
        ),
        'location': dict(
            defn=(
                'A named location entity must be the name of a location. '
                'Demonyms such as "English", "American" and "Greek" are not relevant named entities.'
            ),
            demos=[
                dict(sentence='The Mississippi River runs through several states in the United States, including Minnesota, Wisconsin, and Louisiana.', entity_span='United States', label=CORRECT),
                dict(
                    sentence='TMahatma Gandhi, a leader of the Indian independence movement, used nonviolent civil disobedience to lead India to independence from British rule.',
                    entity_span='British', label=NA),
            ]
        ),
        'organization': dict(
            defn=(
                'A named organization entity must be the name of an organization. '
                'Film titles, works of art and projects such as "Avatar" and "Apollo 11" are not relevant named entities. '
                'Demonyms such as "Chinese", "French" and "American" are also not relevant named entities. '
                'Awards such as "Nobel Prize" and "Fields Medal" are not relevant named entities. '
                '(Governmental or political) titles such as "President of the United States" are also not relevant named entities.'
            ),
            demos=[
                dict(sentence='The World Health Organization is a specialized agency of the United Nations responsible for international public health.', entity_span='United Nations', label=CORRECT),
                dict(sentence='Ernest Hemingway, an American novelist, won the Nobel Prize in Literature in 1954.', entity_span='Nobel Prize', label=WRONG_TYPE, correct_type='other'),
                dict(
                    sentence=' She was the first female Prime Minister of India and served from 1966 to 1977 and then again from 1980 until her assassination in 1984.',
                    entity_span='Prime Minister of India', label=WRONG_TYPE, correct_type='other')
            ]
        ),
    },
    '24-02-11_NER-Dataset_{fmt=n-p2,#l=3,dc=T}_add-super-idx': {
        'person': dict(
            defn=(
                'A named person entity must be the name of a person. '
                # 'Names of figures in history, literature, art and mythology such as "Mona Lisa" are also named person entities. '
                'Names of mythological figures such as "Zeus" and "Athena" are not relevant named entities. '
                'General reference to a person such as "CEO" are also not named entities. '
                'A named person entity should not have any starting titles such as "Dr.".'
                # 'Other names of work of art such as "The Starry Night" are not relevant named entities.'
            ),
            demos=[
                dict(sentence='The novel To Kill a Mockingbird, written by Harper Lee, is a classic of modern American literature.', entity_span='Harper Lee', label=CORRECT),
                # dict(sentence='The Sistine Chapel is a chapel in the Apostolic Palace, the official residence of the Pope, in Vatican City.', entity_span='Pope', label=NA),
                dict(sentence='Dr. James E. Webb was instrumental in the development and success of the Apollo lunar landing program.', entity_span='Dr. James E. Webb', label=WRONG_SPAN, correct_span='James E. Webb'),
                dict(sentence='The ancient Greek myth of Orpheus and Eurydice has been retold in countless plays, operas, and films.', entity_span='Eurydice', label=WRONG_TYPE, correct_type='other'),
                # dict(sentence='The Queen of England, Elizabeth II, celebrated her diamond jubilee in 2012.', entity_span='Queen of England', label=NA)
            ]
        ),
        'location': dict(
            defn=(
                'A named location entity must be the name of a location. '
                'General reference to a location such as "city" and "business park" is not a named entity. '
                'Demonyms such as "French", "American", "African" and "Western" are also not relevant named entities.'
            ),
            demos=[
                dict(sentence='The Mississippi River runs through several states in the United States, including Minnesota, Wisconsin, and Louisiana.', entity_span='United States', label=CORRECT),
                dict(sentence='Dr. Jonas Salk, an American virologist, developed the first successful polio vaccine in 1955.', entity_span='American', label=WRONG_TYPE, correct_type='other'),
            ]
        ),
        'organization': dict(
            defn=(
                'A named organization entity must be the name of an organization. '
                'Landmark projects such as "Artemis program" and "Apollo 11" are not relevant named entities. '
                'Demonyms such as "Chinese", "French" and "American" are also not relevant named entities. '
                'Awards such as "Academy Award" and "Grammy Award" are not relevant named entities. '
                'Festivals such as "Cannes Film Festival" are also not relevant named entities. '
                # 'General references to subjects and concepts such as "Literature" and "Fibonacci sequence" are not relevant named entities. '
                'Subjects such as "literature" and "western philosophy" are not relevant named entities. '
                'Concepts such as "Fibonacci sequence" and "Higher education" are also not relevant named entities. '
            ),
            demos=[
                dict(sentence='The World Health Organization is a specialized agency of the United Nations responsible for international public health.', entity_span='United Nations', label=CORRECT),
                # dict(sentence='Ernest Hemingway, an American novelist, won the Nobel Prize in Literature in 1954.', entity_span='Nobel Prize', label=WRONG_TYPE, correct_type='other'),
                dict(sentence='Aretha Franklin, also known as the "Queen of Soul", was an American singer, songwriter, and pianist.', entity_span='American', label=WRONG_TYPE, correct_type='other'),
                dict(sentence='The Fashion industry plays a significant role in shaping global trends and consumer preferences.', entity_span='Fashion industry', label=NA),
                dict(
                    sentence='The concept of online education has gained popularity in recent years, with many universities and organizations offering virtual courses and degrees.',
                    entity_span='online education', label=NA),
                dict(sentence='The director, Steven Spielberg, won an Academy Award for Best Director for his work on the film.', entity_span='Academy Award', label=WRONG_TYPE, correct_type='other'),
                dict(
                    sentence='educational psychology is a field of study that focuses on understanding how individuals learn and develop in educational settings.',
                    entity_span='educational psychology', label=WRONG_TYPE, correct_type='other')
            ]
        ),
    },
    '24-02-11_NER-Dataset_{fmt=n-p2,#l=3,de=T}_add-super-idx': {
        'person': dict(
            defn=(
                'A named person entity must be the name of a person. '
                'Only first names, last names, and full names are considered named person entities. '
                'Names of people in history such as "Khafre" are also named person entities. '
                'General reference to a person or persons such as "philanthropists" are not named entities. '
                'Film titles and works of art such as "The Dark Knight" and "Mona Lisa" are also not relevant named entities. '
                'A named person entity should not have any starting titles such as "Dr.".'
                # 'Other names of work of art such as "The Starry Night" are not relevant named entities.'
            ),
            demos=[
                dict(sentence='The novel To Kill a Mockingbird, written by Harper Lee, is a classic of modern American literature.', entity_span='Harper Lee', label=CORRECT),
                dict(sentence='Dr. James E. Webb was instrumental in the development and success of the Apollo lunar landing program.', entity_span='Dr. James E. Webb', label=WRONG_SPAN, correct_span='James E. Webb'),
                dict(sentence='The Mona Lisa, also known as La Gioconda, is a half-length portrait painting by Leonardo da Vinci.', entity_span='Mona Lisa', label=WRONG_TYPE, correct_type='other'),
                dict(sentence='She is a well-known environmental activist and has spoken at numerous international conferences on climate change.', entity_span='environmental activist', label=NA),
            ]
        ),
        'location': dict(
            defn=(
                'A named location entity must be the name of a location. '
                'General reference to a location such as "city" and "business park" is not a named entity. '
                'Demonyms such as "French", "American", "African" and "Western" are also not relevant named entities.'
            ),
            demos=[
                dict(sentence='The Mississippi River runs through several states in the United States, including Minnesota, Wisconsin, and Louisiana.', entity_span='United States', label=CORRECT),
                dict(sentence='Dr. Jonas Salk, an American virologist, developed the first successful polio vaccine in 1955.', entity_span='American', label=WRONG_TYPE, correct_type='other'),
            ]
        ),
        'organization': dict(
            defn=(
                'A named organization entity must be the name of an organization. '
                # 'Demonyms such as "Chinese", "French" and "American" are not relevant named entities. '
                'Demonyms such as "American", "French", "European" and "Indian" are not relevant named entities. '
                'Prestigious Awards such as "Academy Award" and "Nobel Prize in Physics" are also not relevant named entities. '
                'Events such as "Olympic Games" and "Cannes Film Festival" are not relevant named entities. '
                'Historical events such as "World War II" and "American Civil War" are also not relevant named entities. '
                'Historical periods such as "Elizabethan era" and "Romantic movement" are not relevant named entities. '
                # 'Novel names and works of art such as "The Two Towers" and "Mona Lisa" are also not relevant named entities. '
                'Literary works and works of art such as "The Two Towers" and "Mona Lisa" are also not relevant named entities. '
                'Film titles and entertainment series such as "The Dark Knight" and "Harry Potter" are not relevant named entities. '
                'General references to an organization or organizations such as "environmental organization" and "political organization" are not named entities. '
            ),
            demos=[
                dict(sentence='The World Health Organization is a specialized agency of the United Nations responsible for international public health.', entity_span='United Nations', label=CORRECT),
                # dict(sentence='Ernest Hemingway, an American novelist, won the Nobel Prize in Literature in 1954.', entity_span='Nobel Prize', label=WRONG_TYPE, correct_type='other'),
                dict(sentence='She won the Nobel Prize in Literature in 2013 for her novels and short stories.', entity_span='Nobel Prize in Literature', label=WRONG_TYPE, correct_type='other'),
                dict(sentence='Aretha Franklin, also known as the "Queen of Soul", was an American singer, songwriter, and pianist.', entity_span='American', label=WRONG_TYPE, correct_type='other'),
                dict(sentence='The Fashion industry plays a significant role in shaping global trends and consumer preferences.', entity_span='Fashion industry', label=NA),
                # dict(sentence='The director, Steven Spielberg, won an Academy Award for Best Director for his work on the film.', entity_span='Academy Award', label=WRONG_TYPE, correct_type='other'),
                dict(sentence='The actress Emma Watson gained international fame from her role as Hermione Granger in the Harry Potter film series.', entity_span='Harry Potter', label=WRONG_TYPE, correct_type='other'),
                dict(sentence='The Declaration of Independence, written by Thomas Jefferson, was adopted by the Continental Congress on July 4, 1776.', entity_span='Declaration of Independence', label=WRONG_TYPE, correct_type='other'),
            ]
        ),
    },
    '24-02-14_NER-Dataset_{fmt=n-p2,#l=3,de=s}_add-super-idx': {
        'person': dict(
            defn=(
                'A named person entity must be the name of a person. '
                'Only first names, last names, and full names are considered named person entities. '
                # 'Names of people in history such as "Khafre" are also named person entities. '
                'Names of fictional figures in film, literature, art and mythology such as "Harry Potter", "Mona Lisa" and "Odin" are not relevant named entities. '
                # 'Film titles and works of art such as "The Dark Knight" and "Mona Lisa" are also not relevant named entities. '
                # '(Governmental, political or executive) titles such as "President of the United States" and "CEO" are also not relevant named entities. '
                '(Governmental, political or executive) titles such as "President of the United States" and "CEO" are also not named entities. '
                'General reference to a person or persons such as "lead singer" are not named entities. '
                'A named person entity should not have any starting titles such as "Dr.".'
                # 'Other names of work of art such as "The Starry Night" are not relevant named entities.'
            ),
            demos=[
                dict(sentence='The novel To Kill a Mockingbird, written by Harper Lee, is a classic of modern American literature.', entity_span='Harper Lee', label=CORRECT),
                dict(sentence='Dr. James E. Webb was instrumental in the development and success of the Apollo lunar landing program.', entity_span='Dr. James E. Webb', label=WRONG_SPAN, correct_span='James E. Webb'),
                dict(sentence='The Mona Lisa, also known as La Gioconda, is a half-length portrait painting by Leonardo da Vinci.', entity_span='Mona Lisa', label=WRONG_TYPE, correct_type='other'),
                dict(sentence='She is a well-known environmental activist and has spoken at numerous international conferences on climate change.', entity_span='environmental activist', label=NA),
                dict(sentence='She is the CEO of Apple Inc., a multinational technology company based in Cupertino, California.', entity_span='CEO', label=NA),
                # dict(sentence='The Moscow Kremlin serves as the official residence of the President of the Russian Federation.', entity_span='President of the Russian Federation', label=NA),
            ]
        ),
        'location': dict(
            defn=(
                'A named location entity must be the name of a location. '
                # 'Locations in mythology such as "Asgard" and "Olympus" are not relevant named entities. '
                'Demonyms such as "French", "American", "African", "Chinese", "Indian", "South Korean" and "Australian" are not relevant named entities. '
                'Works of art such as "Mona Lisa" are also not relevant named entities. '
                'Viruses such as "COVID-19" are not named location entities. '
                'Hurricanes and other natural disasters are also not named location entities. '
                'General references to a location such as "city" and "business park" is not a named entity.'
            ),
            demos=[
                dict(sentence='The Mississippi River runs through several states in the United States, including Minnesota, Wisconsin, and Louisiana.', entity_span='United States', label=CORRECT),
                dict(sentence='Dr. Jonas Salk, an American virologist, developed the first successful polio vaccine in 1955.', entity_span='American', label=WRONG_TYPE, correct_type='other'),
            ]
        ),
        'organization': dict(
            defn=(
                'A named organization entity must be the name of an organization. '
                'Demonyms such as "American", "French", "European" and "Indian" are not relevant named entities. '
                'Prestigious Awards such as "Oscar", "Academy Award" and "Nobel Prize" are also not relevant named entities. '
                # 'Events such as "Olympic Games" and "Cannes Film Festival" are not relevant named entities. '
                # 'Historical events such as "World War II" and "American Civil War" are also not relevant named entities. '
                'Historical periods such as "Enlightenment" and "Italian Renaissance" are not relevant named entities. '
                'Literary works and works of art such as "The Age of Innocence" and "Mona Lisa" are also not relevant named entities. '
                'Products such as "Tesla Model S" are not relevant named entities. '
                # 'Film titles and entertainment series such as "The Dark Knight" and "Harry Potter" are not relevant named entities. '
                'General references to an organization or organizations such as "university" and "non-profit organization" are not named entities.'
            ),
            demos=[
                dict(sentence='The World Health Organization is a specialized agency of the United Nations responsible for international public health.', entity_span='United Nations', label=CORRECT),
                dict(sentence='She won the Nobel Prize in Literature in 2013 for her novels and short stories.', entity_span='Nobel Prize in Literature', label=WRONG_TYPE, correct_type='other'),
                dict(sentence='Aretha Franklin, also known as the "Queen of Soul", was an American singer, songwriter, and pianist.', entity_span='American', label=WRONG_TYPE, correct_type='other'),
                dict(sentence='The company partnered with a local non-profit organization to organize a charity event in Tokyo, Japan.', entity_span='non-profit organization', label=NA),
            ]
        ),
    },
    '24-02-14_NER-Dataset_{fmt=n-p2,#l=3,ap={dc=T,de=s}}_add-super-idx': {
        'person': dict(
            defn=(
                'A named person entity must be the name of a person. '
                # 'Only first names, last names, and full names are considered named person entities. '
                'Names of fictional figures in mythology such as "Zeus" and "Apollo" are not relevant named entities. '
                '(Governmental, political or executive) titles such as "President of the United States" and "CEO" are also not relevant named entities. '
                'A named person entity should not have any starting titles such as "Dr." and "General".'
            ),
            demos=[
                dict(sentence='The novel To Kill a Mockingbird, written by Harper Lee, is a classic of modern American literature.', entity_span='Harper Lee', label=CORRECT),
                dict(sentence='Dr. James E. Webb was instrumental in the development and success of the Apollo lunar landing program.', entity_span='Dr. James E. Webb', label=WRONG_SPAN, correct_span='James E. Webb'),
                dict(sentence='Hera is often depicted as the queen of the gods in Greek mythology, and is the wife and sister of Zeus.', entity_span='Zeus', label=WRONG_TYPE, correct_type='other'),
                dict(sentence='She is a well-known environmental activist and has spoken at numerous international conferences on climate change.', entity_span='environmental activist', label=NA),
                # dict(sentence='She is the CEO of Apple Inc., a multinational technology company based in Cupertino, California.', entity_span='CEO', label=NA),
                # dict(sentence='The Moscow Kremlin serves as the official residence of the President of the Russian Federation.', entity_span='President of the Russian Federation', label=NA),
            ]
        ),
        'location': dict(
            defn=(
                'A named location entity must be the name of a location. '
                'Locations in mythology such as "Underworld" are also named location entities. '
                'Demonyms such as "French", "British", "American", "Chinese", "Spanish" and "Greek" are not relevant named entities. '
                'General references to a location such as "city" and "business park" are not named entities.'
            ),
            demos=[
                dict(sentence='The Mississippi River runs through several states in the United States, including Minnesota, Wisconsin, and Louisiana.', entity_span='United States', label=CORRECT),
                dict(sentence='Dr. Jonas Salk, an American virologist, developed the first successful polio vaccine in 1955.', entity_span='American', label=WRONG_TYPE, correct_type='other'),
                dict(
                    sentence='The headquarters, located in the heart of Silicon Valley, provides the ideal environment for fostering creativity and collaboration among employees.',
                    entity_span='headquarters', label=NA
                ),
            ]
        ),
        'organization': dict(
            defn=(
                'A named organization entity must be the name of an organization. '
                'Demonyms such as "American", "French", "Greek" and "Norse" are not relevant named entities. '
                'Prestigious awards and rankings such as "Grammy Award", "Nobel Peace Prize" and "Billboard 200" are also not relevant named entities. '
                'Space missions such as "Chandrayaan-1" and "V-2 rocket" are not relevant named entities. '
                'Events such as "Olympic Games" and "Cannes Film Festival" are also not relevant named entities. '
                'General references to an organization or organizations such as "university" and "non-profit organization" are not named entities.'
            ),
            demos=[
                dict(sentence='The World Health Organization is a specialized agency of the United Nations responsible for international public health.', entity_span='United Nations', label=CORRECT),
                dict(sentence='She won the Nobel Prize in Literature in 2013 for her novels and short stories.', entity_span='Nobel Prize in Literature', label=WRONG_TYPE, correct_type='other'),
                dict(sentence='Aretha Franklin, also known as the "Queen of Soul", was an American singer, songwriter, and pianist.', entity_span='American', label=WRONG_TYPE, correct_type='other'),
                dict(sentence='The company partnered with a local non-profit organization to organize a charity event in Tokyo, Japan.', entity_span='non-profit organization', label=NA),
                # dict(sentence="Antarctica is the Earth's southernmost continent, containing the geographic South Pole and is situated in the Antarctic region of the [Southern Hemisphere], almost entirely south of the Antarctic Circle.")
            ]
        ),
    }
}
