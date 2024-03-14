from src.generate.step_wise.entity_type2info_dict.util import *


__all__ = ['entity_type2info_dict']


entity_type2info_dict = {
    'default': {
        'Title': dict(
            defn=(
                # 'A named movie title entity must be the name of a movie.'

                'A named movie title entity must be the name of a movie. '
                'General reference to movies such as "latest film" are not named entities.'
                # 'Movie production companies such as "Marvel" are also not named movie title entities.'  # can be a Genre
            ),
            name='movie title',
            demos=[
                dict(sentence="I heard there's a new movie called Battle for Survival, what's it about?", entity_span='Battle for Survival', label=CORRECT),
                dict(sentence="can I watch the trailer clip for the new frozen movie featuring princess anna", entity_span='new frozen movie', label=WRONG_SPAN, correct_span='frozen'),
                dict(sentence="can you provide me with a preview screening of the upcoming James Bond film", entity_span='preview screening', label=WRONG_TYPE, correct_type='Trailer'),
                dict(sentence="Show me the trailer for the latest Marvel movie", entity_span='Marvel', label=WRONG_TYPE, correct_type='Genre'),
                dict(sentence="Who directed the latest movie featuring Harley Quinn?", entity_span='latest movie', label=NA)
            ]
        ),
        "Viewers' Rating": dict(
            # taken from entity pool gen
            # defn="A named viewers' rating entity must be a rating score or a brief recommendation/popularity level.",
            defn={
                'named entity':
                    # "A named viewers' rating entity must be a rating score or a brief recommendation or popularity level",
                    # "A named viewers' rating entity must be a rating score of a movie or a short recommendation or popularity level for a movie",
                    # "A named viewers' rating entity must be a movie rating score or a short recommendation or popularity level",

                    # 'A named viewers\' rating entity must be a rating score, such as "4 out of 5 stars", '
                    # 'a brief description, such as "good", or a recommendation/popularity level, such as "highly recommended" or "must see"',

                    # 'A named viewers\' rating entity must be one of a rating score, such as "4 out of 5 stars", '
                    # 'a brief movie assessment, such as "good", or a recommendation/popularity level, such as "highly recommended" or "must see"',

                    # 'A named viewers\' rating entity must be either a rating score, a brief movie assessment, or a recommendation/popularity level. '
                    # 'For example, "4 out of 5 stars", "good", "highly recommended" and "must see" are all named viewers\' rating entities.',

                    'A named viewers\' rating entity must be either a rating score, a brief movie assessment, or a recommendation/popularity level. '
                    'For example, "4 out of 5 stars", "good", "highly recommended" and "must see" are all named viewers\' rating entities. '
                    # 'General reference to movie ratings such as "viewers\' rating" are not named viewers\' rating entities.'
                    'General reference to movie ratings such as "viewers\' rating" are not named entities.'
                # 'term': "A viewers' rating term must be a rating score or a brief recommendation or popularity level",
            },
            # examples=['4 out of 5 stars', 'must see'],
            # name="viewer's rating",
            name="viewers' rating",
            equivalents=['Popularity Level'],  # LLM-generated corrected entity types that can be considered as the current entity type
            demos=[
                dict(sentence="can you recommend an Awful horror movie from the 1980s", entity_span='Awful', label=CORRECT),
                # dict(
                #     sentence="who directed the highest rated movie of 2019",
                #     entity_span="highest rated", label=CORRECT
                # ),
                dict(sentence='who is the director of the highest-rated romantic comedy movie', entity_span="highest-rated", label=CORRECT),
                # dict(
                #     sentence="Can you recommend a movie that is not rated but has an Excellent plot",
                #     entity_span="Excellent", label=CORRECT
                # ),
                dict(sentence="show me a movie directed by Christopher Nolan that has a viewers' rating of 9 or higher", entity_span="viewers' rating of 9 or higher", label=WRONG_SPAN, correct_span='9 or higher'),
                dict(sentence="who directed the mystery film that came out in 1995 with a 5-star rating", entity_span="5-star rating", label=WRONG_SPAN, correct_span='5-star'),
                dict(sentence="I'm looking for a film starring Natalie Dormer with a high viewers' rating.", entity_span="high", label=WRONG_SPAN, correct_span="high viewers' rating"),
                # Not a good, representative demo for just wrong type, TODO: try just ignore a demo for this class
                # _LABEL_WRONG_TYPE: [dict(
                #     sentence="Can you recommend a feel-good comedy directed by the Coen Brothers and rated R?",
                #     entity_span='feel-good', label=WRONG_TYPE, correct_type='Genre'
                # )],
                dict(sentence="What is the viewers' rating for the movie Sleeping Beauty?", entity_span="viewers' rating", label=NA)
            ]
        ),
        'Character': dict(
            defn=(
                # 'A named character entity must be the name of a specific character in a movie.'

                'A named character entity must be the name of a specific character in a movie. '
                'General reference to characters such as "character" and "main character" are not named entities. '
                'Categories or groups of characters such as "female character" are also not named entities.'
            ),
            # name='character',
            name='movie character',
            demos=[
                dict(sentence="Show me the trailer for the new James Bond film with Daniel Craig", entity_span='James Bond', label=CORRECT),
                dict(sentence="can I watch the trailer clip for the new frozen movie featuring Princess Anna", entity_span='Princess Anna', label=CORRECT),
                # _LABEL_WRONG_TYPE: [dict(
                #     sentence="What is the monster movie with the highest viewers' rating?",
                #     entity_span='monster', label=WRONG_TYPE, correct_type='Plot'
                # )],
                dict(sentence="who played the lead character in the action movie Die Hard", entity_span='lead character', label=NA),
                dict(sentence="Who directed the top-rated movie from the 2010s with a strong female lead character?", entity_span='strong female lead', label=NA),
                dict(sentence="who plays the lead role in the romantic comedy film with the highest Viewers' Rating", entity_span='lead role', label=NA),
            ]
        ),
        'Year': dict(
            defn=(
                # 'A named year entity must be a specific year or a year range, such as "2000" or "last three years".'

                'A named year entity must be a specific year or a year range, such as "2000" or "last three years". '
                'General reference to years such as "year" and "years" are not named entities. '
                'Vague descriptions of movie release such as "released" and "come out" are also not named entities.'
            ),
            name='year',
            demos=[
                dict(sentence="What is a good character-driven drama film from the 1990s?", entity_span='1990s', label=CORRECT),
                dict(sentence="Show me a sci-fi movie from the 90s with an alien invasion theme", entity_span="the 90s", label=WRONG_SPAN, correct_span='90s'),
                # _LABEL_WRONG_TYPE: [dict(
                #     sentence="TODO",
                #     entity_span='monster', label=WRONG_TYPE, correct_type='Plot'
                # )],
                    # dict(
                    #     sentence='Who directed the movie The Tin Man and what year was it released?',
                    #     entity_span='year', label=NA
                    # ),
                dict(sentence='Who directed The Lion King and what year was it released?', entity_span='released', label=NA),
                dict(sentence='What is the plot of the latest Marvel superhero movie?', entity_span='latest', label=NA),
                dict(sentence='Show me the trailer for the new James Bond film with Daniel Craig', entity_span='new', label=NA)
            ]
        ),
        'Actor': dict(
            defn=(
                # 'A named actor entity must be the name of a specific actor or actress.'

                'A named actor entity must be the name of a specific actor or actress. '
                'General reference to actors such as "actor" are not named entities. '
                'Categories or groups of actors such as "lead actor" and "main actors" are also not named entities.'
            ),
            # name='actor',
            name='movie actor',
            demos=[
                dict(sentence="Show me the trailer for the new James Bond film with Daniel Craig", entity_span='Daniel Craig', label=CORRECT),
                dict(sentence="can you give me some information about the Mockbuster starring Adam Sandler", entity_span="starring Adam Sandler", label=WRONG_SPAN, correct_span='Adam Sandler'),
                # _LABEL_WRONG_TYPE: [dict(
                #     sentence="TODO",
                #     entity_span='monster', label=WRONG_TYPE, correct_type='Plot'
                # )],
                dict(sentence='Who is the actor that played the main character in the movie Inception', entity_span='actor', label=NA),
                dict(sentence='Who is the lead actor in the movie The Shawshank Redemption', entity_span='lead actor', label=NA)
            ]
        ),
        'Genre': dict(
            defn=(
                # 'A named genre entity must be a specific film category, such as "documentary", '
                # 'or a adjective describing a film category, such as "funny". '
                # 'Named genre entity spans must not have trailing "movie" or "film" words.'

                # 'A named genre entity must be a specific film category, such as "documentary", '
                # 'or a adjective describing a film category, such as "funny". '
                # 'Ensure that named genre entity spans must never have trailing "movie" or "film" words.'

                'A named genre entity must be a specific film category, such as "documentary", '
                'or a adjective describing a film category, such as "funny". '
                'Named genre entity spans must not have trailing "movie" or "film" words.'
                # 'If you see a named genre entity with trailing "movie" or "film" words, '
                # 'you should always correct the span by removing the trailing "movie" or "film" words.'
                'If a potential named genre entity has a trailing "movie" or "film" word, '
                'you should always correct the span by removing the trailing "movie" or "film" word.'
            ),
            name='movie genre',
            demos=[
                dict(sentence="Can you recommend a feel-good comedy directed by the Coen Brothers and rated R?", entity_span='feel-good comedy', label=CORRECT),
                dict(sentence="Can you recommend a popular action movie from the 1980s?", entity_span='popular action movie', label=WRONG_SPAN, correct_span='action'),
                dict(sentence="who directed the animated film Aladdin and what year was it released", entity_span='animated film', label=WRONG_SPAN, correct_span='animated'),
                dict(sentence="What is the plot of the comedy film featuring the song 'Bohemian Rhapsody'?", entity_span='comedy film', label=WRONG_SPAN, correct_span='comedy'),
                dict(sentence='First look, who directed the thriller movie released in 2015 with a gripping plot?', entity_span='thriller movie', label=WRONG_SPAN, correct_span='thriller'),
                # _LABEL_WRONG_TYPE: [dict(
                #     sentence="TODO",
                #     entity_span='monster', label=WRONG_TYPE, correct_type='Plot'
                # )],
                # dict(
                #     sentence='What are some Upcoming releases in the action genre',
                #     entity_span='Upcoming releases', label=NA
                # ),
                dict(sentence='What genre is the movie directed by E?', entity_span='genre', label=NA),
            ]
        ),
        'Review': dict(
            defn='A named review entity must be a detailed comment of a movie, such as "funniest of all time". '
                 'Named review entities must describe specific aspects of a movie, such as "most awards". '
                 'Ensure to separate named review entities from named viewers\' rating entities. '
                 'Named viewers\' rating entities describe more general aspects of a movie, such as "good" and "best-rated".',
            name='movie review',
            demos=[
                dict(sentence="Which movie won the most awards in 2010?", entity_span='most awards', label=CORRECT),
                # dict(
                #     sentence='Can you recommend a science fiction movie from the 1980s with amazing special effects?',
                #     entity_span='amazing special effects', label=CORRECT
                # ),
                # _LABEL_WRONG_BOUNDARY: [
                #     dict(
                #         sentence="What is the plot of the latest Marvel superhero movie?",
                #         entity_span='latest Marvel superhero movie', label=WRONG_SPAN, correct_span='latest'
                #     ),
                # ],
                dict(sentence="What is the B movie with the highest viewers' rating?", entity_span="highest viewers' rating", label=WRONG_TYPE, correct_type="Viewers' Rating"),
                dict(sentence='what is the best-rated movie of the year 2020', entity_span="best-rated", label=WRONG_TYPE, correct_type="Viewers' Rating"),
                dict(sentence="Which movie has the highest viewers' rating on rotten tomatoes?", entity_span='rotten tomatoes', label=NA),
                dict(sentence='Is there a review for the best-rated Western film from the 1950s?', entity_span='review', label=NA)
            ]
        ),
        'MPAA Rating': dict(
            defn='A named MPAA rating entity must be a specific MPAA rating category, such as "PG-13". '
                 'Named MPAA rating entity spans must not have starting or trailing "rated" or "rating" words. '
                 'The description of a MPAA rating is not a named MPAA rating entity, such as "Parental Guidance Suggested". '
                 'TV Parental Guidelines ratings are not named MPAA rating entities, such as "TV-14".',
            name='MPAA rating',
            demos=[
                dict(sentence="Can you recommend a family-friendly movie rated PG with the song Sound of Silence?", entity_span='PG', label=CORRECT),
                dict(sentence="What are some unforgettable G-rated movies from the 90s", entity_span='G-rated', label=WRONG_SPAN, correct_span='G'),
                # _LABEL_WRONG_TYPE: [dict(
                #     sentence="TODO",
                #     entity_span='monster', label=WRONG_TYPE, correct_type='Plot'
                # )],
                dict(sentence='what is the MPAA rating of The Bridge on the River Kwai', entity_span='MPAA rating', label=NA),
                dict(sentence='Can you recommend a movie with a TV-MA rating and a gripping plot', entity_span='TV-MA', label=NA),
            ]
        ),
        'Plot': dict(
            defn=(
                # 'A named plot entity must be a particular movie theme or plot element, such as "bounty hunter" and "zombie". '
                # 'Adjectives describing a plot such as "exciting" in "exciting plot" are not named plot entities. '
                # 'Named movie plot entity spans must not have trailing "plot" words.',

                'A named plot entity must be a particular movie theme or plot element, such as "bounty hunter" and "zombie". '
                'General reference to movie plots such as "plot" and "plot summary" are not named plot entities. '
                'Specific adjectives describing movie plots such as "exciting plot" and "compelling plot" are also not named plot entities.'
            ),
            name='movie plot',
            demos=[
                dict(sentence="Can you suggest a good heist movie with a PG-7 rating", entity_span='heist', label=CORRECT),
                dict(sentence="Have any historical movies with a revenge plot been released in 2022", entity_span='revenge plot', label=WRONG_SPAN, correct_span='revenge'),
                # _LABEL_WRONG_TYPE: [dict(
                #     sentence="TODO",
                #     entity_span='monster', label=WRONG_TYPE, correct_type='Plot'
                # )],
                dict(sentence='Can you recommend any movies with a Dynamic plot that came out in 2016?', entity_span='Dynamic', label=NA),
                # dict(
                #     sentence='Show me a movie with the most intense action scenes and also features Gal Gadot.',
                #     entity_span='most intense action scenes', label=NA
                # ),
                dict(sentence='can you recommend a movie with intense action scenes?', entity_span='intense action scenes', label=NA),
                dict(sentence='Who directed the popular sci-fi film Arrival, and what is the plot about?', entity_span='plot', label=NA),
                dict(sentence='Can I watch a teen movie with an AA rating that has an exciting plot?', entity_span='exciting plot', label=NA)
            ]
        ),
        'Director': dict(
            defn=(
                # 'A named director entity must be the name of a specific film director. '
                # 'General reference to directors such as "director" and "directed" are not named director entities. '

                'A named director entity must be the name of a specific film director. '
                'General reference to directors such as "director" and "directed" are not named entities. '
                # 'Categories of directors such as "female director" are also not named entities.'
                'Categories or groups of directors such as "female director" are also not named entities.'
            ),
            name='movie director',
            demos=[
                dict(sentence="What is the highest rated film directed by Steven Spielberg?", entity_span='Steven Spielberg', label=CORRECT),
                # _LABEL_WRONG_BOUNDARY: [
                #     dict(
                #         sentence="TODO",
                #         entity_span='directed by Christopher Nolan', label=WRONG_SPAN, correct_span='Christopher Nolan'
                #     ),
                # ],
                # _LABEL_WRONG_TYPE: [dict(
                #     sentence="TODO",
                #     entity_span='monster', label=WRONG_TYPE, correct_type='Plot'
                # )],
                dict(sentence='Are there any movies based on a Stephen King novel?', entity_span='Stephen King', label=NA),
                dict(sentence='which director is known for making the most thrilling movies', entity_span='director', label=NA),
                dict(sentence='Who directed the latest movie featuring Harley Quinn?', entity_span='directed', label=NA)
            ]
        ),
        'Trailer': dict(
            defn=(
                'A named movie trailer entity is a term that indicates a segment or a clip from a movie, such as "clip" and "trailer". '
                'Named movie trailer entity spans must not have starting or trailing "movie" or "film" words.'
            ),
            name='movie trailer',
            demos=[
                dict(sentence="Show me the trailer for the latest James Cameron movie", entity_span='trailer', label=CORRECT),
                dict(sentence="Can you play the movie clip where Yoda gives his famous advice to Luke Skywalker?", entity_span='movie clip', label=WRONG_SPAN, correct_span='clip'),
                # _LABEL_WRONG_TYPE: [dict(
                #     sentence="TODO",
                #     entity_span='monster', label=WRONG_TYPE, correct_type='Plot'
                # )],
                # _LABEL_NOT_NAMED_ENTITY: [
                #     dict(
                #         sentence='TODO',
                #         entity_span='trailer', label=NA
                #     )
                # ]
            ]
        ),
        'Song': dict(
            defn=(
                # 'A named song entity must be the name of a specific song in a movie. '
                # 'General reference to songs such as "song" and "soundtrack" are not named song entities. '
                # 'Specific adjectives describing songs such as "catchy song" and "great music" are not named song entities. '
                # 'Names of singers such as "Adele" are also not named song entities.'

                'A named song entity must be the name of a specific song in a movie. '
                'General reference to songs such as "song" and "soundtrack" are not named entities. '
                'Specific adjectives describing songs such as "catchy song" and "great soundtrack" are not also named entities. '
                'Further, names of singers such as "Adele" are not named song entities.'
            ),
            name='song',
            demos=[
                dict(sentence="Is Every Breath You Take featured in any popular movie soundtracks?", entity_span='Every Breath You Take', label=CORRECT),
                # _LABEL_WRONG_BOUNDARY: [
                #     dict(
                #         sentence="TODO",
                #         entity_span='song', label=WRONG_SPAN, correct_span='song from the movie Frozen 2'
                #     )
                # ],
                # _LABEL_WRONG_TYPE: [dict(
                #     sentence="TODO",
                #     entity_span='monster', label=WRONG_TYPE, correct_type='Plot'
                # )],
                dict(sentence='What movie from 1986 features iconic music by Kenny Loggins?', entity_span='Kenny Loggins', label=NA),
                dict(sentence='Is there a specific song featured in the classic movie Road Trip?', entity_span='specific song', label=NA),
                # dict(
                #     sentence="what's a romantic comedy film with a great soundtrack",
                #     entity_span='great soundtrack', label=NA
                # ),
                dict(sentence="'show me a movie with a song that is both uplifting and heartwarming'", entity_span='uplifting', label=NA)
            ]
        )
    },
    '24-02-06_NER-Dataset_{fmt=n-p2,#l=50,lc=T}_re-run': {
        'Title': dict(
            defn=(
                'A named movie title entity must be the name of a movie. '
                'Awards such as "Oscar" are not relevant named entities. '
                # 'General reference to movies such as "latest film" are not named entities.'
            ),
            name='movie title',
            demos=[
                dict(sentence='What character did Tom Hanks play in "Forest Gump"?', entity_span='Forest Gump', label=CORRECT),
                dict(sentence='What film won the Best Picture award at the Oscars in 2019?', entity_span='Oscars', label=WRONG_TYPE, correct_type='Other')
            ]
        ),
        "Viewers' Rating": dict(
            defn=(
                # 'A named viewers\' rating entity must be a rating score or a brief recommendation or popularity level. '
                # 'A named viewers\' rating entity must be a rating score or a value judgement. '
                # 'A named viewers\' rating entity must be a rating score such as "9 out of 10" or a value judgement such as "best" and "highly recommended". '
                'A named viewers\' rating entity can be a rating score such as "9 out of 10". '
                # 'Recommendation phrases such as "must see" and "highly recommended" are also named viewers\' rating entities. '
                'Endorsement phrases such as "must see" and "highly recommended" are also named viewers\' rating entities. '
                # 'Value judgements such as "best", "good", "popular" and "favorite" are also named viewers\' rating entities. '
                # 'Value-judgement adjectives such as "best", "good", "popular", "famous" and "favorite" are named viewers\' rating entities. '
                'Evaluative descriptors such as "best", "good", "popular", "famous" and "favorite" are named viewers\' rating entities. '
                'Performance metrics such as "highest-grossing" and "best-rated" are also named viewers\' rating entities. '
                # 'You should capture the complete descriptive phrase such as "high viewers\' rating" as opposed to "high". '
                # 'Just mentioning the verb "recommend" is not a named entity.'
                # 'Rating score platforms such as "Rotten Tomatoes" and "IMDB rating" are not relevant named entities. '
                'Mere queries on rating score platforms such as "Rotten Tomatoes" and "IMDB rating" are not relevant named entities. '
                'General reference to movie ratings such as "viewers\' rating" and "rating" are not named entities.'
            ),
            equivalents=['Rating'],
            demos=[
                dict(sentence='What is the viewers\' rating for the movie Sleeping Beauty?', entity_span="viewers' rating", label=NA),
                # dict(sentence='Name a popular romance movie from the 1990s', entity_span='popular', label=CORRECT),
                dict(sentence='What is the best superhero movie of all time?', entity_span='best', label=CORRECT),
                dict(sentence='I want to watch a horror movie with a high Rotten Tomatoes rating.', entity_span='high Rotten Tomatoes rating', label=CORRECT)
            ]
        ),
        'Year': dict(
            defn=(
                'A named year entity must be a specific year or a year range indicating movie release, such as "2000" or "last three years". '
                # 'General reference to years such as "year" and "years" are not named entities. '
                'Vague temporal descriptors such as "latest", "recent" and "come out" are not named entities. '
                'Release qualifiers such as "released" and "first" are also not named entities.'
            ),
            demos=[
                dict(sentence='What is a good character-driven drama film from the 1990s?', entity_span='1990s', label=CORRECT),
                dict(sentence='Who is the actor playing Spider-Man in the latest movie', entity_span='latest', label=NA),
                dict(sentence='Can you tell me the year in which the movie Pulp Fiction was released', entity_span='released', label=NA)
                # dict(sentence='Show me a sci-fi movie from the 90s with an alien invasion theme', entity_span="the 90s", label=WRONG_SPAN, correct_span='90s')
            ]
        ),
        'Genre': dict(
            defn=(
                'Named genre entities are broad and well-recognized movie categories, such as "documentary" and "comedy". '
                # 'A named genre entity must be a specific film category, such as "documentary", or a adjective describing a film category, such as "funny". '
                'Style descriptors such as "funny", "musical" and "animated" are also named genre entities. '
                'Film studios such as "Marvel" and "Disney" are also named genre entities.\n'
                'You should always drop trailing "movie" or "film" words from named genre entity spans.\n'
                'Ensure to distinguish named genre entities from named plot entities. '
                'Named plot entities are specific themes, topics, or elements of a movie storyline, such as "space exploration" and "zombie".'
                # 'Named genre entity spans must not have trailing "movie" or "film" words.'
            ),
            demos=[
                # dict(sentence='Can you recommend a feel-good comedy directed by the Coen Brothers and rated R?', entity_span='feel-good comedy', label=CORRECT),
                # dict(sentence='Can you recommend a popular action movie from the 1980s?', entity_span='popular action movie', label=WRONG_SPAN, correct_span='action')
                dict(sentence='Who directed the psychological thriller "Black Swan"?', entity_span='psychological thriller', label=CORRECT),
                dict(sentence='Name a famous [heist] movie from the 2000s', entity_span='heist', label=WRONG_TYPE, correct_type='Plot'),
                dict(sentence='Can you provide a list of musical movies', entity_span='musical movies', label=WRONG_SPAN, correct_span='musical')
            ]
        ),
        'Director': dict(
            defn=(
                'A named director entity must be the name of a film director. '
                'Only first names, last names, and full names of directors are considered named director entities. '
                # 'General reference to directors such as "director" and "directed" are not named entities. '
                'Ambiguous identifiers such as "director" and "female directors" are not named entities.'
            ),
            demos=[
                dict(sentence='Are there any movies directed by Steven Spielberg that I should watch?', entity_span='Steven Spielberg', label=CORRECT),
                dict(sentence='who directed the movie Inception', entity_span='directed', label=NA)
            ]
        ),
        'MPAA Rating': dict(
            defn=(
                # 'A named MPAA rating entity must be a specific MPAA rating category, such as "PG-13". '
                'Only actual MPAA rating labels such as "G", "PG", "PG-13", "R", and "NC-17" are named MPAA rating entities. '
                'Rating interpretations such as "Parental Guidance Suggested" and "suitable for children" are not named MPAA rating entities.\n'
                'You should always drop starting and trailing "rated" or "MPAA rating of" words from named MPAA rating entity spans.\n'
                # this made huge number of false-positive genre corrections
                # 'Audience descriptors such as "family-friendly", "family" and "kids" should be named genre entities instead.'
                # 'TV Parental Guidelines ratings such as "TV-14" are not named MPAA rating entities.'
            ),
            demos=[
                dict(sentence='Can you recommend a family-friendly movie rated PG with the song Sound of Silence?', entity_span='PG', label=CORRECT),
                dict(sentence='What are some unforgettable G-rated movies from the 90s', entity_span='G-rated', label=WRONG_SPAN, correct_span='G'),
                dict(sentence='What is the best MPAA rating for a family movie', entity_span='MPAA rating', label=NA),
                dict(sentence='please recommend a family-friendly fantasy film', entity_span='family-friendly', label=WRONG_TYPE, correct_type='Genre')
            ]
        ),
        'Plot': dict(
            defn=(
                # 'A named plot entity must be a particular movie theme or plot element, such as "bounty hunter" and "zombie". '
                'Named plot entities are specific themes, topics, or elements of a movie storyline, such as "space exploration" and "zombie".\n'
                # 'Adjectives describing a plot such as "exciting" in "exciting plot" are not named plot entities. '
                'Narrative descriptors such as "plot" and "summary" are not named entities.'
                # 'Named movie plot entity spans must not have trailing "plot" words.'
            ),
            demos=[
                dict(sentence='Can you suggest a good heist movie with a PG-7 rating', entity_span='heist', label=CORRECT),
                # dict(sentence='Have any historical movies with a revenge plot been released in 2022', entity_span='revenge plot', label=WRONG_SPAN, correct_span='revenge')
                dict(sentence='Who wrote the screenplay for the movie Jurassic Park', entity_span='screenplay', label=NA),
                dict(sentence='Can you give me a brief overview of the plot of "Forrest Gump"?', entity_span='plot', label=NA)
            ]
        ),
        'Actor': dict(
            defn=(
                'A named actor entity must be the name of an actor or actress. '
                'Only first names, last names, and full names of actors and actresses are considered named actor entities.\n'
                # 'Ambiguous identifiers such as "actor" and "lead actor" are not named entities.'
                # 'Ambiguous identifiers such as "actor" and "lead actor" are not named entities and should be filtered out.'
                'Ambiguous identifiers such as "actor", "lead actor", "lead actress" and "male lead" are not named entities.'
                # 'Ambiguous identifiers such as "actor", "lead actress" and "male lead" are not named entities.'
                # 'Role descriptors such as "actor", "lead actress" and "male lead" are not named entities.'
            ),
            demos=[
                # dict(sentence='Who played the lead role in the movie "The Godfather"?', entity_span='lead role', label=CORRECT),
                dict(sentence='What character did Tom Hanks play in "Forest Gump"?', entity_span='Tom Hanks', label=CORRECT),
                dict(sentence='What is the name of the actor who played the main character in "The Shawshank Redemption"?', entity_span='actor', label=NA),
                dict(sentence='Who are the primary actors in the film The Departed?', entity_span='primary actors', label=NA)
            ]
        ),
        'Trailer': dict(
            defn=(
                'A named trailer entity is a phrase that indicates a segment or a clip from a movie, such as "clip" and "trailer". '
                # 'Named movie trailer entity spans must not have starting or trailing "movie" or "film" words.'
            ),
            demos=[
                dict(sentence='Can you show me the trailer for the movie "The Matrix"?', entity_span='trailer', label=CORRECT),
                dict(sentence='Can you play the trailer for the latest Mission: Impossible movie?', entity_span='trailer', label=CORRECT),
                # dict(sentence='Can you play the movie clip where Yoda gives his famous advice to Luke Skywalker?', entity_span='movie clip', label=WRONG_SPAN, correct_span='clip')
            ]
        ),
        'Song': dict(
            defn=(
                'A named song entity must be the name of a song. '
                # 'Ambiguous identifiers such as "song" and "soundtrack" are not named entities.'
                'You should always drop starting or trailing "song" words from named song entity spans.\n'
                'Musical descriptors such as "song", "theme song" and "soundtrack" are not named entities.'
            ),
            demos=[
                dict(sentence='Is Every Breath You Take featured in any popular movie soundtracks?', entity_span='Every Breath You Take', label=CORRECT),
                # dict(sentence='Is there a specific song featured in the classic movie Road Trip?', entity_span='specific song', label=NA),
                dict(sentence='What is the theme song of the James Bond movie "Skyfall"?', entity_span='theme song', label=NA),
                dict(sentence='Can you play a song from the movie "The Sound of Music"?', entity_span='song', label=NA),
                dict(sentence='Can you suggest a movie with a famous soundtrack?', entity_span='famous soundtrack', label=NA)
            ]
        ),
        'Review': dict(
            defn=(
                # 'A named review entity must be a detailed comment of a movie, such as "funniest of all time". '
                'A named review entity must be a recognition or a detailed, qualitative assessment of a movie, such as "funniest of all time" and "incredible visual effects". '
                'Named review entities must describe specific aspects of a movie, such as "most awards".\n'
                'Ensure to distinguish named review entities from named viewers\' rating entities. '
                # 'Named viewers\' rating entities tend to be shorter. '
                'Short evaluative descriptors such as "good", "highest rated" and "top-rated" are named viewers\' rating entities. '
                'Performance metrics such as "highest-grossing" and "best-rated" are also named viewers\' rating entities.\n'
                'Ambiguous identifiers such as "review" are not named entities.'
            ),
            demos=[
                # dict(sentence='What movie won the Best Picture award at the Oscars in 2019?', entity_span='Oscars', label=WRONG_TYPE, correct_type='Other'),
                # dict(sentence='What is the best superhero movie of all time?', entity_span='best', label=WRONG_TYPE, correct_type="Viewers' Rating"),
                # dict(sentence='I want to watch a horror movie with a high Rotten Tomatoes rating.', entity_span='high Rotten Tomatoes rating', label=WRONG_TYPE, correct_type="Viewers' Rating")
                dict(sentence='Which movie has the best special effects of all time', entity_span='best special effects', label=CORRECT),
                dict(sentence='Can you recommend a good movie from the 1980s', entity_span='good', label=WRONG_TYPE, correct_type="Viewers' Rating"),
                dict(sentence="What's your favorite movie review website?", entity_span='review', label=NA)
            ]
        ),
        'Character': dict(
            defn=(
                'A named character entity must be the name of a character.\n'
                'Ambiguous identifiers such as "character", "main character" and "lead role" are not named entities. '
                'Character qualifiers such as "main antagonist", "villain character", "strong female lead character" are also not named entities.'
            ),
            demos=[
                dict(sentence='who is the actor in the lead role in forrest gump', entity_span='lead role', label=NA),
                dict(sentence='Who is the main character in the Star Wars series', entity_span='main character', label=NA),
                dict(sentence='Tell me about the actress who played Hermione Granger in the Harry Potter series.', entity_span='Hermione Granger', label=CORRECT)
            ]
        )
    },
    '24-02-05_NER-Dataset_{fmt=n-p2,#l=3,dc=T,lc=T}': {
        'Title': dict(
            defn=(
                'A named movie title entity must be the name of a movie. '
                'You should always drop starting "latest" or "new" and trailing "movie" or "film" words from named movie title entity spans.'
                # 'Awards such as "Oscar" are not relevant named entities. '
                # 'General reference to movies such as "latest film" are not named entities.'
            ),
            demos=[
                dict(sentence='What character did Tom Hanks play in "Forest Gump"?', entity_span='Forest Gump', label=CORRECT),
                dict(sentence='Can I find tickets for the new James Bond movie?', entity_span='the new James Bond movie', label=WRONG_SPAN, correct_span='James Bond')
            ]
        ),
        "Viewers' Rating": dict(
            defn=(
                'A named viewers\' rating entity can be a rating score such as "9 out of 10". '
                'Endorsement phrases such as "must see" and "highly recommended" are also named viewers\' rating entities. '
                'Evaluative descriptors such as "best", "good", "popular", "famous" and "favorite" are named viewers\' rating entities. '
                'Performance metrics such as "highest-grossing" and "best-rated" are also named viewers\' rating entities.\n'
                'You should capture the complete descriptive phrase such as "high viewers\' rating" as opposed to "high".\n'
                
                'Ensure to distinguish named viewers\' rating entities from named genre entities. '
                'Named genre entities are broad and well-recognized movie categories, such as "documentary" and "comedy". '
                'Style descriptors such as "funny", "musical" and "animated" are also named genre entities.\n'
                
                # 'Just mentioning the verb "recommend" is not a named entity.'
                # 'Rating score platforms such as "Rotten Tomatoes" and "IMDB rating" are not relevant named entities. '
                # 'Mere queries on rating score platforms such as "Rotten Tomatoes" and "IMDB rating" are not relevant named entities. '
                'General reference to movie ratings such as "viewers\' rating" and "rating" are not named entities.'
            ),
            equivalents=['Rating'],
            demos=[
                dict(sentence='What is the viewers\' rating for the movie Sleeping Beauty?', entity_span="viewers' rating", label=NA),
                # dict(sentence='Name a popular romance movie from the 1990s', entity_span='popular', label=CORRECT),
                dict(sentence='What is the best superhero movie of all time?', entity_span='best', label=CORRECT),
                dict(sentence='Can you recommend a family-friendly animated movie with a catchy soundtrack?', entity_span='family-friendly', label=WRONG_TYPE, correct_type='Genre')
            ]
        ),
        'Year': dict(
            defn=(
                'A named year entity must be a specific year or a year range indicating movie release, such as "2000" or "last three years".\n'
                'Vague temporal descriptors such as "latest", "recent", "recent movies" and "come out" are not named entities. '
                # 'Release qualifiers such as "released" and "first" are also not named entities.'
                # this trailing `of any relevant type` helps reducing false corrections
                # 'Release qualifiers such as "released" and "first" are also not named entities of any relevant type. '
                # 'Release qualifiers such as "released", "new release", "re-release" and "first" are also not named entities. '
                'Release qualifiers such as "released", "new release", "re-release" and "first" are also not named entities of any relevant type. '
                'Note that a named movie title entity must be the name of a movie.'
            ),
            demos=[
                dict(sentence='What is a good character-driven drama film from the 1990s?', entity_span='1990s', label=CORRECT),
                dict(sentence='Who is the actor playing Spider-Man in the latest movie', entity_span='latest', label=NA),
                dict(sentence='Can you tell me the year in which the movie Pulp Fiction was released', entity_span='released', label=NA),
                dict(sentence='Do you have tickets available for the re-release of The Lion King?', entity_span='re-release', label=WRONG_TYPE, correct_type='other')
            ]
        ),
        'Genre': dict(
            defn=(
                'Named genre entities are broad and well-recognized movie categories, such as "documentary" and "comedy". '
                'Style descriptors such as "funny", "musical" and "animated" are also named genre entities. '
                'Film studios such as "Marvel" and "Disney" are also named genre entities.\n'
                'You should always drop starting evaluative descriptors such as "new" and "good" and trailing "movie" or "film" words from named genre entity spans.\n'
                # 'Ensure to distinguish named genre entities from named plot entities. '
                # 'Named plot entities are specific themes, topics, or elements of a movie storyline, such as "space exploration" and "zombie".'
            ),
            demos=[
                # dict(sentence='Can you recommend a feel-good comedy directed by the Coen Brothers and rated R?', entity_span='feel-good comedy', label=CORRECT),
                # dict(sentence='Can you recommend a popular action movie from the 1980s?', entity_span='popular action movie', label=WRONG_SPAN, correct_span='action')
                dict(sentence='Who directed the psychological thriller "Black Swan"?', entity_span='psychological thriller', label=CORRECT),
                # dict(sentence='Name a famous [heist] movie from the 2000s', entity_span='heist', label=WRONG_TYPE, correct_type='Plot'),
                dict(sentence='Can you provide a list of musical movies', entity_span='musical movies', label=WRONG_SPAN, correct_span='musical')
            ]
        ),
        'Director': dict(
            defn=(
                'A named director entity must be the name of a film director. '
                'Only first names, last names, and full names of directors are considered named director entities.\n'
                'You should always drop starting "new" and trailing "film" and "movie" words from named director entity spans.\n'
                'Ambiguous identifiers such as "director", "female filmmaker" and "female directors" are not named entities.'
            ),
            demos=[
                dict(sentence='Are there any movies directed by Steven Spielberg that I should watch?', entity_span='Steven Spielberg', label=CORRECT),
                dict(sentence='who directed the movie Inception', entity_span='directed', label=NA),
                dict(sentence='Can I see a list of theaters showing the new Christopher Nolan film?', entity_span='new Christopher Nolan film', label=WRONG_SPAN, correct_span='Christopher Nolan')
            ]
        ),
        'MPAA Rating': dict(
            defn=(
                'Only actual MPAA rating labels such as "G", "PG", "PG-13", "R", and "NC-17" are named MPAA rating entities.\n'
                'Rating interpretations such as "Parental Guidance Suggested" and "suitable for children" are not named MPAA rating entities.'
                # 'You should always drop starting and trailing "rated" or "MPAA rating of" words from named MPAA rating entity spans.\n'
            ),
            demos=[
                dict(sentence='Can you recommend a family-friendly movie rated PG with the song Sound of Silence?', entity_span='PG', label=CORRECT),
                # dict(sentence='What are some unforgettable G-rated movies from the 90s', entity_span='G-rated', label=WRONG_SPAN, correct_span='G'),
                dict(sentence='What is the best MPAA rating for a family movie', entity_span='MPAA rating', label=NA),
                # dict(sentence='please recommend a family-friendly fantasy film', entity_span='family-friendly', label=WRONG_TYPE, correct_type='Genre')
            ]
        ),
        'Plot': dict(
            defn=(
                # 'Named plot entities are specific themes, topics, or elements of a movie storyline, such as "space exploration" and "zombie".\n'
                'Named plot entities must be specific themes, topics, or elements of a movie storyline, such as "space exploration" and "zombie".\n'
                'Plot descriptors such as "captivating plot", "compelling plot", "intense plot" and "thrilling plot" are not relevant named entities. '
                'Narrative descriptors such as "plot", "summary" and "storyline" are also not named entities.'
            ),
            demos=[
                dict(sentence='Can you suggest a good heist movie with a PG-7 rating', entity_span='heist', label=CORRECT),
                # dict(sentence='Have any historical movies with a revenge plot been released in 2022', entity_span='revenge plot', label=WRONG_SPAN, correct_span='revenge')
                # dict(sentence='Who wrote the screenplay for the movie Jurassic Park', entity_span='screenplay', label=NA),
                dict(sentence='Can you give me a brief overview of the plot of "Forrest Gump"?', entity_span='plot', label=NA),
                dict(sentence='Could you recommend a movie with a strong female lead character and a compelling plot', entity_span='compelling plot', label=WRONG_TYPE, correct_type='other')
            ]
        ),
        'Actor': dict(
            defn=(
                'Only first names, last names, and full names of actors and actresses are considered named actor entities.\n'
                'Ambiguous identifiers such as "actor", "lead actor", "lead actress", "male lead", "cast" and "diverse cast" are not named entities.'
            ),
            demos=[
                # dict(sentence='Who played the lead role in the movie "The Godfather"?', entity_span='lead role', label=CORRECT),
                dict(sentence='What character did Tom Hanks play in "Forest Gump"?', entity_span='Tom Hanks', label=CORRECT),
                dict(sentence='What is the name of the actor who played the main character in "The Shawshank Redemption"?', entity_span='actor', label=NA),
                dict(sentence='Who are the primary actors in the film The Departed?', entity_span='primary actors', label=NA)
            ]
        ),
        'Trailer': dict(
            defn=(
                'A named movie trailer entity is a term that indicates a segment or a clip from a movie, such as "clip", "trailer" and "preview".\n'
                'You should always drop starting "captivating" or "fun" descriptive words from named movie trailer entity spans.'
                # 'Named movie trailer entity spans must not have starting or trailing "movie" or "film" words.'
            ),
            demos=[
                dict(sentence='Can you show me the trailer for the movie "The Matrix"?', entity_span='trailer', label=CORRECT),
                # dict(sentence='Can you play the trailer for the latest Mission: Impossible movie?', entity_span='trailer', label=CORRECT),
                # dict(sentence='Can you play the movie clip where Yoda gives his famous advice to Luke Skywalker?', entity_span='movie clip', label=WRONG_SPAN, correct_span='clip')
                dict(sentence='Could you show me a preview for a new release that leaves something to the imagination?', entity_span='preview', label=CORRECT)
            ]
        ),
        'Song': dict(
            defn=(
                'A named song entity must be the name of a song.\n'
                # 'You should always drop starting or trailing "song" words from named song entity spans.\n'
                'Song attributes such as "iconic song," "love song," "famous song," and "original song" are not relevant named entities. '
                'Musical descriptors such as "song", "theme song" and "soundtrack" are also not named entities. '
            ),
            demos=[
                dict(sentence='Is Every Breath You Take featured in any popular movie soundtracks?', entity_span='Every Breath You Take', label=CORRECT),
                # dict(sentence='Is there a specific song featured in the classic movie Road Trip?', entity_span='specific song', label=NA),
                dict(sentence='What is the theme song of the James Bond movie "Skyfall"?', entity_span='theme song', label=NA),
                dict(sentence='Can you play a song from the movie "The Sound of Music"?', entity_span='song', label=NA),
                # dict(sentence='Can you suggest a movie with a famous soundtrack?', entity_span='famous soundtrack', label=NA)
                dict(sentence='Can you suggest a movie with a famous soundtrack?', entity_span='famous soundtrack', label=WRONG_TYPE, correct_type='other')
            ]
        ),
        'Review': dict(
            defn=(
                'A named review entity must be a recognition or a detailed, qualitative assessment of a movie, such as "funniest of all time" and "incredible visual effects". '
                'Named review entities must describe specific aspects of a movie, such as "most awards".\n'
                'Ensure to distinguish named review entities from named viewers\' rating entities. '
                'Short evaluative descriptors such as "good", "highest rated" and "top-rated" are named viewers\' rating entities. '
                'Performance metrics such as "highest-grossing" and "best-rated" are also named viewers\' rating entities.\n'
                'Ambiguous identifiers such as "review" are not named entities.'
            ),
            demos=[
                # dict(sentence='What movie won the Best Picture award at the Oscars in 2019?', entity_span='Oscars', label=WRONG_TYPE, correct_type='Other'),
                # dict(sentence='What is the best superhero movie of all time?', entity_span='best', label=WRONG_TYPE, correct_type="Viewers' Rating"),
                # dict(sentence='I want to watch a horror movie with a high Rotten Tomatoes rating.', entity_span='high Rotten Tomatoes rating', label=WRONG_TYPE, correct_type="Viewers' Rating")
                dict(sentence='Which movie has the best special effects of all time', entity_span='best special effects', label=CORRECT),
                dict(sentence='Can you recommend a good movie from the 1980s', entity_span='good', label=WRONG_TYPE, correct_type="Viewers' Rating"),
                dict(sentence="What's your favorite movie review website?", entity_span='review', label=NA)
            ]
        ),
        'Character': dict(
            defn=(
                'A named character entity must be the name of a character.\n'
                'Character qualifiers such as "main antagonist", "villain character", "strong female lead character" are not relevant named entities. '
                'Ambiguous identifiers such as "character", "main character" and "lead role" are also not named entities.'
            ),
            demos=[
                dict(sentence='who is the actor in the lead role in forrest gump', entity_span='lead role', label=NA),
                dict(sentence='Who is the main character in the Star Wars series', entity_span='main character', label=NA),
                dict(sentence='Tell me about the actress who played Hermione Granger in the Harry Potter series.', entity_span='Hermione Granger', label=CORRECT),
                dict(sentence='Could you recommend a family-friendly movie with a strong female character as the lead?', entity_span='strong female character', label=WRONG_TYPE, correct_type='other')
            ]
        )
    },
    '24-02-06_NER-Dataset_{fmt=n-p2,#l=3,de=T,lc=T}_#et=4.5': {
        'Title': dict(
            defn=(
                'A named title entity must be the name of a movie. '
                'You should always drop starting "latest" or "new" and trailing "movie" or "film" words from named movie entity spans.\n'
                # 'Awards such as "Oscar" are not relevant named entities. '
                'General reference to movies such as "latest film" and "newest movie" are not named entities.'
            ),
            demos=[
                dict(sentence='What character did Tom Hanks play in "Forest Gump"?', entity_span='Forest Gump', label=CORRECT),
                # dict(sentence='What film won the Best Picture award at the Oscars in 2019?', entity_span='Oscars', label=WRONG_TYPE, correct_type='Other')
                dict(sentence='Can I find tickets for the new James Bond movie?', entity_span='the new James Bond movie', label=WRONG_SPAN, correct_span='James Bond'),
                dict(sentence='Show me the trailer for a movie directed by GP and released in 2019', entity_span='movie', label=NA)
            ]
        ),
        "Viewers' Rating": dict(
            defn=(
                'A named viewers\' rating entity can be a rating score such as "9 out of 10". '
                'Endorsement phrases such as "must see" and "highly recommended" are also named viewers\' rating entities. '
                'Evaluative descriptors such as "best", "good", "popular", "famous" and "favorite" are named viewers\' rating entities. '
                'Performance metrics such as "highest-grossing" and "best-rated" are also named viewers\' rating entities.\n'
                'You should capture the complete descriptive phrase such as "high viewers\' rating" as opposed to "high".\n'
                'Ambiguous identifiers such as "rating" and "viewers\' rating" are not named entities.'
                # 'General references to movie ratings such as "viewers\' rating" and "rating" are not named entities.'
                # 'Simply querying about movie ratings such as "viewers\' rating" and "rating" are not named entities.'
            ),
            equivalents=['Rating'],
            demos=[
                dict(sentence='What is the viewers\' rating for the movie Sleeping Beauty?', entity_span="viewers' rating", label=NA),
                dict(sentence="Show me a Marielle Heller film released in 2019 with a high Viewers' Rating", entity_span='high', label=WRONG_SPAN, correct_span='high Viewers\' Rating'),
                # dict(sentence='Name a popular romance movie from the 1990s', entity_span='popular', label=CORRECT),
                dict(sentence='What is the best superhero movie of all time?', entity_span='best', label=CORRECT),
                dict(sentence='I want to watch a horror movie with a high Rotten Tomatoes rating.', entity_span='high Rotten Tomatoes rating', label=CORRECT)
            ]
        ),
        'Year': dict(
            defn=(
                'A named year entity must be a specific year or a year range indicating movie release, such as "2000" or "last three years".\n'
                'You should always drop starting "the" word from named year entity spans.\n'
                'Vague temporal descriptors such as "latest", "recent", "recent movies" and "come out" are not named entities. '
                'Release qualifiers such as "released", "new release", "re-release" and "first" are also not named entities. '
                'Ambiguous identifiers such as "year" are not named entities.'
            ),
            demos=[
                dict(sentence='What is a good character-driven drama film from the 1990s?', entity_span='1990s', label=CORRECT),
                dict(sentence='Show me a sport movie from the 90s with an intense plot and a catchy trailer.', entity_span='the 90s', label=WRONG_SPAN, correct_span='90s'),
                dict(sentence='Who is the actor playing Spider-Man in the latest movie', entity_span='latest', label=NA),
                dict(sentence='Can you tell me the year in which the movie Pulp Fiction was released', entity_span='released', label=NA),
                dict(sentence='who directed the movie Aquaman and what year was it released', entity_span='year', label=NA)
            ]
        ),
        'Genre': dict(
            defn=(
                'Named genre entities are broad and well-recognized movie categories, such as "documentary" and "comedy". '
                'Style descriptors such as "funny", "musical" and "animated" are also named genre entities. '
                'Film studios such as "Marvel" and "Disney" are also named genre entities.\n'
                'You should always drop starting evaluative descriptors such as "new" and "good" and trailing "movie" or "film" words from named genre entity spans.\n'
                'Ensure to distinguish named genre entities from named plot entities. '
                'Named plot entities are specific themes, topics, or elements of a movie storyline, such as "space exploration" and "zombie".'
            ),
            demos=[
                # dict(sentence='Can you recommend a feel-good comedy directed by the Coen Brothers and rated R?', entity_span='feel-good comedy', label=CORRECT),
                # dict(sentence='Can you recommend a popular action movie from the 1980s?', entity_span='popular action movie', label=WRONG_SPAN, correct_span='action')
                dict(sentence='Who directed the psychological thriller "Black Swan"?', entity_span='psychological thriller', label=CORRECT),
                dict(sentence='Name a famous [heist] movie from the 2000s', entity_span='heist', label=WRONG_TYPE, correct_type='Plot'),
                dict(sentence='Can you provide a list of musical movies', entity_span='musical movies', label=WRONG_SPAN, correct_span='musical')
            ]
        ),
        'Director': dict(
            defn=(
                'A named director entity must be the name of a film director. '
                'Only first names, last names, and full names of directors are considered named director entities.\n'
                # 'You should always drop starting "new" and trailing "film" and "movie" words from named director entity spans.\n'
                'Ambiguous identifiers such as "director", "female filmmaker" and "female directors" are not named entities.'
            ),
            demos=[
                dict(sentence='Are there any movies directed by Steven Spielberg that I should watch?', entity_span='Steven Spielberg', label=CORRECT),
                dict(sentence='who directed the movie Inception', entity_span='directed', label=NA),
                # dict(sentence='Can I see a list of theaters showing the new Christopher Nolan film?', entity_span='new Christopher Nolan film', label=WRONG_SPAN, correct_span='Christopher Nolan')
            ]
        ),
        'MPAA Rating': dict(
            defn=(
                'Only actual MPAA rating labels such as "G", "PG", "PG-13", "R", and "NC-17" are named MPAA rating entities.\n'
                'You should always drop starting and trailing "rated" or "MPAA rating of" words from named MPAA rating entity spans.\n'
                # 'Rating interpretations such as "Parental Guidance Suggested" and "suitable for children" are not named MPAA rating entities.'
                'Rating interpretations such as "Parental Guidance Suggested" and "suitable for children" are not relevant named entities.'
            ),
            demos=[
                dict(sentence='Can you recommend a family-friendly movie rated PG with the song Sound of Silence?', entity_span='PG', label=CORRECT),
                dict(sentence='What are some unforgettable G-rated movies from the 90s', entity_span='G-rated', label=WRONG_SPAN, correct_span='G'),
                # dict(sentence='What is the best MPAA rating for a family movie', entity_span='MPAA rating', label=NA),
                dict(sentence='I want to see a trailer for The Wolf of Wall Street, rated R', entity_span='rated R', label=WRONG_SPAN, correct_span='R'),
                dict(sentence='who directed the movie Amadeus and is it appropriate for mature audiences', entity_span='mature audiences', label=WRONG_TYPE, correct_type='other')
                # dict(sentence='please recommend a family-friendly fantasy film', entity_span='family-friendly', label=WRONG_TYPE, correct_type='Genre')
            ]
        ),
        'Plot': dict(
            defn=(
                'Named plot entities must be specific themes, topics, or elements of a movie storyline, such as "space exploration" and "zombie".\n'
                'Plot descriptors such as "captivating plot", "compelling plot", "intense plot" and "thrilling plot" are not relevant named entities. '
                'Narrative descriptors such as "plot", "summary" and "storyline" are also not named entities.'
            ),
            demos=[
                dict(sentence='Can you suggest a good heist movie with a PG-7 rating', entity_span='heist', label=CORRECT),
                # dict(sentence='Have any historical movies with a revenge plot been released in 2022', entity_span='revenge plot', label=WRONG_SPAN, correct_span='revenge')
                # dict(sentence='Who wrote the screenplay for the movie Jurassic Park', entity_span='screenplay', label=NA),
                dict(sentence='Can you give me a brief overview of the plot of "Forrest Gump"?', entity_span='plot', label=NA),
                dict(sentence='Could you recommend a movie with a strong female lead character and a compelling plot', entity_span='compelling plot', label=WRONG_TYPE, correct_type='other')
            ]
        ),
        'Actor': dict(
            defn=(
                'A named actor entity must be the name of an actor or actress. '
                'Only first names, last names, and full names of actors and actresses are considered named actor entities.\n'
                # 'Names of Artists such as "Beyonce" are not named actor entities.'
                # 'Ambiguous identifiers such as "actor", "lead actor", "lead actress", "male lead", "cast" and "diverse cast" are not named entities.'
            ),
            demos=[
                # dict(sentence='Who played the lead role in the movie "The Godfather"?', entity_span='lead role', label=CORRECT),
                dict(sentence='What character did Tom Hanks play in "Forest Gump"?', entity_span='Tom Hanks', label=CORRECT),
                dict(sentence='What is the name of the actor who played the main character in "The Shawshank Redemption"?', entity_span='actor', label=NA),
                # dict(sentence='Who are the primary actors in the film The Departed?', entity_span='primary actors', label=NA)
            ]
        ),
        'Trailer': dict(
            defn=(
                'A named movie trailer entity is a term that indicates a segment or a clip from a movie, such as "clip", "trailer" and "preview".\n'
                'You should always drop starting "captivating" or "fun" descriptive words from named trailer entity spans.\n'
                # 'Named movie trailer entity spans must not have starting or trailing "movie" or "film" words.'
                'You should always drop starting "movie" or "film" words from named trailer entity spans.'
            ),
            demos=[
                dict(sentence='Can you show me the trailer for the movie "The Matrix"?', entity_span='trailer', label=CORRECT),
                # dict(sentence='Can you play the trailer for the latest Mission: Impossible movie?', entity_span='trailer', label=CORRECT),
                dict(sentence='Can you play the movie clip where Yoda gives his famous advice to Luke Skywalker?', entity_span='movie clip', label=WRONG_SPAN, correct_span='clip')
                # dict(sentence='Could you show me a preview for a new release that leaves something to the imagination?', entity_span='preview', label=CORRECT)
            ]
        ),
        'Song': dict(
            defn=(
                'A named song entity must be the name of a song.\n'
                # 'You should always drop starting or trailing "song" words from named song entity spans.\n'
                'Song attributes such as "iconic song," "love song," "famous song," and "original song" are not relevant named entities. '
                'Names of Artists such as "Beyonce" are also not relevant named entities. '
                'Musical descriptors such as "song", "theme song" and "soundtrack" are not named entities.'
            ),
            demos=[
                dict(sentence='Is Every Breath You Take featured in any popular movie soundtracks?', entity_span='Every Breath You Take', label=CORRECT),
                # dict(sentence='Is there a specific song featured in the classic movie Road Trip?', entity_span='specific song', label=NA),
                dict(sentence='What is the theme song of the James Bond movie "Skyfall"?', entity_span='theme song', label=NA),
                dict(sentence='Can you play a song from the movie "The Sound of Music"?', entity_span='song', label=NA),
                # dict(sentence='Can you suggest a movie with a famous soundtrack?', entity_span='famous soundtrack', label=NA),
                dict(sentence='Can you suggest a movie with a famous soundtrack?', entity_span='famous soundtrack', label=WRONG_TYPE, correct_type='other')
            ]
        ),
        'Review': dict(
            defn=(
                'A named review entity must be a recognition or a detailed, qualitative assessment of a movie, such as "funniest of all time" and "incredible visual effects". '
                'Named review entities must describe specific aspects of a movie, such as "most awards".\n'
                'Ensure to distinguish named review entities from named viewers\' rating entities. '
                'Short evaluative descriptors such as "good", "highest rated" and "top-rated" are named viewers\' rating entities. '
                'Performance metrics such as "highest-grossing" and "best-rated" are also named viewers\' rating entities.\n'
                'Ambiguous identifiers such as "review" are not named entities.'
            ),
            demos=[
                # dict(sentence='What movie won the Best Picture award at the Oscars in 2019?', entity_span='Oscars', label=WRONG_TYPE, correct_type='Other'),
                # dict(sentence='What is the best superhero movie of all time?', entity_span='best', label=WRONG_TYPE, correct_type="Viewers' Rating"),
                # dict(sentence='I want to watch a horror movie with a high Rotten Tomatoes rating.', entity_span='high Rotten Tomatoes rating', label=WRONG_TYPE, correct_type="Viewers' Rating")
                dict(sentence='Which movie has the best special effects of all time', entity_span='best special effects', label=CORRECT),
                dict(sentence='Can you recommend a good movie from the 1980s', entity_span='good', label=WRONG_TYPE, correct_type="Viewers' Rating"),
                # dict(sentence="What's your favorite movie review website?", entity_span='review', label=NA)
            ]
        ),
        'Character': dict(
            defn=(
                'A named character entity must be the name of a character.\n'
                'Character qualifiers such as "main antagonist", "villain character", "strong female lead character" are not relevant named entities. '
                'Ambiguous identifiers such as "character", "main character" and "lead role" are also not named entities.'
            ),
            demos=[
                dict(sentence='who is the actor in the lead role in forrest gump', entity_span='lead role', label=NA),
                dict(sentence='Who is the main character in the Star Wars series', entity_span='main character', label=NA),
                dict(sentence='Tell me about the actress who played Hermione Granger in the Harry Potter series.', entity_span='Hermione Granger', label=CORRECT),
                dict(sentence='Could you recommend a family-friendly movie with a strong female character as the lead?', entity_span='strong female character', label=WRONG_TYPE, correct_type='other')
            ]
        )
    },
    '24-02-06_NER-Dataset_{fmt=n-p2,#l=3,de=s,lc=T}': {
        'Title': dict(
            defn=(
                'A named title entity must be the name of a movie. '
                'You should always drop starting "latest" or "new" and trailing "movie" or "film" words from named movie entity spans.\n'
                # 'Awards such as "Oscar" are not relevant named entities. '
                'General reference to movies such as "latest film" and "newest movie" are not named entities.'
            ),
            demos=[
                dict(sentence='What character did Tom Hanks play in "Forest Gump"?', entity_span='Forest Gump', label=CORRECT),
                # dict(sentence='What film won the Best Picture award at the Oscars in 2019?', entity_span='Oscars', label=WRONG_TYPE, correct_type='Other')
                dict(sentence='Can I find tickets for the new James Bond movie?', entity_span='the new James Bond movie', label=WRONG_SPAN, correct_span='James Bond'),
                dict(sentence='Show me the trailer for a movie directed by GP and released in 2019', entity_span='movie', label=NA)
            ]
        ),
        "Viewers' Rating": dict(
            defn=(
                'A named viewers\' rating entity can be a rating score such as "9 out of 10". '
                'Endorsement phrases such as "must see" and "highly recommended" are also named viewers\' rating entities. '
                'Evaluative descriptors such as "best", "good", "popular", "famous" and "favorite" are named viewers\' rating entities. '
                'Performance metrics such as "highest-grossing" and "best-rated" are also named viewers\' rating entities.\n'
                'You should capture the complete descriptive phrase such as "high viewers\' rating" as opposed to "high".\n'
                'Ambiguous identifiers such as "rating" and "viewers\' rating" are not named entities.'
                # 'General references to movie ratings such as "viewers\' rating" and "rating" are not named entities.'
                # 'Simply querying about movie ratings such as "viewers\' rating" and "rating" are not named entities.'
            ),
            equivalents=['Rating'],
            demos=[
                dict(sentence='What is the viewers\' rating for the movie Sleeping Beauty?', entity_span="viewers' rating", label=NA),
                dict(sentence="Show me a Marielle Heller film released in 2019 with a high Viewers' Rating", entity_span='high', label=WRONG_SPAN, correct_span='high Viewers\' Rating'),
                # dict(sentence='Name a popular romance movie from the 1990s', entity_span='popular', label=CORRECT),
                dict(sentence='What is the best superhero movie of all time?', entity_span='best', label=CORRECT),
                dict(sentence='Are there any good teen movies that came out in 2014', entity_span='good', label=CORRECT),
                dict(sentence='I want to watch a horror movie with a high Rotten Tomatoes rating.', entity_span='high Rotten Tomatoes rating', label=CORRECT)
            ]
        ),
        'Year': dict(
            defn=(
                'A named year entity must be a specific year or a year range indicating movie release, such as "2000" or "last three years".\n'
                'You should always drop starting "the" word from named year entity spans.\n'
                'Vague temporal descriptors such as "latest", "recent", "recent movies" and "come out" are not named entities. '
                'Release qualifiers such as "released", "new release", "re-release" and "first" are also not named entities. '
                'Ambiguous identifiers such as "year" are not named entities.'
            ),
            demos=[
                dict(sentence='What is a good character-driven drama film from the 1990s?', entity_span='1990s', label=CORRECT),
                dict(sentence='Show me a sport movie from the 90s with an intense plot and a catchy trailer.', entity_span='the 90s', label=WRONG_SPAN, correct_span='90s'),
                dict(sentence='Who is the actor playing Spider-Man in the latest movie', entity_span='latest', label=NA),
                dict(sentence='Can you tell me the year in which the movie Pulp Fiction was released', entity_span='released', label=NA),
                dict(sentence='who directed the movie Aquaman and what year was it released', entity_span='year', label=NA)
            ]
        ),
        'Genre': dict(
            defn=(
                'Named genre entities are broad and well-recognized movie categories, such as "documentary" and "comedy". '
                'Style descriptors such as "funny", "musical", "thrilling" and "animated" are also named genre entities. '
                'Film studios such as "Marvel" and "Disney" are also named genre entities.\n'
                'You should always drop starting evaluative descriptors such as "new" and "good" and trailing "movie" or "film" words from named genre entity spans.\n'
                'Ensure to distinguish named genre entities from named plot entities. '
                'Named plot entities are specific themes, topics, or elements of a movie storyline, such as "space exploration" and "zombie".'
                # 'Plot descriptors such as "captivating plot", "compelling plot", "intense plot" and "thrilling plot" are not relevant named entities. '
            ),
            demos=[
                # dict(sentence='Can you recommend a feel-good comedy directed by the Coen Brothers and rated R?', entity_span='feel-good comedy', label=CORRECT),
                # dict(sentence='Can you recommend a popular action movie from the 1980s?', entity_span='popular action movie', label=WRONG_SPAN, correct_span='action')
                dict(sentence='Who directed the psychological thriller "Black Swan"?', entity_span='psychological thriller', label=CORRECT),
                dict(sentence='Name a famous [heist] movie from the 2000s', entity_span='heist', label=WRONG_TYPE, correct_type='Plot'),
                dict(sentence='Can you provide a list of musical movies', entity_span='musical movies', label=WRONG_SPAN, correct_span='musical')
            ]
        ),
        'Director': dict(
            defn=(
                'A named director entity must be the name of a film director. '
                'Only first names, last names, and full names of directors are considered named director entities.\n'
                # 'You should always drop starting "new" and trailing "film" and "movie" words from named director entity spans.\n'
                'Ambiguous identifiers such as "director", "female filmmaker" and "female directors" are not named entities.'
            ),
            demos=[
                dict(sentence='Are there any movies directed by Steven Spielberg that I should watch?', entity_span='Steven Spielberg', label=CORRECT),
                dict(sentence='who directed the movie Inception', entity_span='directed', label=NA),
                # dict(sentence='Can I see a list of theaters showing the new Christopher Nolan film?', entity_span='new Christopher Nolan film', label=WRONG_SPAN, correct_span='Christopher Nolan')
            ]
        ),
        'MPAA Rating': dict(
            defn=(
                'Only actual MPAA rating labels such as "G", "PG", "PG-13", "R", and "NC-17" are named MPAA rating entities.\n'
                'You should always drop starting and trailing "rated" or "MPAA rating of" words from named MPAA rating entity spans.\n'
                # 'Rating interpretations such as "Parental Guidance Suggested" and "suitable for children" are not named MPAA rating entities.'
                'Rating interpretations such as "Parental Guidance Suggested" and "suitable for children" are not relevant named entities.'
            ),
            demos=[
                dict(sentence='Can you recommend a family-friendly movie rated PG with the song Sound of Silence?', entity_span='PG', label=CORRECT),
                dict(sentence='What are some unforgettable G-rated movies from the 90s', entity_span='G-rated', label=WRONG_SPAN, correct_span='G'),
                # dict(sentence='What is the best MPAA rating for a family movie', entity_span='MPAA rating', label=NA),
                dict(sentence='I want to see a trailer for The Wolf of Wall Street, rated R', entity_span='rated R', label=WRONG_SPAN, correct_span='R'),
                dict(sentence='who directed the movie Amadeus and is it appropriate for mature audiences', entity_span='mature audiences', label=WRONG_TYPE, correct_type='other')
                # dict(sentence='please recommend a family-friendly fantasy film', entity_span='family-friendly', label=WRONG_TYPE, correct_type='Genre')
            ]
        ),
        'Plot': dict(
            defn=(
                'Named plot entities must be specific themes, topics, or elements of a movie storyline, such as "space exploration" and "zombie".\n'
                'Plot descriptors such as "captivating plot", "compelling plot", "intense plot" and "thrilling plot" are not relevant named entities. '
                'Narrative descriptors such as "plot", "summary" and "storyline" are also not named entities.'
            ),
            demos=[
                dict(sentence='Can you suggest a good heist movie with a PG-7 rating', entity_span='heist', label=CORRECT),
                # dict(sentence='Have any historical movies with a revenge plot been released in 2022', entity_span='revenge plot', label=WRONG_SPAN, correct_span='revenge')
                # dict(sentence='Who wrote the screenplay for the movie Jurassic Park', entity_span='screenplay', label=NA),
                dict(sentence='Can you give me a brief overview of the plot of "Forrest Gump"?', entity_span='plot', label=NA),
                dict(sentence='Could you recommend a movie with a strong female lead character and a compelling plot', entity_span='compelling plot', label=WRONG_TYPE, correct_type='other')
            ]
        ),
        'Actor': dict(
            defn=(
                'A named actor entity must be the name of an actor or actress. '
                'Only first names, last names, and full names of actors and actresses are considered named actor entities.\n'
                # 'You should differentiate named actor entities from named director entities based on the context of the query. '
                # 'Names of Artists such as "Beyonce" are not named actor entities.'
                # 'Ambiguous identifiers such as "actor", "lead actor", "lead actress", "male lead", "cast" and "diverse cast" are not named entities.'
            ),
            demos=[
                # dict(sentence='Who played the lead role in the movie "The Godfather"?', entity_span='lead role', label=CORRECT),
                dict(sentence='What character did Tom Hanks play in "Forest Gump"?', entity_span='Tom Hanks', label=CORRECT),
                dict(sentence='What is the name of the actor who played the main character in "The Shawshank Redemption"?', entity_span='actor', label=NA),
                # dict(sentence='Who are the primary actors in the film The Departed?', entity_span='primary actors', label=NA)
            ]
        ),
        'Trailer': dict(
            defn=(
                'A named movie trailer entity is a term that indicates a segment or a clip from a movie, such as "clip", "trailer", "teaser" and "preview".\n'
                'You should always drop starting "captivating" or "fun" descriptive words from named trailer entity spans.\n'
                # 'Named movie trailer entity spans must not have starting or trailing "movie" or "film" words.'
                'You should always drop starting "movie" or "film" words from named trailer entity spans.'
            ),
            demos=[
                dict(sentence='Can you show me the trailer for the movie "The Matrix"?', entity_span='trailer', label=CORRECT),
                # dict(sentence='Can you play the trailer for the latest Mission: Impossible movie?', entity_span='trailer', label=CORRECT),
                dict(sentence='Can you play the movie clip where Yoda gives his famous advice to Luke Skywalker?', entity_span='movie clip', label=WRONG_SPAN, correct_span='clip')
                # dict(sentence='Could you show me a preview for a new release that leaves something to the imagination?', entity_span='preview', label=CORRECT)
            ]
        ),
        'Song': dict(
            defn=(
                'A named song entity must be the name of a song.\n'
                # 'You should always drop starting or trailing "song" words from named song entity spans.\n'
                'Song attributes such as "iconic song," "love song," "famous song," and "original song" are not relevant named entities. '
                'Music genres such as "EDM" and "Jazz" are also not relevant named entities. '
                'Names of Artists such as "Beyonce" are not relevant named entities. '
                'Musical descriptors such as "song", "theme song" and "soundtrack" are also not named entities.'
            ),
            demos=[
                dict(sentence='Is Every Breath You Take featured in any popular movie soundtracks?', entity_span='Every Breath You Take', label=CORRECT),
                # dict(sentence='Is there a specific song featured in the classic movie Road Trip?', entity_span='specific song', label=NA),
                dict(sentence='What is the theme song of the James Bond movie "Skyfall"?', entity_span='theme song', label=NA),
                dict(sentence='Can you play a song from the movie "The Sound of Music"?', entity_span='song', label=NA),
                # dict(sentence='Can you suggest a movie with a famous soundtrack?', entity_span='famous soundtrack', label=NA),
                dict(sentence='Can you suggest a movie with a famous soundtrack?', entity_span='famous soundtrack', label=WRONG_TYPE, correct_type='other')
            ]
        ),
        'Review': dict(
            defn=(
                'A named review entity must be a recognition or a detailed, qualitative assessment of a movie, such as "funniest of all time" and "incredible visual effects". '
                'Named review entities must describe specific aspects of a movie, such as "most awards".\n'
                'Ensure to distinguish named review entities from named viewers\' rating entities. '
                'Short evaluative descriptors such as "good", "highest rated" and "top-rated" are named viewers\' rating entities. '
                'Performance metrics such as "highest-grossing" and "best-rated" are also named viewers\' rating entities.\n'
                'Ambiguous identifiers such as "review" are not named entities.\n'
                'Named genre entities are broad and well-recognized movie categories, such as "documentary" and "comedy". '
                # 'Named director entities must be names of film directors.'
            ),
            demos=[
                # dict(sentence='What movie won the Best Picture award at the Oscars in 2019?', entity_span='Oscars', label=WRONG_TYPE, correct_type='Other'),
                # dict(sentence='What is the best superhero movie of all time?', entity_span='best', label=WRONG_TYPE, correct_type="Viewers' Rating"),
                # dict(sentence='I want to watch a horror movie with a high Rotten Tomatoes rating.', entity_span='high Rotten Tomatoes rating', label=WRONG_TYPE, correct_type="Viewers' Rating")
                dict(sentence='Which movie has the best special effects of all time', entity_span='best special effects', label=CORRECT),
                dict(sentence='Can you recommend a good movie from the 1980s', entity_span='good', label=WRONG_TYPE, correct_type="Viewers' Rating"),
                # dict(sentence="What's your favorite movie review website?", entity_span='review', label=NA)
            ]
        ),
        'Character': dict(
            defn=(
                'A named character entity must be the name of a character.\n'
                'Character qualifiers such as "main antagonist", "villain character", "strong female lead character" are not relevant named entities. '
                'Ambiguous identifiers such as "character", "main character" and "lead role" are also not named entities.'
            ),
            demos=[
                dict(sentence='who is the actor in the lead role in forrest gump', entity_span='lead role', label=NA),
                dict(sentence='Who is the main character in the Star Wars series', entity_span='main character', label=NA),
                dict(sentence='Tell me about the actress who played Hermione Granger in the Harry Potter series.', entity_span='Hermione Granger', label=CORRECT),
                dict(sentence='Could you recommend a family-friendly movie with a strong female character as the lead?', entity_span='strong female character', label=WRONG_TYPE, correct_type='other')
            ]
        )
    },
    '24-02-07_NER-Dataset_{fmt=n-p2,#l=3,ap={dc=T,de=s},lc=T}': {
        'Title': dict(
            defn=(
                'A named title entity must be the name of a movie. '
                # 'You should always drop starting "latest" or "new" and trailing "movie" or "film" words from named movie entity spans.\n'
                # 'Awards such as "Oscar" are not relevant named entities. '
                # 'General reference to movies such as "latest film" and "newest movie" are not named entities.'
            ),
            equivalents=['Movie'],
            demos=[
                dict(sentence='What character did Tom Hanks play in "Forest Gump"?', entity_span='Forest Gump', label=CORRECT),
                # dict(sentence='What film won the Best Picture award at the Oscars in 2019?', entity_span='Oscars', label=WRONG_TYPE, correct_type='Other')
                # dict(sentence='Can I find tickets for the new James Bond movie?', entity_span='the new James Bond movie', label=WRONG_SPAN, correct_span='James Bond'),
                # dict(sentence='Show me the trailer for a movie directed by GP and released in 2019', entity_span='movie', label=NA)
                dict(sentence="I'm bored, can you tell me the plot of the movie Ghostbusters?", entity_span='Ghostbusters', label=CORRECT)
            ]
        ),
        "Viewers' Rating": dict(
            defn=(
                'A named viewers\' rating entity can be a rating score such as "9 out of 10". '
                'Endorsement phrases such as "must see" and "highly recommended" are also named viewers\' rating entities. '
                'Evaluative descriptors such as "best", "good", "popular", "famous" and "favorite" are named viewers\' rating entities. '
                'Performance metrics such as "highest-grossing" and "best-rated" are also named viewers\' rating entities.\n'
                'You should capture the complete descriptive phrase such as "high viewers\' rating" as opposed to "high".\n'
                'Ambiguous identifiers such as "rating" and "viewers\' rating" are not named entities.'
                # 'General references to movie ratings such as "viewers\' rating" and "rating" are not named entities.'
                # 'Simply querying about movie ratings such as "viewers\' rating" and "rating" are not named entities.'
            ),
            equivalents=['Rating'],
            demos=[
                dict(sentence='What is the viewers\' rating for the movie Sleeping Beauty?', entity_span="viewers' rating", label=NA),
                dict(sentence="Show me a Marielle Heller film released in 2019 with a high Viewers' Rating", entity_span='high', label=WRONG_SPAN, correct_span='high Viewers\' Rating'),
                # dict(sentence='Name a popular romance movie from the 1990s', entity_span='popular', label=CORRECT),
                dict(sentence='What is the best superhero movie of all time?', entity_span='best', label=CORRECT),
                dict(sentence='Are there any good teen movies that came out in 2014', entity_span='good', label=CORRECT),
                dict(sentence='I want to watch a horror movie with a high Rotten Tomatoes rating.', entity_span='high Rotten Tomatoes rating', label=CORRECT)
            ]
        ),
        'Year': dict(
            defn=(
                'A named year entity must be a specific year or a year range indicating movie release, such as "2000" or "last three years".\n'
                'You should always drop starting "the" word from named year entity spans.\n'
                'Vague temporal descriptors such as "latest", "recent", "recent movies" and "come out" are not named entities. '
                'Release qualifiers such as "released", "new release", "re-release" and "first" are also not named entities. '
                'Ambiguous identifiers such as "year" are not named entities.'
            ),
            demos=[
                dict(sentence='What is a good character-driven drama film from the 1990s?', entity_span='1990s', label=CORRECT),
                dict(sentence='Show me a sport movie from the 90s with an intense plot and a catchy trailer.', entity_span='the 90s', label=WRONG_SPAN, correct_span='90s'),
                dict(sentence='Who is the actor playing Spider-Man in the latest movie', entity_span='latest', label=NA),
                dict(sentence='Can you tell me the year in which the movie Pulp Fiction was released', entity_span='released', label=NA),
                dict(sentence='who directed the movie Aquaman and what year was it released', entity_span='year', label=NA)
            ]
        ),
        'Genre': dict(
            defn=(
                'Named genre entities are broad and well-recognized movie categories, such as "documentary" and "comedy". '
                'Style descriptors such as "funny", "musical", "thrilling" and "animated" are also named genre entities. '
                'Film studios such as "Marvel" and "Disney" are also named genre entities.\n'
                'You should always drop starting evaluative descriptors such as "new" and "good" and trailing "movie" or "film" words from named genre entity spans.\n'
                'Ensure to distinguish named genre entities from named plot entities. '
                'Named plot entities are specific themes, topics, or elements of a movie storyline, such as "space exploration" and "zombie".'
                # 'Plot descriptors such as "captivating plot", "compelling plot", "intense plot" and "thrilling plot" are not relevant named entities.\n'
                # 'Music genres such as "disco" and "Jazz" are also not relevant named entities.'
            ),
            name='movie genre',
            demos=[
                # dict(sentence='Can you recommend a feel-good comedy directed by the Coen Brothers and rated R?', entity_span='feel-good comedy', label=CORRECT),
                # dict(sentence='Can you recommend a popular action movie from the 1980s?', entity_span='popular action movie', label=WRONG_SPAN, correct_span='action')
                dict(sentence='Who directed the psychological thriller "Black Swan"?', entity_span='psychological thriller', label=CORRECT),
                dict(sentence='Name a famous [heist] movie from the 2000s', entity_span='heist', label=WRONG_TYPE, correct_type='Plot'),
                dict(sentence='Can you provide a list of musical movies', entity_span='musical movies', label=WRONG_SPAN, correct_span='musical')
            ]
        ),
        'Director': dict(
            defn=(
                'A named director entity must be the name of a film director. '
                'Only first names, last names, and full names of directors are considered named director entities.\n'
                # 'You should always drop starting "new" and trailing "film" and "movie" words from named director entity spans.\n'
                'Ambiguous identifiers such as "director", "female filmmaker" and "female directors" are not named entities.'
            ),
            demos=[
                dict(sentence='Are there any movies directed by Steven Spielberg that I should watch?', entity_span='Steven Spielberg', label=CORRECT),
                dict(sentence='who directed the movie Inception', entity_span='directed', label=NA),
                # dict(sentence='Can I see a list of theaters showing the new Christopher Nolan film?', entity_span='new Christopher Nolan film', label=WRONG_SPAN, correct_span='Christopher Nolan')
            ]
        ),
        'MPAA Rating': dict(
            defn=(
                'Only actual MPAA rating labels such as "G", "PG", "PG-13", "R", and "NC-17" are named MPAA rating entities.\n'
                'You should always drop starting and trailing "rated" or "MPAA rating of" words from named MPAA rating entity spans.\n'
                # 'Rating interpretations such as "Parental Guidance Suggested" and "suitable for children" are not named MPAA rating entities.'
                'Rating interpretations such as "Parental Guidance Suggested" and "suitable for children" are not relevant named entities.'
            ),
            demos=[
                dict(sentence='Can you recommend a family-friendly movie rated PG with the song Sound of Silence?', entity_span='PG', label=CORRECT),
                dict(sentence='What are some unforgettable G-rated movies from the 90s', entity_span='G-rated', label=WRONG_SPAN, correct_span='G'),
                # dict(sentence='What is the best MPAA rating for a family movie', entity_span='MPAA rating', label=NA),
                dict(sentence='I want to see a trailer for The Wolf of Wall Street, rated R', entity_span='rated R', label=WRONG_SPAN, correct_span='R'),
                dict(sentence='who directed the movie Amadeus and is it appropriate for mature audiences', entity_span='mature audiences', label=WRONG_TYPE, correct_type='other')
                # dict(sentence='please recommend a family-friendly fantasy film', entity_span='family-friendly', label=WRONG_TYPE, correct_type='Genre')
            ]
        ),
        'Plot': dict(
            defn=(
                'Named plot entities must be specific themes, topics, or elements of a movie storyline, such as "space exploration" and "zombie".\n'
                'Plot descriptors such as "captivating plot", "compelling plot", "intense plot" and "thrilling plot" are not relevant named entities. '
                'Narrative descriptors such as "plot", "summary" and "storyline" are also not named entities. '
                'Production insights such as "behind-the-scenes", "making of a film" and "making of movies" are not relevant named entities.'
            ),
            demos=[
                dict(sentence='Can you suggest a good heist movie with a PG-7 rating', entity_span='heist', label=CORRECT),
                # dict(sentence='Have any historical movies with a revenge plot been released in 2022', entity_span='revenge plot', label=WRONG_SPAN, correct_span='revenge')
                # dict(sentence='Who wrote the screenplay for the movie Jurassic Park', entity_span='screenplay', label=NA),
                dict(sentence='Can you give me a brief overview of the plot of "Forrest Gump"?', entity_span='plot', label=NA),
                dict(sentence='Could you recommend a movie with a strong female lead character and a compelling plot', entity_span='compelling plot', label=WRONG_TYPE, correct_type='other'),
                dict(sentence="Are there any exceptional behind-the-scenes documentaries about the making of Jim Carrey's movies?", entity_span='behind-the-scenes', label=WRONG_TYPE, correct_type='other')
            ]
        ),
        'Actor': dict(
            defn=(
                'A named actor entity must be the name of an actor or actress. '
                'Only first names, last names, and full names of actors and actresses are considered named actor entities.\n'
                # 'Fictional actor names based on context should also be named actor entities.\n'
                # 'You should differentiate named actor entities from named director entities based on the context of the query. '
                # 'Names of Artists such as "Beyonce" are not named actor entities.'
                # 'Ambiguous identifiers such as "actor", "lead actor", "lead actress", "male lead", "cast" and "diverse cast" are not named entities.'
            ),
            demos=[
                # dict(sentence='Who played the lead role in the movie "The Godfather"?', entity_span='lead role', label=CORRECT),
                dict(sentence='What character did Tom Hanks play in "Forest Gump"?', entity_span='Tom Hanks', label=CORRECT),
                dict(sentence='What is the name of the actor who played the main character in "The Shawshank Redemption"?', entity_span='actor', label=NA),
                # dict(sentence='Can you provide some background on the art direction in the latest spy film featuring Jane Smith?', entity_span='Jane Smith', label=CORRECT),
                # dict(sentence='Who are the primary actors in the film The Departed?', entity_span='primary actors', label=NA)
            ]
        ),
        'Trailer': dict(
            defn=(
                'A named movie trailer entity is a term that indicates a segment or a clip from a movie, such as "clip", "trailer", "teaser" and "preview".\n'
                'You should always drop starting "captivating" or "fun" descriptive words from named trailer entity spans.\n'
                # 'Named movie trailer entity spans must not have starting or trailing "movie" or "film" words.'
                'You should always drop starting "movie" or "film" words from named trailer entity spans.'
            ),
            demos=[
                dict(sentence='Can you show me the trailer for the movie "The Matrix"?', entity_span='trailer', label=CORRECT),
                # dict(sentence='Can you play the trailer for the latest Mission: Impossible movie?', entity_span='trailer', label=CORRECT),
                dict(sentence='Can you play the movie clip where Yoda gives his famous advice to Luke Skywalker?', entity_span='movie clip', label=WRONG_SPAN, correct_span='clip')
                # dict(sentence='Could you show me a preview for a new release that leaves something to the imagination?', entity_span='preview', label=CORRECT)
            ]
        ),
        'Song': dict(
            defn=(
                'A named song entity must be the name of a song.\n'
                # 'You should always drop starting or trailing "song" words from named song entity spans.\n'
                'Song attributes such as "iconic song," "love song," "famous song," and "original song" are not relevant named entities. '
                'Music genres such as "EDM" and "Jazz" are also not relevant named entities. '
                'Names of Artists such as "Beyonce" are not relevant named entities. '
                'Musical descriptors such as "song", "theme song" and "soundtrack" are also not named entities.'
            ),
            demos=[
                dict(sentence='Is Every Breath You Take featured in any popular movie soundtracks?', entity_span='Every Breath You Take', label=CORRECT),
                # dict(sentence='Is there a specific song featured in the classic movie Road Trip?', entity_span='specific song', label=NA),
                dict(sentence='What is the theme song of the James Bond movie "Skyfall"?', entity_span='theme song', label=NA),
                dict(sentence='Can you play a song from the movie "The Sound of Music"?', entity_span='song', label=NA),
                # dict(sentence='Can you suggest a movie with a famous soundtrack?', entity_span='famous soundtrack', label=NA),
                dict(sentence='Can you suggest a movie with a famous soundtrack?', entity_span='famous soundtrack', label=WRONG_TYPE, correct_type='other')
            ]
        ),
        'Review': dict(
            defn=(
                'A named review entity must be a recognition or a detailed, qualitative assessment of a movie, such as "funniest of all time" and "incredible visual effects". '
                'Named review entities must describe specific aspects of a movie, such as "most awards".\n'
                'Ensure to distinguish named review entities from named viewers\' rating entities. '
                'Short evaluative descriptors such as "good", "highest rated" and "top-rated" are named viewers\' rating entities. '
                'Performance metrics such as "highest-grossing" and "best-rated" are also named viewers\' rating entities.\n'
                'Ambiguous identifiers such as "review" are not named entities.\n'
                'Named genre entities are broad and well-recognized movie categories, such as "documentary" and "comedy". '
                # 'Named director entities must be names of film directors.'
            ),
            demos=[
                # dict(sentence='What movie won the Best Picture award at the Oscars in 2019?', entity_span='Oscars', label=WRONG_TYPE, correct_type='Other'),
                # dict(sentence='What is the best superhero movie of all time?', entity_span='best', label=WRONG_TYPE, correct_type="Viewers' Rating"),
                # dict(sentence='I want to watch a horror movie with a high Rotten Tomatoes rating.', entity_span='high Rotten Tomatoes rating', label=WRONG_TYPE, correct_type="Viewers' Rating")
                dict(sentence='Which movie has the best special effects of all time', entity_span='best special effects', label=CORRECT),
                dict(sentence='Can you recommend a good movie from the 1980s', entity_span='good', label=WRONG_TYPE, correct_type="Viewers' Rating"),
                dict(sentence='Can you tell me if the movie Glory received any awards or nominations?', entity_span='awards or nominations', label=NA)
                # dict(sentence="What's your favorite movie review website?", entity_span='review', label=NA)
            ]
        ),
        'Character': dict(
            defn=(
                'A named character entity must be the name of a character.\n'
                'Character qualifiers such as "main antagonist", "villain character", "strong female lead character" are not relevant named entities. '
                'Ambiguous identifiers such as "character", "main character" and "lead role" are also not named entities.'
            ),
            demos=[
                dict(sentence='who is the actor in the lead role in forrest gump', entity_span='lead role', label=NA),
                dict(sentence='Who is the main character in the Star Wars series', entity_span='main character', label=NA),
                dict(sentence='Tell me about the actress who played Hermione Granger in the Harry Potter series.', entity_span='Hermione Granger', label=CORRECT),
                dict(sentence='Could you recommend a family-friendly movie with a strong female character as the lead?', entity_span='strong female character', label=WRONG_TYPE, correct_type='other')
            ]
        )
    },
}
