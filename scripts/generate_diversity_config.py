"""
For generating and processing
    1> attribute values for Diversify X, and
    2> named entity pools for Diversify Y
"""

import argparse
from scripts.utils import optimize_imports, add_argument, process_args, DATASET_NAME2TOPIC_DIM
optimize_imports()

from stefutil import get_logger, pl
from src.generate.diversify import CategoryGenerator, EntityGenerator, Attribute2Categories, ENTITY_KEY_SEEDED


logger = get_logger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_argument(parser, arg=[
        'dataset_name', 'prompt_seed',
        'chat_model_name', 'chat_max_tokens', 'chat_temperature', 'chat_seed', 'chat_logprobs', 'chat_timeout'
    ])

    # for sample diversity
    parser.add_argument(
        '--diversity_variant', required=True, type=str, choices=['diversify-x', 'diversify-y-vanilla', 'diversify-y-latent'],
        help='The diversify variant to generate requirement configurations. One of [diversify-x, diversify-y-vanilla, diversify-y-latent].'
    )
    parser.add_argument(
        '--n_call', required=True, type=int,
        help='Number of OpenAI API calls to make for each group. '
             'For Diversify X, each group is an attribute dimension; '
             'For Diversify Y, each group is an (entity class) for the vanilla variant and '
             'an (entity class, topic attribute value) tuple for the latent variant. '
    )

    # for the latent variant of Diversify Y
    parser.add_argument(
        '--diversify_y_latent_attribute', type=str,
        help='The latent topic attribute values '
             'accessed from the path of a Diversify X config file (see reproduce) '
             'or a json string (topic attribute name => List of attribute values as `Dict[str, List[str]`)'
             'for Diversify Y (latent). '
             'If not given, defaults to the attribute values as reported in the paper.'
    )
    return parser


if __name__ == '__main__':
    def main():
        parser = get_parser()
        args = process_args(args=parser.parse_args())
        logger.info(f'Running command w/ args {pl.i(vars(args), indent=1)}')

        dnm, chat_args, variant, n_call = args.dataset_name, args.chat_args, args.diversity_variant, args.n_call

        if variant == 'diversify-x':
            gen = CategoryGenerator(dataset_name=dnm)
        else:
            assert 'diversify-y' in variant  # sanity check
            seeded, a2c = variant == 'diversify-y-latent', None
            if seeded and args.diversify_y_latent_attribute:
                lat_dim = DATASET_NAME2TOPIC_DIM[dnm]
                a2c = Attribute2Categories.from_json(
                    dataset_name=dnm, diverse_context=False, diverse_entity='seeded',
                    diverse_x_config=args.diversify_y_latent_attribute,
                    **({ENTITY_KEY_SEEDED: dict(seed_category=lat_dim)})  # follow internal `Attribute2Categories` API
                )
            gen = EntityGenerator(dataset_name=dnm, seeded=seeded, a2c=a2c)
        out = gen.write_completions(n_prompt=n_call, **chat_args)
        gen.process_completions(completions_dir_name=out.output_dir)
    main()
