"""
For generating and processing NER samples into training dataset
    including saving uncertainty scores for each entity annotation
"""


import argparse
from scripts.utils import *
optimize_imports()

from stefutil import *
from src.util import *
from src.generate import *


logger = get_logger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_argument(parser, arg=[
        'dataset_name', 'n_demo', 'include_negative_sample', 'prompt_seed',
        'chat_model_name', 'chat_max_tokens', 'chat_temperature', 'chat_seed', 'chat_logprobs', 'chat_timeout',
        'lowercase'
    ])

    # about NER sample generation
    parser.add_argument(
        '--n_list', required=True, type=int,
        help='Number of NER samples LLMs are requested to generated in a single prompt/API call.'
    )
    parser.add_argument(
        '--n_call', required=True, type=int,
        help='Number of OpenAI API calls to make.'
    )

    # about sample diversity
    variants = ['simple-prompt', 'diversify-x', 'diversify-y-vanilla', 'diversify-y-latent', 'diversify-x+y']
    parser.add_argument(
        '--diversity_variant', required=True, type=str, choices=variants,
        help=f'The diversify variant to generate NER samples. One of {pl.nc(variants)}.'
    )
    parser.add_argument(
        '--diversify_x_config', type=str,
        help='Path to a config file (see reproduce) '
             'or a json string as config file (attribute name => List of attribute values as `Dict[str, List[str]`) '
             'for Diversify X and Diversify X+Y. '
             'If not given, defaults to config as reported in the paper.'
    )
    parser.add_argument(
        '--diversify_x_sample_prob', type=str,
        help='A json string for sampling probability of each attribute dimension '
             '(attribute dimension => sampling probability as `Dict[str, float]`) '
             'for Diversify X and Diversify X+Y. '
             'If not given, defaults to sampling probability as reported in the paper.'
    )
    parser.add_argument(
        '--diversify_y_config', type=str,
        help='Path to a config file (see reproduce) '
             'or a json string as config file ((latent attribute value =>) entity class => List of entity values '
             'as `Dict[str, List[str]]` or `Dict[str, Dict[str, List[str]]]`)'
             'for Diversify Y variants and Diversify X+Y. '
             'If not given, use the default config for Diversify Y.'
    )
    parser.add_argument(
        '--diversify_y_n_exp_entity', type=float, default=-1,
        help='Expected number of named entities sampled/requested in a single prompt/API call '
             'for Diversify Y variants and Diversify X+Y. '
             'If not given (< 0), defaults to setup as reported in the paper.'
    )

    return parser


SAMPLE_FORMAT = 'natural-pair-v2'


if __name__ == '__main__':
    def main():
        parser = get_parser()
        args = process_args(args=parser.parse_args())
        logger.info(f'Running command w/ args {pl.i(vars(args), indent=1)}')

        dnm, chat_args, variant, n_call = args.dataset_name, args.chat_args, args.diversity_variant, args.n_call

        # generate NER samples
        dc = variant in ['diversify-x', 'diversify-x+y']  # diverse context
        de = False  # diverse entity
        if variant in ['diversify-y-vanilla']:
            de = True
        elif variant in ['diversify-y-latent', 'diversify-x+y']:
            de = 'seeded'

        a2c = None
        if args.diversify_x_config or args.diversify_y_config:
            a2c = diversify.Attribute2Categories.from_json(
                dataset_name=dnm, diverse_context=dc, diverse_entity=de,
                diverse_x_config=args.diversify_x_config, diverse_y_config=args.diversify_y_config,
                diverse_y_latent_attr_dim=DATASET_NAME2TOPIC_DIM[dnm] if de == 'seeded' else None)

        post = dict()  # for custom output file name
        if chat_args['seed'] != 42:
            post['sd'] = chat_args['seed']
        if de:
            if args.diversify_y_n_exp_entity < 0:
                if dnm in ['mit-movie']:
                    args.diversify_y_n_exp_entity = 4.5
                else:
                    assert dnm in ['conll2003-no-misc', 'wiki-gold-no-misc', 'mit-restaurant']  # sanity check
                    args.diversify_y_n_exp_entity = 1.5
            else:
                post['#et'] = args.diversify_y_n_exp_entity
        diversity_args = dict(a2c=a2c, attr2prob=args.diversify_x_sample_prob, expected_n_entity=args.diversify_y_n_exp_entity)
        gen = DataGenerator(
            dataset_name=dnm, sample_format=SAMPLE_FORMAT, diverse_context=dc, diverse_entity=de, diversity_args=diversity_args,
            diversity_instr_at_end=True)

        ppt_args = dict(n_demo=args.n_demo, demo_args=dict(include_none_samples=args.include_negative_sample))
        post = pl.pa(post) if post != dict() else None
        out_dir_nm = dataset_meta(n_list=args.n_list, diverse_context=dc, diverse_entity=de, lowercase=args.lowercase, postfix=post)
        out = gen.write_completions(
            n_prompt=n_call, n_list=args.n_list, output_dir_nm=out_dir_nm, prompt_args=ppt_args, generator=args.prompt_seed, **chat_args)

        # process generated NER samples into dataset
        writer = NerDatasetWriter(dataset_name=dnm, sample_format=SAMPLE_FORMAT, detect_enum=True, allowed_entity_types=True)
        writer(
            completions_dir_name=out.output_dir, output_dir_name=out_dir_nm, expected_samples_per_completion=args.n_list,
            lowercase=args.lowercase, logprobs=args.chat_logprobs
        )
    main()
