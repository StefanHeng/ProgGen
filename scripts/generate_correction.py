"""
For generating and processing LLM Entity Annotation Self-Corrections into
    training dataset with annotations overridden
"""


import argparse

from scripts.utils import *
optimize_imports()

from stefutil import *
from src.util import *
from src.generate.step_wise import *


logger = get_logger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_argument(parser, arg=[
        'dataset_name', 'prompt_seed',
        'chat_model_name', 'chat_max_tokens', 'chat_temperature', 'chat_seed', 'chat_logprobs', 'chat_timeout',
        'lowercase'
    ])

    # about LLM Correction generation
    parser.add_argument(
        '--n_correct', type=int, default=3,
        help='Number of entity annotations LLMs are requested to correct in a single prompt/API call.'
    )
    parser.add_argument(
        '--logprob_thresh', type=float, default=-2e-2,
        help='Threshold of logprob as uncertainty ranking score for selecting entity annotations to correct.'
    )
    parser.add_argument(
        '--top_n', type=float, default=0.2,
        help='Threshold on total number of top-uncertain entity annotations to correct as an integer count or a float ratio.'
    )
    parser.add_argument(
        '--correction_config', type=str, required=True,
        help='Path to a config file containing entity annotation instruction and correction demos (see reproduce) '
             '(entity class => Dict of relevant info as `Dict[str, Dict[str, Any]`) for entity annotation correction.'
    )

    # about correction dataset input & output
    parser.add_argument(
        '--generated_dataset_dir_name', type=str, required=True,
        help='Directory name for the generated NER dataset to self-correct.'
    )
    parser.add_argument(
        '--output_postfix', type=str,  # since the default name for different generated datasets is the same
        help='Custom output directory name postfix for the generated/corrected NER dataset.'
    )

    return parser


if __name__ == '__main__':
    def main():
        parser = get_parser()
        args = process_args(args=parser.parse_args())
        logger.info(f'Running command w/ args {pl.i(vars(args), indent=1)}')

        dnm, chat_args, n_crt = args.dataset_name, args.chat_args, args.n_correct
        thresh, tn = args.logprob_thresh, args.top_n

        # generate self-corrections
        gen = CorrectionGenerator(dataset_name=dnm, sample_format='natural-pair-v2', highlight_span='braces')

        post = dict()  # for custom output file name
        if thresh != -2e-2:
            post['lp'] = f'{thresh:+.0e}'
        if tn != 0.2:
            post['tp'] = tn
        if chat_args['temperature'] != 0:
            post['t'] = chat_args['temperature']
        post = pl.pa(post) if post != dict() else None
        if args.output_postfix:
            post = f'{args.output_postfix}{post}' if post else args.output_postfix
        out_dir_nm = dataset_meta(n_correct=n_crt, postfix=post)

        select_annot_args = dict(logprob_thresh=thresh, top_n=tn)
        out = gen.write_completions(
            triples_dir_name=args.generated_dataset_dir_name, correction_config=args.correction_config,
            n_correct=n_crt, uncertain_triple_args=select_annot_args,
            shuffle_samples=args.prompt_seed, generator=args.prompt_seed, output_dir_nm=out_dir_nm, **chat_args)

        # process self-corrections and override annotations into a new processed dataset
        gen.process_completions(
            triples_dir_name=args.generated_dataset_dir_name, correction_config=args.correction_config,
            uncertain_triple_args=select_annot_args, shuffle_samples=args.prompt_seed, output_dir_name=out_dir_nm,
            completions_dir_name=out.output_dir, expected_samples_per_completion=n_crt, lowercase=args.lowercase)
    main()
