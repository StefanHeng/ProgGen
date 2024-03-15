"""
For training a BERT-class model on generated NER data
    with epoch-wise evaluation
"""

import os
import argparse

# for loading **only** relevant `stefutil` functions for training
os.environ['SU_USE_PLT'] = 'F'
os.environ['SU_USE_ML'] = 'T'
os.environ['SU_USE_DL'] = 'T'

from stefutil import get_logger, pl
from src.trainer.run_ner import TrainDummyArgs, run_train
from scripts.utils import add_argument


logger = get_logger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_argument(parser, arg=['dataset_name'])

    # training data
    parser.add_argument(
        '--generated_dataset_dir_name', type=str, required=True,
        help='Generated dataset (optionally with self-correction) directory name containing samples in the BIO format in a jsonl file.'
    )
    parser.add_argument(
        '--few_shot_demo_file', type=str, required=True,
        help='Filename/path to a jsonl file containing the few-shot demo samples (used also for generating NER dataset; written in Step 1).'
    )
    parser.add_argument(
        '--test_file', type=str, required=True,
        help='Filename/path to a jsonl file containing samples in the original dataset test set (written in Step 1).'
    )

    # training hyperparameters
    parser.add_argument(
        '--hf_model_name', type=str, default='microsoft/deberta-v3-base',
        help='The BERT-class model to train as a HuggingFace model name.'
    )
    parser.add_argument(
        '--learning_rate', default=4e-5, type=float,
        help='Learning rate'
    )
    parser.add_argument(
        '--n_epochs', default=16.0, type=float,
        help='Number of epochs to train.'
    )
    parser.add_argument(
        '--train_batch_size', default=24, type=int,
        help='Train batch size on generated NER dataset.'
    )
    parser.add_argument(
        '--eval_batch_size', default=128, type=int,
        help='Eval batch size (on original dataset test set).'
    )
    parser.add_argument(
        '--seed', default=42, type=int,
        help='Random seed for reproducibility.'
    )
    parser.add_argument(
        '--save_trained', type=bool, default=False,
        help='If True, save the model checkpoint after training.'
    )
    parser.add_argument(
        '--demo_weight', type=int, default=5,
        help='The weight of the few-shot demo samples during training.'
    )
    return parser


if __name__ == '__main__':
    def main():
        parser = get_parser()
        args = parser.parse_args()
        logger.info(f'Running command w/ args {pl.i(vars(args), indent=1)}')

        run_train(args=TrainDummyArgs.from_script(args=args))
    main()
