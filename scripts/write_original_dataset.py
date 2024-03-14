"""
For writing original dataset (BIO format) into a file, used for downstream model training
    1> the 1-shot demo samples (selected from the train split)
    2> the entire test split
"""


import os
import argparse
from os.path import join as os_join

from scripts.utils import *
optimize_imports()
from stefutil import *
from src import write_original_dataset
from src.util import pu
from src.util.ner_example import DatasetLoader


logger = get_logger(__name__)


def get_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='type', required=True)
    demo_parser, test_parser = subparsers.add_parser('demo'), subparsers.add_parser('test')
    add_argument(parser=demo_parser, arg=['dataset_name', 'n_demo', 'include_negative_sample'])
    add_argument(parser=test_parser, arg='dataset_name')
    return parser


if __name__ == '__main__':
    # demo_parser.add_argument("--num_epochs", type=int, required=False, default=8)
    # demo_parser.add_argument("--learning_rate", type=float, required=False, default=2e-5)
    # demo_parser.add_argument("--weight_decay", type=float, required=False, default=0.01)

    def main():
        parser = get_parser()
        args = process_args(args=parser.parse_args())

        dnm = args.dataset_name
        # write original dataset if not available
        if dnm in DatasetLoader.from_local_bio_file:
            dnm_ = dnm
            if dnm.endswith('-no-misc'):  # drop the postfix for original dataset write
                dnm_ = dnm[:-len('-no-misc')]
            path = os_join(pu.proj_path, 'original-dataset', dnm_)
            if not os.path.exists(path):
                # write the original bio file: for all 3 datasets except conll2003 cos from HuggingFace
                write_func = getattr(write_original_dataset.raw, dnm_.replace('-', '_'))
                write_func()

        tp = args.type
        if tp == 'demo':
            n, neg = args.n_demo, args.include_negative_sample
            write_original_dataset.write_train(dataset_name=dnm, few='n-shot', shuffle=True, n=n, include_none_samples=neg)
        else:
            assert tp == 'test'
            write_original_dataset.write_test(dataset_name=dnm)
    main()
