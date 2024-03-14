import argparse
from os.path import join as os_join
from dataclasses import asdict

from torch.nn import CrossEntropyLoss
from transformers import AutoModelForTokenClassification, AutoTokenizer

from stefutil import *
from src.util import *
from src.trainer.utils_ner import load_and_cache_examples, get_ner_tags, get_tag_to_id, EvalDummyArgs
from src.trainer.eval import evaluate


logger = get_logger(__name__)


def check_annotation_accuracy(
        dataset_name: str = 'conll2003',
        dataset_sub_dir: str = None,
        model_dir_name: str = None,
        generated_dataset_dir_name: str = None
):
    """
    Check ChatGPT annotation accuracy using a supervised NER model
        Intended for Conll-2003 since supervised model achieves 92 f1
    """
    sic(dataset_name, dataset_sub_dir, model_dir_name, generated_dataset_dir_name)
    if dataset_name not in ['conll2003', 'conll2003-no-misc', 'mit-movie']:
        raise NotImplementedError

    dset_path = dataset_name2data_dir(dataset_name=dataset_name, sub_dir=dataset_sub_dir, input_dir=generated_dataset_dir_name).path
    out = dataset_name2model_dir(dataset_name=dataset_name, model_dir=model_dir_name)
    model_base_path, model_path = out.base_path, out.path

    date = now(for_path=True, fmt='short-date')
    d = dict(dnm=dataset_name, dset_nm=generated_dataset_dir_name)
    out_dir_nm = f'{date}_Check-Annot-Qual_{pl.pa(d)}'
    output_path = os_join(model_base_path, out_dir_nm)
    add_file_handler(logger=logger, file_path=os_join(output_path, 'check-annot-qual.log'))

    ets = sconfig(f'datasets.{dataset_name}.readable-entity-types')
    d_log = {
        'dataset-name': dataset_name, 'entity-types': ets,
        'generated-dataset-dir-name': generated_dataset_dir_name, 'dataset-path': dset_path,
        'model-dir-name': model_dir_name, 'model-path': model_base_path, 'output-path': output_path
    }
    logger.info(f'Checking annotation accuracy w/ {pl.fmt(d_log)}...')

    model_path = os_join(model_base_path, model_dir_name, 'trained')
    # use default ones, these are used for training the supervised model
    labels = get_ner_tags(dataset_name=dataset_name, verbose=True)
    tag2id = get_tag_to_id(dataset_name=dataset_name, verbose=True)
    d_log = {
        'dataset-name': dataset_name, 'generated-dataset-dir-name': generated_dataset_dir_name,
        'model-dir-name': model_dir_name,
        'labels': labels, 'tag2id': tag2id, '#label': len(labels), 'output-path': output_path
    }
    logger.info(f'Checking annotation accuracy w/ {pl.fmt(d_log)}...')

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True,)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    args = EvalDummyArgs(
        dataset_name=dataset_name, data_dir=dset_path, train_file='bio-train.jsonl', output_dir=output_path, cache_datasets=False)
    logger.info(f'Using dummy arguments for eval: {pl.fmt(asdict(args))}')
    model.to(args.device)
    pad_id = CrossEntropyLoss().ignore_index
    load_dset_args = dict(args=args, tokenizer=tokenizer, labels=labels, pad_token_label_id=pad_id, mode='train')
    out = load_and_cache_examples(**load_dset_args, tag2id=tag2id, return_texts=True)

    # for the sake of eval, use default tag2id instead of the one from generated dataset,
    #   for we are running inference on the supervised model
    args.data_dir = None
    res, _, _ = evaluate(
        args=args, model=model, labels=labels, pad_token_label_id=pad_id, eval_dataset=out.tensor_dataset,
        split='gen-data', texts=out.texts, swap_pred_true=True, logger=logger, entity_types=ets
    )
    res.write_reports()
    for sort in [False, True]:
        fnm = 'model-predictions'
        if sort:
            fnm += '-sorted'
        res.write_predictions(pred_prefix='LLM', true_prefix='BERT', sort_by_loss=sort, filename=fnm)
    d_log = {'micro avg': res.to_f1_log(digit=1, as_percentage=True), 'entity-wise': res.entity_wise_f1()}
    logger.info(f'LLM annotation quality: {pl.fmt(d_log)}')


if __name__ == '__main__':
    # for CoNLL-03
    supervised_model_dir_name = '23-10-13_23-05-05_{md=bert-base-cased,gpt-md=3.5T,n_ep=16,lr=2e-5}_ori-train-data-all'

    def check_annot_acc():
        dset_dir_nm = '23-10-12_Processed-NER-Data_{fmt=n-p2,#l=3,dc={pst={cat=30}}}_n=2K'  # best performing set-up so far, 70.4 f1
        check_annotation_accuracy(model_dir_name=supervised_model_dir_name, generated_dataset_dir_name=dset_dir_nm)
    # check_annot_acc()

    def command_prompt():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_name', type=str, default='conll2003')
        parser.add_argument('--dataset_sub_directory', type=str, default=None)
        parser.add_argument("--model_dir_name", default=supervised_model_dir_name, type=str)
        parser.add_argument('--dataset_dir_name', type=str, required=True)

        args = parser.parse_args()
        check_annotation_accuracy(
            dataset_name=args.dataset_name, dataset_sub_dir=args.dataset_sub_directory,
            model_dir_name=args.model_dir_name, generated_dataset_dir_name=args.dataset_dir_name)
    command_prompt()
