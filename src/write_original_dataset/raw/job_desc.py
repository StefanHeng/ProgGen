"""
For job description dataset from Job Description from paper
    "Benchmark Corpus to Support Entity Recognition in Job Descriptions"
"""

import os
import csv
import json
from os.path import join as os_join
from typing import List, Dict, Any
from dataclasses import dataclass

from stefutil import *
from src.util.ner_example import *
from src.write_original_dataset.raw.util import *


logger = get_logger(__name__)


URL_BASE = 'https://github.com/acp19tag/skill-extraction-dataset/raw/main/preprocessed_data'
URL = dict(
    train='df_answers.csv',
    test='df_testset.csv'
)
URL = {split: os_join(URL_BASE, fnm) for split, fnm in URL.items()}


@dataclass
class Csv:
    header: List[str] = None
    rows: List[Dict[str, Any]] = None


def load_csv(path, delim=','):
    data = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=delim)
        header = next(reader)
        for row in reader:
            data.append({hd: r for hd, r in zip(header, row)})
    return Csv(header=header, rows=data)


if __name__ == '__main__':
    sic.output_width = 128

    dnm = 'job-desc'
    dataset_path = os_join('original-dataset', dnm)
    os.makedirs(dataset_path, exist_ok=True)

    def download():
        for split in ['train', 'test']:
            url = URL[split]
            fnm = f'{split}.csv'
            dset_path = os_join(dataset_path, '')
            download_raw_file(url=url, output_path=dset_path, output_filename=fnm)
    download()

    def write_to_jsonl(seed: int = 42):
        """
        Write to jsonl format for easy loading

        Split training set into train and dev
        """
        def _sent_rows2sample(rows: List[Dict[str, Any]]) -> NerBioExample:
            toks = [r['word'] for r in rows]
            tags = [r['tag'] for r in rows]
            # by construction, len(toks) == len(tags)
            return NerBioExample.from_tokens_n_tags(tokens=toks, tags=tags)

        def path2samples(path: str) -> List[NerBioExample]:
            data = load_csv(path).rows
            # partition into sentences by searching for indices of sentence_id change
            idxs = [i for i, row in enumerate(data) if row['sentence_id'] != data[i-1]['sentence_id']]
            sentences = [data[i:j] for i, j in zip(idxs, idxs[1:] + [None])]  # group tokens for each sentence
            # sanity check sentence ids increments by 1
            assert [int(s[0]['sentence_id']) for s in sentences] == list(range(1, len(sentences) + 1))
            return [_sent_rows2sample(rows) for rows in sentences]

        dset_path = os_join(dataset_path, '')
        tr_path = os_join(dset_path, 'train.csv')
        ts_path = os_join(dset_path, 'test.csv')
        tr_samples = path2samples(tr_path)
        ts_samples = path2samples(ts_path)
        write_dataset(train=tr_samples, test=ts_samples, output_path=dataset_path, seed=seed)
    # write_to_jsonl()

    def check_entity_type_dist():
        """
        Check counts/distribution of entity types for the test set
        """
        ts_path = os_join(dataset_path, 'test.jsonl')
        samples = [NerBioExample(**json.loads(line)) for line in open(ts_path, 'r')]
        # sic(samples[:10])
        ets = ['Skill', 'Qualification', 'Experience', 'Domain', 'Occupation']
        print(pl.i(entity_type_dist(data=samples, entity_types=ets)))
    check_entity_type_dist()
