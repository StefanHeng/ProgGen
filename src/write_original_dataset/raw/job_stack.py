"""
Processing the JobStack dataset

From their [homepage](https://github.com/kris927b/JobStack/tree/master/data):
> In the JobStack paper we create our own data, which can be acquired by contacting us.
"""


if __name__ == '__main__':
    import os
    from os.path import join as os_join

    from stefutil import pl
    from src.util import pu
    from src.write_original_dataset.raw.util import load_conll_style, write_dataset, entity_type_dist

    dnm = 'job-stack'
    dataset_path = os_join(pu.proj_path, 'original-dataset', dnm)
    os.makedirs(dataset_path, exist_ok=True)

    def convert2jsonl():
        dset_path = os_join(dataset_path, 'raw')
        tr_path, vl_path, ts_path = [os_join(dset_path, fnm) for fnm in ['train.conll.txt', 'dev.conll.txt', 'test.conll.txt']]

        tr_samples = load_conll_style(tr_path)
        vl_samples = load_conll_style(vl_path)
        ts_samples = load_conll_style(ts_path)
        # sic(len(tr_samples), tr_samples[:10], tr_samples[-10:])
        # sic(len(tr_samples), len(vl_samples), len(ts_samples))
        write_dataset(train=tr_samples, dev=vl_samples, test=ts_samples, output_path=dataset_path)
    # convert2jsonl()

    def check_entity_type_dist():
        ets = ['Organization', 'Location', 'Profession', 'Contact', 'Name']
        path = os_join(dataset_path, 'test.jsonl')
        print(pl.i(entity_type_dist(data=path, entity_types=ets)))
    check_entity_type_dist()
