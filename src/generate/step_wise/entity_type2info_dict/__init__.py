import os
import re
import json
from copy import deepcopy
from os.path import join as os_join
from typing import Dict, List, Any

from stefutil import *
from src.util import *
import src.generate.step_wise.util_entity_correct as uec
from src.generate.step_wise.entity_type2info_dict.util import *

from .conll2003_no_misc import entity_type2info_dict as conll2003_no_misc
from .wiki_gold_no_misc import entity_type2info_dict as wiki_gold_no_misc
from .mit_movie import entity_type2info_dict as mit_movie
from .mit_restaurant import entity_type2info_dict as mit_restaurant


__all__ = ['Type2EntityCorrectionInfo']


dataset_name2entity_type_info = {  # dedicated demos for each generated dataset
    'conll2003-no-misc': conll2003_no_misc,
    'wiki-gold-no-misc': wiki_gold_no_misc,
    'mit-movie': mit_movie,
    'mit-restaurant': mit_restaurant,
}


class Type2EntityCorrectionInfo:
    def __init__(self, dataset_name: str = None, triples_dir_name: str = None, d: Dict[str, Any] = None):
        self.dataset_name = dataset_name
        self.triples_dir_name = triples_dir_name

        d = d or dataset_name2entity_type_info[dataset_name]
        if triples_dir_name is not None:
            d = d[triples_dir_name]
        self.d = deepcopy(d)

        for et, d_ in self.d.items():
            defn = d_['defn']
            # make sure no accidentally missing whitespace in inner sentences
            idxs_period = [m.start() for m in re.finditer(r'\.[^$"]', defn)]
            assert all(defn[i + 1] in [' ', '\n'] for i in idxs_period)
            d_['defn'] = defn.strip()  # make-sure no accidentally added trailing whitespace'

            if 'name' not in d_:
                d_['name'] = et.lower()
            if 'name_full' not in d_:
                d_['name_full'] = f'named {et.lower()} entity'

            demos = d_.get('demos', [])
            # the demos can be dict objects, convert them to correction sample objects
            if any(not isinstance(demo, uec.EntityCorrectionSample) for demo in demos):
                assert all(isinstance(demo, dict) for demo in demos)  # sanity check
                d_['demos'] = [uec.EntityCorrectionSample.from_dict(demo) for demo in demos]

    def __call__(self, entity_type: str = None) -> Dict[str, Any]:
        return self.d[entity_type]

    def to_json(self, save_path: str = None, save_fnm: str = None, meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Write all info to a json file
        """
        ret = deepcopy(self.d)
        for et, d in ret.items():
            demos: List[uec.EntityCorrectionSample] = d.pop('demos')
            d['demos'] = [demo.to_dict() for demo in demos]
        meta = dict(dataset_name=self.dataset_name, source_dataset_dir_name=self.triples_dir_name, **(meta or dict()))
        ret = dict(meta=meta, entity_type2correction_info=ret)

        if save_path or save_fnm:
            save_path: str = save_path or os_join(pu.proj_path, uec.CORRECTION_DIR_NAME)
            os.makedirs(save_path, exist_ok=True)
            save_fnm = f'{save_fnm}.json' or f'{now(for_path=True, fmt="short-full")}_Entity-Correction-Config.json'
            with open(os_join(save_path, save_fnm), 'w') as f:
                json.dump(ret, f, indent=4)
        return ret

    @classmethod
    def from_json(cls, dataset_name: str = None, config: str = None) -> 'Type2EntityCorrectionInfo':
        """
        Read all info from a json file
        """
        assert os.path.exists(config)  # sanity check
        d = json.load(open(config, 'r'))
        d = d['entity_type2correction_info']  # see `to_json` above
        return cls(dataset_name=dataset_name, d=d)
