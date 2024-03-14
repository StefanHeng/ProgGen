import os

__all__ = ['BASE_PATH', 'PROJ_DIR', 'PKG_NM', 'DSET_DIR', 'MODEL_DIR']

paths = __file__.split(os.sep)
paths = paths[:paths.index('util')]

# Absolute system path for root directory;
#   e.g.: '/Users/stefanhg/Documents/Georgia Tech/Research/LLM for IE Data Gen'
BASE_PATH = os.sep.join(paths[:-2])  # System data path
# Repo root folder name with package name; e.g.: 'LLM-Data-Gen-IE'
PROJ_DIR = paths[-2]
PKG_NM = paths[-1]  # Package/Module name, e.g. `src`

MODEL_DIR = 'models'  # Save models
DSET_DIR = 'dataset'


if __name__ == '__main__':
    from stefutil import sic
    sic(BASE_PATH, type(BASE_PATH), PROJ_DIR, PKG_NM)
