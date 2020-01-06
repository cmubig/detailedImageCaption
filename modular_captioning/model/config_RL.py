# -*- coding: utf-8 -*-
# File: config.py

import numpy as np
import os
import pprint
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu

__all__ = ['config', 'finalize_configs']


class AttrDict():

    _freezed = False
    """ Avoid accidental creation of new hierarchies. """

    def __getattr__(self, name):
        if self._freezed:
            raise AttributeError(name)
        ret = AttrDict()
        setattr(self, name, ret)
        return ret

    def __setattr__(self, name, value):
        if self._freezed and name not in self.__dict__:
            raise AttributeError("Cannot create new attribute!")
        super().__setattr__(name, value)

    def __str__(self):
        return pprint.pformat(self.to_dict(), indent=1)

    __repr__ = __str__

    def to_dict(self):
        """Convert to a nested dict. """
        return {k: v.to_dict() if isinstance(v, AttrDict) else v
                for k, v in self.__dict__.items() if not k.startswith('_')}

    def update_args(self, args):
        """Update from command line args. """
        for cfg in args:
            keys, v = cfg.split('=', maxsplit=1)
            keylist = keys.split('.')

            dic = self
            for i, k in enumerate(keylist[:-1]):
                assert k in dir(dic), "Unknown config key: {}".format(keys)
                dic = getattr(dic, k)
            key = keylist[-1]

            oldv = getattr(dic, key)
            if not isinstance(oldv, str):
                v = eval(v)
            setattr(dic, key, v)

    def freeze(self):
        self._freezed = True
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.freeze()

    # avoid silent bugs
    def __eq__(self, _):
        raise NotImplementedError()

    def __ne__(self, _):
        raise NotImplementedError()


config = AttrDict()
_C = config     # short alias to avoid coding

# mode flags ---------------------


# dataset -----------------------
_C.DATA.BASEDIR = '/data1/junjiaot/data'#'/path/to/your/COCO/DIR'
_C.DATA.IMGDIR = '/data1/junjiaot/images/resized'
_C.DATA.FEATUREDIR = '/home/junjiaot/data_local'
_C.DATA.DATA = 'frcnn_features36'
# TRAINING ----------------------
_C.LR = 5e-5
_C.DECAY_EPOCH = 70
_C.MIL_STOP_EPOCH = 5
_C.DECAY_STEP = 70
_C.DECAY_RATE = 0.5
_C.MAX_EPOCH = 70
_C.EVAL_PERIOD = 200
_C.PRINT_EVERY = 50
_C.BATCH_SIZE = 128
_C.START_EPOCH = 0
_C.EVAL = False
_C.LOAD =  '/data1/junjiaot/data/checkpoint_v17_max/model_max-281257'#None#'/data1/junjiaot/data/checkpoint_v10_3/checkpoint'
_C.CHECKPOINT_DIR = 'ADSFADSFA'#'/data1/junjiaot/data/f0eckpoint_v16'
_C.CHECKPOINT_DIR_MAX = '/data1/junjiaot/data/checkpoint_v17_RL_spice_1'
_C.SCORE_DIR = './score_v17_RL_spice_1'
_C.EVAL_CAP = 'eval_v17_R_spice_1'
#MODEL --------------------
_C.VOCAB_DIM = 300
_C.NUM_REGION = 36
_C.GLOBAL_FEATURE_DIM = 2048
_C.REGION_FEATURE_DIM = 2053
_C.PRE_EMB = True

def finalize_configs(is_training):
    """
    Run some sanity checks, and populate some configs from others
    """ 
    pass
