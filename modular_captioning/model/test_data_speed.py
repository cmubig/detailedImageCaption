import cv2
import numpy as np
import copy
import json
import pickle
import os
import tensorflow as tf
from utils.utils import load_pickle
from config import config as cfg
from tensorpack.utils.argtools import memoized, log_once
from tensorpack.dataflow import (
    imgaug, TestDataSpeed,MapData,
    MultiProcessMapDataZMQ, MultiThreadMapData,BatchData,PrefetchDataZMQ,
    MapDataComponent, DataFromList)
from tensorpack.utils import logger
from config import config as cfg

def get_train_dataflow():
 # data from list 
 # each ele should have: name
    annotations = load_pickle(os.path.join(cfg.DATA.BASEDIR,'train/train.annotations.pkl')) #karparthy split
    file_names = annotations['file_name'] #560432 names
    names  =[]
    save_dir = []

    for i,name in enumerate(file_names):
        names.append(name.split('\\')[2])
        save_dir.append(name.split('\\')[1])

    num = len(file_names)
    data = [[name,save_dir[i]] for i,name in enumerate(names)] #560432
    ds = DataFromList(data, shuffle=True)
    def preprocess(data):
        file,save_dir = data
        feature_dir = os.path.join(cfg.DATA.FEATUREDIR,cfg.DATA.DATA, save_dir)
        feature = load_pickle(os.path.join(feature_dir,file.split('.')[0]+'.pkl'),p=False) # [mean(1,2048),region(10,2048)]
        ret = [feature[0],feature[1]]
        return ret
    #ds = MultiProcessMapDataZMQ(ds, 10, preprocess)
    ds = PrefetchDataZMQ(MultiThreadMapData(ds,10,preprocess),1)
    #ds = MapData(ds, preprocess)
    ds = BatchData(ds,cfg.BATCH_SIZE, remainder=True, use_list=False)
    #ds = PrefetchDataZMQ(ds, 5)
    return ds


if __name__ == '__main__':
    import os
    ds = get_train_dataflow()
    TestDataSpeed(ds,100).start()
    ds.reset_state()
