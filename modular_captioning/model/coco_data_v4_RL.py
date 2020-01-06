# -*- coding: utf-8 -*-
# File: data.py

import cv2
import numpy as np
import copy
import json
import pickle
import os
import tensorflow as tf
from utils.utils import load_pickle
from config_RL import config as cfg
from tensorpack.utils.argtools import memoized, log_once
from tensorpack.dataflow import (
    imgaug, TestDataSpeed,MapData,
    MultiProcessMapDataZMQ, MultiThreadMapData,BatchData,PrefetchDataZMQ,
    MapDataComponent, DataFromList)
from tensorpack.utils import logger

def get_train_dataflow():
 # data from list 
 # each ele should have: name
    file_name_short = load_pickle(os.path.join(cfg.DATA.BASEDIR,'train/train.file.names.pkl')) #(113287,)
    #names  =[]
    save_dir = []
    names_short = []
    for name in file_name_short:
        save_dir.append(name.split('\\')[1])
        names_short.append(name.split('\\')[2])

    #captions = load_pickle(os.path.join(cfg.DATA.BASEDIR,'train/train.captions.pkl')) #560432 captions
    #ref_cap  = load_pickle(os.path.join(cfg.DATA.BASEDIR,'train/train.references.pkl')) #113287 dict refs


    boxes = load_pickle(os.path.join(cfg.DATA.BASEDIR,'train/train2014_box36.pkl')) #(82783, 10, 4)
    boxes = np.concatenate((boxes,load_pickle(os.path.join(cfg.DATA.BASEDIR,'val/val2014_box36.pkl'))))#(123287, 10, 4)
    
    histogram = load_pickle(os.path.join(cfg.DATA.BASEDIR,'train/train_hist36.pkl'))
    histogram = np.concatenate((histogram,load_pickle(os.path.join(cfg.DATA.BASEDIR,'val/val_hist36.pkl'))))

    image_name = os.listdir(os.path.join(cfg.DATA.IMGDIR,'train2014')) #(82783)
    image_name.extend(os.listdir(os.path.join(cfg.DATA.IMGDIR,'val2014')))#(123287,)

    named_boxes = dict(zip(image_name,boxes)) #(123287,)
    named_hist = dict(zip(image_name,histogram)) #(123287,)

    num = len(names_short) #113287
    #import ipdb;ipdb.set_trace()
    data = [[name,save_dir[i],i,named_boxes[name],named_hist[name]] for i,name in enumerate(names_short)] #560432
    #data = [[name,save_dir[i],ref_cap[i],named_boxes[name],named_hist[name]] for i,name in enumerate(names_short)] #560432
    ds = DataFromList(data, shuffle=True)
    def preprocess(data):
        file,save_dir,ref_key,box,hist = data
        feature_dir = os.path.join(cfg.DATA.FEATUREDIR,cfg.DATA.DATA, save_dir)
        feature = load_pickle(os.path.join(feature_dir,file.split('.')[0]+'.pkl'),p=False) # [mean(1,2048),region(10,2048)]
        ret = [feature[0],feature[1],box,hist,ref_key]

        #import ipdb;ipdb.set_trace()
        return ret
    ds = MultiProcessMapDataZMQ(ds, 2, preprocess)
    #ds = MapData(ds, preprocess)
    #ds = MultiThreadMapData(ds,5,preprocess)
    #import ipdb;ipdb.set_trace()
    ds = BatchData(ds,cfg.BATCH_SIZE, remainder=True, use_list=False)
    #ds = PrefetchDataZMQ(ds, 10)
    return ds


def get_eval_dataflow(shard=0, num_shards=1):
    """
    Args:
        shard, num_shards: to get subset of evaluation data
    """
    file_names = load_pickle(os.path.join(cfg.DATA.BASEDIR,'val/val.file.names.pkl'))#5000 images #karparthy split
    #import ipdb;ipdb.set_trace()
    names  =[]
    save_dir = []
    for i,name in enumerate(file_names):
        names.append(name.split('\\')[2])
        save_dir.append(name.split('\\')[1])

    #captions = load_pickle(os.path.join(cfg.DATA.BASEDIR,'val/val.captions.pkl')) #560432 captions

    boxes = load_pickle(os.path.join(cfg.DATA.BASEDIR,'train/train2014_box36.pkl')) #(82783, 10, 4)
    boxes = np.concatenate((boxes,load_pickle(os.path.join(cfg.DATA.BASEDIR,'val/val2014_box36.pkl'))))#(123287, 10, 4)
    #histogram = np.zeros((123287,10,243),dtype=np.float32) # (123287,10,243)
    histogram = load_pickle(os.path.join(cfg.DATA.BASEDIR,'train/train_hist36.pkl'))
    histogram = np.concatenate((histogram,load_pickle(os.path.join(cfg.DATA.BASEDIR,'val/val_hist36.pkl'))))
    #import ipdb;ipdb.set_trace()
    #class_label = load_pickle(os.path.join(cfg.DATA.BASEDIR,'train/train_class_label_80.pkl')) #(82783,)
    #class_label=np.concatenate((class_label,load_pickle(os.path.join(cfg.DATA.BASEDIR,'val/val_class_label_80.pkl')))) #(123287,)

    image_name = os.listdir(os.path.join(cfg.DATA.IMGDIR,'train2014')) #(82783)
    image_name.extend(os.listdir(os.path.join(cfg.DATA.IMGDIR,'val2014')))#(123287,)

    named_boxes = dict(zip(image_name,boxes)) #(123287,)
    named_hist = dict(zip(image_name,histogram)) #(123287,)
    #named_label = dict(zip(image_name,class_label)) #(123287,)
    num = len(file_names)

    data_per_shard = num // num_shards
    data_range = (shard * data_per_shard, (shard + 1) * data_per_shard if shard + 1 < num_shards else num)
    # data: [name,caption,box,hist,label]
    data = [[name,save_dir[i],named_boxes[name],named_hist[name]] for i,name in enumerate(names)] 
    #import ipdb;ipdb.set_trace()
    ds = DataFromList(data[data_range[0]: data_range[1]], shuffle=False)

    def preprocess(data):
        file,save_dir,box,histogram = data
        feature_dir = os.path.join(cfg.DATA.FEATUREDIR,cfg.DATA.DATA, save_dir)
        feature = load_pickle(os.path.join(feature_dir,file.split('.')[0]+'.pkl'),p=False) # [mean(1,2048),region(10,2048)]
        #ret = {'mean_feature':feature[0],'region_feature':feature[1],'box':box,'histogram':histogram,'class_labels':label}
        ret = [feature[0],feature[1],box,histogram]
        #import ipdb;ipdb.set_trace()
        return ret
    ds = MapData(ds, preprocess)
    #ds1 = MultiThreadMapData(
    #      ds, nr_thread=25,
    #      map_func=preprocess, buffer_size=1000)
    ds = BatchData(ds,cfg.BATCH_SIZE, remainder=True, use_list=False)
    return ds


if __name__ == '__main__':
    import os
    from tensorpack.dataflow import PrintData
    ds = get_eval_dataflow()
    ds = PrintData(ds, 1)
    TestDataSpeed(ds,1).start()
    ds.reset_state()
    for k in ds:
        pass
