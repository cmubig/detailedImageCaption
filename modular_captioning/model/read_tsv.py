import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import json
from config import config as cfg
from utils.utils import *
import progressbar
import os
csv.field_size_limit(sys.maxsize)
   
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infile = '/data1/junjiaot/data/frcnn_features36/features/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv'
with open(os.path.join(cfg.DATA.BASEDIR,'annotations/captions_train2014.json')) as f:
    train_annotations = json.load(f)
with open(os.path.join(cfg.DATA.BASEDIR,'annotations/captions_val2014.json')) as f:
    val_annotations = json.load(f)

id_to_filename = {image['id']: image['file_name'] for image in train_annotations['images']} #82783
id_to_filename.update({image['id']: image['file_name'] for image in val_annotations['images']})#123287
save_basedir = '/home/junjiaot/data_local/frcnn_features36/'

trian_name_order = os.listdir(os.path.join(cfg.DATA.IMGDIR,'train2014'))
val_name_order = os.listdir(os.path.join(cfg.DATA.IMGDIR,'val2014'))

train_box = np.zeros((82783,36,4))# N,10,4
val_box = np.zeros((40504,36,4))# N,10,4
cnt = 0
if __name__ == '__main__':
    # Verify we can read a tsv
    with progressbar.ProgressBar(max_value=len(id_to_filename)) as bar:
        with open(infile) as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
            for item in reader:
                item['image_id'] = int(item['image_id'])
                item['image_h'] = int(item['image_h'])
                item['image_w'] = int(item['image_w'])   
                item['num_boxes'] = int(item['num_boxes'])
                for field in ['boxes', 'features']:
                    item[field] = np.frombuffer(base64.b64decode(item[field]), 
                          dtype=np.float32).reshape((item['num_boxes'],-1))
                region_feature = item['features']   
                s_b = (item['boxes'][:,2,np.newaxis]-item['boxes'][:,0,np.newaxis])*(item['boxes'][:,3,np.newaxis]-item['boxes'][:,1,np.newaxis])
                s_i = item['image_h']*item['image_w']
                spatial = np.concatenate((item['boxes'][:,0,np.newaxis]/item['image_w'],item['boxes'][:,1,np.newaxis]/item['image_h'],item['boxes'][:,2,np.newaxis]/item['image_w'],item['boxes'][:,3,np.newaxis]/item['image_h'],s_b/s_i),axis=1)
                region_feature = np.concatenate((region_feature,spatial),axis=1)
                #import ipdb;ipdb.set_trace()
                global_feature = np.sum(item['features'],axis=0,keepdims=True)/36
                image_name = id_to_filename[item['image_id']]
                split = image_name.split('_')[1]
                
                if split == 'train2014':
                    index = trian_name_order.index(image_name)
                    train_box[index,:,:] = item['boxes']
                else:
                    index = val_name_order.index(image_name)
                    val_box[index,:,:] = item['boxes']
                
                file_name = os.path.join(save_basedir,split,image_name.split('.')[0]+'.pkl')
                features = [global_feature,region_feature]
                pickle.dump(features,open(file_name,'wb'))
                #import ipdb;ipdb.set_trace()
                bar.update(cnt)
                cnt = cnt + 1
save_pickle(val_box,os.path.join(cfg.DATA.BASEDIR,'val/val2014_box36.pkl'))
save_pickle(train_box,os.path.join(cfg.DATA.BASEDIR,'train/train2014_box36.pkl'))

