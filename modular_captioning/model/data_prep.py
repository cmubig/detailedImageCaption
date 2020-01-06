import numpy as np 
import pickle
import ipdb
import progressbar
from config import config as cfg
import os
from pycocotools.coco import COCO
from utils.utils import *
import cv2
from utils.model_box import crop_and_resize


#with open(os.path.join(cfg.DATA.BASEDIR,'val/val2014_frcnn.pkl'),'rb') as f:
#    val2014 = pickle.load(f)
#with open(os.path.join(cfg.DATA.BASEDIR,'train/train2014_frcnn.pkl'),'rb') as f:
#    train2014 = pickle.load(f)

ids_to_cats=load_pickle(os.path.join(cfg.DATA.BASEDIR,'train/ids_to_cats.pkl'))
word_to_idx = load_pickle(os.path.join(cfg.DATA.BASEDIR,'train/word_to_idx.pkl'))
val_box = load_pickle(os.path.join(cfg.DATA.BASEDIR,'val/val2014_box36.pkl'))
train_box = load_pickle(os.path.join(cfg.DATA.BASEDIR,'train/train2014_box36.pkl'))
base_dir = '/home/junjiaot/data_local'
partition = ['train2014']

num_box = val_box.shape[1]
val_hist = []
train_hist = []
#import ipdb;ipdb.set_trace()

for part in partition:
    if part == 'train2014':
        box = train_box
    else:
        box = val_box
    files = os.listdir(os.path.join(base_dir,part))
    with progressbar.ProgressBar(max_value=len(files)) as bar:
        for i,file in enumerate(files): 
            file_name = os.path.join(os.path.join(base_dir,part),file)
            img = cv2.imread(file_name, cv2.IMREAD_COLOR)
            hist = []
            for j in range(num_box):
                hist.append([])
                mask = np.zeros(img.shape[:2], np.uint8)
                mask[int(box[i,j,1]):int(box[i,j,3]), int(box[i,j,0]):int(box[i,j,2])] = 255
                masked_img = cv2.bitwise_and(img,img,mask = mask)      
                hist_mask0 = cv2.calcHist([img],[0],mask,[81],[0,256])
                hist_mask1 = cv2.calcHist([img],[1],mask,[81],[0,256])
                hist_mask2 = cv2.calcHist([img],[2],mask,[81],[0,256])
                denum =  np.sum(hist_mask0)
                hist[j].extend(np.concatenate([hist_mask0/denum,hist_mask1/denum,hist_mask2/denum],0))
            if part == 'train2014':
                train_hist.append(np.squeeze(hist,2))
            else:
                val_hist.append(np.squeeze(hist,2))
            bar.update(i)
train_hist = np.array(train_hist)
val_hist = np.array(val_hist)
#import ipdb;ipdb.set_trace()
save_pickle(train_hist,os.path.join(cfg.DATA.BASEDIR,'train/train_hist36.pkl'))
#save_pickle(val_hist,os.path.join(cfg.DATA.BASEDIR,'val/val_hist36.pkl'))

'''
val_cnt = []
train_cnt = []
for ele in val2014:
    val_cnt.append(len(ele))
for ele in train2014:
    train_cnt.append(len(ele))

val_box = []# N,10,4
train_box = [] # N,10,4
val_class =[] # N,5
train_class = [] # N,5

with progressbar.ProgressBar(max_value=len(val_cnt)) as bar:
    for i,ele in enumerate(val2014):
        # sort accroding to area
        area = []
        if len(ele) != 0:
            for j in range(len(ele)):
                xmin,ymin,xmax,ymax = ele[j].box
                area.append((xmax-xmin) * (ymax-ymin)) 
            temp = list(zip(area,ele))
            temp = sorted(temp, key=lambda temp2:temp[0],reverse=True)
            area,ele = zip(*temp)
        #ipdb.set_trace()
        val_box.append([])
        val_class.append([])
        temp = []
        for j in range (10):
            if j>=len(ele):
                val_box[i].append([0.0,0.0,0.0,0.0])
                if len(val_class[i]) < 10:
                        val_class[i].append(0)
            else:
                val_box[i].append(ele[j].box)
                if ele[j].class_id not in temp:
                    temp.append(ele[j].class_id)
                    cat_name = ids_to_cats[ele[j].class_id]
                    for word in cat_name.split(' '):
                        if len(val_class[i]) < 10:
                            val_class[i].append(word_to_idx[word])
                else:
                    if len(val_class[i]) < 10:
                        val_class[i].append(0)      
        bar.update(i)
val_box = np.array(val_box)
val_class = np.array(val_class)
save_pickle(val_box,os.path.join(cfg.DATA.BASEDIR,'val/val2014_box.pkl'))
save_pickle(val_class,os.path.join(cfg.DATA.BASEDIR,'val/val_class_label_80.pkl'))

with progressbar.ProgressBar(max_value=len(train_cnt)) as bar:
    for i,ele in enumerate(train2014):
        # sort accroding to area
        area = []
        if len(ele) != 0:
            for j in range(len(ele)):
                xmin,ymin,xmax,ymax = ele[j].box
                area.append((xmax-xmin) * (ymax-ymin)) 
            temp = list(zip(area,ele))
            temp = sorted(temp, key=lambda temp2:temp[0],reverse=True)
            area,ele = zip(*temp)
        #ipdb.set_trace()
        train_box.append([])
        train_class.append([])
        temp = []
        for j in range (10):
            if j>=len(ele):
                train_box[i].append([0.0,0.0,0.0,0.0])
                if len(train_class[i]) < 10:
                    train_class[i].append(0)
            else:
                train_box[i].append(ele[j].box)
                if ele[j].class_id not in temp:
                    temp.append(ele[j].class_id)
                    cat_name = ids_to_cats[ele[j].class_id]
                    for word in cat_name.split(' '):
                        if len(train_class[i]) < 10:
                            train_class[i].append(word_to_idx[word])
                else:
                    if len(train_class[i]) <10:
                        train_class[i].append(0)    
        bar.update(i)
train_box = np.array(train_box)
train_class = np.array(train_class)
#pickle.dump(train_box,open('train2014_box.pkl','wb'))
save_pickle(train_box,os.path.join(cfg.DATA.BASEDIR,'train/train2014_box.pkl'))
save_pickle(train_class,os.path.join(cfg.DATA.BASEDIR,'train/train_class_label_80.pkl'))
import ipdb;ipdb.set_trace()

# bug.... not sure

annotation_file=os.path.join(cfg.DATA.BASEDIR, 'instances_{}.json'.format(val2014))
self.coco = COCO(annotation_file)
cat_ids = self.coco.getCatIds()
cat_names = [c['name'] for c in self.coco.loadCats(cat_ids)]
ids_to_cats = dict(zip(cat_ids,cat_names))
save_pickle(ids_to_cats,os.path.join(cfg.DATA.BASEDIR,'train/ids_to_cats.pkl'))
#import ipdb;ipdb.set_trace()
'''
