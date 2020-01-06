from config import config as cfg
from utils.utils import load_pickle
import cv2
import os

file_names = load_pickle(os.path.join(cfg.DATA.BASEDIR,'val/val.file.names.pkl'))
save_dir = '/data1/junjiaot/images/karpathy_eval'
if not os.path.exists(save_dir):
        os.mkdir(save_dir) 

for i,name in enumerate(file_names):
    file = os.path.join(cfg.DATA.IMGDIR,name.split('\\')[1],name.split('\\')[2])
    img = cv2.imread(file,3)
    save_file = os.path.join(save_dir,name.split('\\')[2])
    cv2.imwrite(save_file,img)
    #import ipdb;ipdb.set_trace()
print('end')
