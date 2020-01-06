import os
import argparse
import cv2
import shutil
import itertools
import tqdm
import numpy as np
import json
import six
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.common import get_tf_version_tuple
import tensorpack.utils.viz as tpviz
from config import finalize_configs, config as cfg
from eval import *
from utils.bleu import *
from model_v16 import Model
from coco_data_v4 import (
    get_train_dataflow, get_eval_dataflow,)



def predict(pred_func, input_file):
    pass

class SaverCallback(Callback):
    def __init__(self,evaluate=False):
        self.eval = evaluate

    def _setup_graph(self):
        self.saver = tf.train.Saver(max_to_keep=5)

    def _eval(self):
        self.saver.save(tf.get_default_session(),os.path.join(cfg.CHECKPOINT_DIR, 'model'), global_step=tf.train.get_global_step(),write_meta_graph=False)
        print('Model saved to ' + os.path.join(cfg.CHECKPOINT_DIR, 'model'))

    def _before_epoch(self):
        if self.eval == False:
            self._eval()

class EvalCallback(Callback):

    _chief_only = False

    def __init__(self,in_names, out_names, word_to_idx,evaluate=False):
        # 
        self._in_names, self._out_names = in_names, out_names
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.max_cider = 0
        self.eval = evaluate

    def _setup_graph(self):
        num_gpu = 1
        self.num_predictor = num_gpu * 2
        self.predictors =  self._build_coco_evaluator(1)         
        self.dataflows = get_eval_dataflow(shard=0, num_shards=1)
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        self.max_saver = tf.train.Saver(max_to_keep=1)

    def _build_coco_evaluator(self, idx):
        graph_func = self.trainer.get_predictor(self._in_names, self._out_names, device=idx)
        return lambda data: predict_batch(data, graph_func)

    def _eval(self):
        all_gen_cap,all_alpha_module = eval_coco(self.dataflows,self.predictors)
        all_decoded = decode_captions(all_gen_cap, self.idx_to_word)
        sample_index = np.random.randint(0,len(all_decoded),size=1)
        for index in sample_index:
            print(all_decoded[index])
            print(all_alpha_module[index,:10,:])
        save_pickle(all_decoded,os.path.join(cfg.DATA.BASEDIR,'val/val.'+cfg.EVAL_CAP))
        print('Calculating scores...')
        scores,spice_scores = evaluate(data_path=cfg.DATA.BASEDIR, split='val', eval_file=cfg.EVAL_CAP,get_scores=True)
        if scores['CIDEr'] > self.max_cider:
                self.max_cider = scores['CIDEr']
                self.max_saver.save(tf.get_default_session(),os.path.join(cfg.CHECKPOINT_DIR_MAX, 'model_max'), global_step=tf.train.get_global_step(),write_meta_graph=False)
                print('Model saved to ' + os.path.join(cfg.CHECKPOINT_DIR_MAX, 'model'))
        g=tf.get_default_graph()
        write_bleu(scores=scores,spice_scores=spice_scores, path=cfg.SCORE_DIR, epoch=self.trainer.epoch_num,lr=self.trainer.sess.run(g.get_tensor_by_name('learning_rate:0')))

    def _trigger_step(self):
        if self.trainer.local_step % cfg.EVAL_PERIOD == 0:
            logger.info("Running evaluation ...")
            self._eval()

    def before_train(self):
        if self.eval == True:
            all_gen_cap,all_alpha_module = eval_coco(self.dataflows,self.predictors)
            all_decoded = decode_captions(all_gen_cap, self.idx_to_word)
            save_pickle(all_decoded,os.path.join(cfg.DATA.BASEDIR,'val/val.'+cfg.EVAL_CAP))
            print('Calculating scores...')
            scores,spice_scores = evaluate(data_path=cfg.DATA.BASEDIR, split='val', eval_file=cfg.EVAL_CAP,get_scores=True)
            import ipdb;ipdb.set_trace()


if __name__ == '__main__':
	#export CUDA_VISIBLE_DEVICES=1
	#python train.py --load /home/junjiaot/tensorpack/examples/FasterRCNN/models/COCO-R101FPN-MaskRCNN-BetterParams.npz 
	# --evaluate /data1/junjiaot/
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS',
    							default='/home/junjiaot/tensorpack/examples/FasterRCNN/models/COCO-R101FPN-MaskRCNN-BetterParams.npz')
    parser.add_argument('--starting_epoch',default = 0)
    parser.add_argument('--mode',default = 'module') # 'baseline' or 'module'
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        nargs='+')

    if get_tf_version_tuple() < (1, 6):
        # https://github.com/tensorflow/tensorflow/issues/14657
        logger.warn("TF<1.6 has a bug which may lead to crash in FasterRCNN if you're unlucky.")

    args = parser.parse_args()
    #if args.config:
    #    cfg.update_args(args.config)

    lr_schedule=[]
    for e in range(cfg.MAX_EPOCH):
        if e < cfg.DECAY_EPOCH:
            lr_schedule.append((e,cfg.LR))
        else:
            lr = cfg.LR * cfg.DECAY_RATE **((e-cfg.DECAY_EPOCH)/(cfg.MAX_EPOCH-cfg.DECAY_EPOCH))
            lr_schedule.append((e, lr))
    #import ipdb;ipdb.set_trace()

    MIL_schedule = []
    for e in range(cfg.MAX_EPOCH):
        if e < cfg.MIL_STOP_EPOCH:
            MIL_schedule.append((e, 1.0))
        else:
            MIL_schedule.append((e, 0.0))

    word_to_idx = load_pickle(os.path.join(cfg.DATA.BASEDIR,'train/word_to_idx.pkl'))
    input_tensor = ['mean_feature','region_feature','box','histogram']
    #input_tensor = ['mean_feature','region_feature']
    #output_tensor = ['sampled_captions','alpha_module','beta_module']
    output_tensor = ['sampled_captions','alpha_module']
    if not os.path.exists(cfg.CHECKPOINT_DIR):
    	os.mkdir(cfg.CHECKPOINT_DIR)

    if not os.path.exists(cfg.CHECKPOINT_DIR_MAX):
        os.mkdir(cfg.CHECKPOINT_DIR_MAX)

    MODEL = Model(word_to_idx,mode = args.mode,M=cfg.VOCAB_DIM,R=cfg.NUM_REGION,D=cfg.GLOBAL_FEATURE_DIM,DR=cfg.REGION_FEATURE_DIM,pre_trained_embedding = cfg.PRE_EMB)
    
    callbacks = [
        #PeriodicCallback(
        #    ModelSaver(max_to_keep=5, keep_checkpoint_every_n_hours=1,checkpoint_dir=cfg.CHECKPOINT_DIR),
        #   every_k_steps=1),
        SaverCallback(evaluate=cfg.EVAL),
        EvalCallback(input_tensor,output_tensor,word_to_idx,evaluate=cfg.EVAL),
        ScheduledHyperParamSetter('learning_rate', lr_schedule),
        ScheduledHyperParamSetter('MIL_schedule', MIL_schedule),
        PeakMemoryTracker(),
        EstimatedTimeLeft(median=True),
        GPUUtilizationTracker()
    ]
    
    session_init=None
    if cfg.LOAD:
        session_init = get_model_loader(cfg.LOAD)
    train_dataflow = get_train_dataflow()
    traincfg = TrainConfig(
        model=MODEL,
        data=QueueInput(train_dataflow),
        callbacks=callbacks,
        starting_epoch=cfg.START_EPOCH,
        max_epoch=cfg.MAX_EPOCH,
        session_init=session_init,
    )
    trainer = SimpleTrainer()
    launch_train_with_config(traincfg, trainer)
