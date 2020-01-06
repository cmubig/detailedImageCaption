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
from config_RL import finalize_configs, config as cfg
from utils.bleu import *
from utils.utils import *
from model_v16_RL import Model
from coco_data_v4_RL import (
    get_train_dataflow, get_eval_dataflow,)
sys.path.append('../coco-caption')
from cider_custom.cider import Cider
from spice_custom.spice import Spice


class CaptioningSolver(object):
    def __init__(self):
        self.model = Model(M=cfg.VOCAB_DIM,R=cfg.NUM_REGION,
                            D=cfg.GLOBAL_FEATURE_DIM,DR=cfg.REGION_FEATURE_DIM,pre_trained_embedding = cfg.PRE_EMB)
        self.cider = Cider()
        self.spice = Spice()
        self.cider_max = 0

    def train_op(self,loss,learning_rate): 
            optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.8)
            train_op = optimizer.minimize(loss)
            return train_op

    def calculate_rewards(self,feed_dict,sampled,sampled_decoded,argmax_captions,ref_key,compute=True,metric = 'cider'):
        if compute: 
            argmax = np.array(self.sess.run(argmax_captions,feed_dict))
            argmax_decoded = decode_captions(argmax, self.model.idx_to_word)#(N,T)
            cand = sampled_decoded + argmax_decoded
            if metric == 'cider':
                score = self.cider.compute_score(cand,np.concatenate((ref_key,ref_key)))
                score = np.array(score)
            else:
                score = self.spice.compute_score(cand,np.concatenate((ref_key,ref_key)))
                score = np.array(score)
            rewards = score[:self.n_sentences] - score[self.n_sentences:] 
            mask = np.ones((self.n_sentences,19))
            
            for indx in range(self.n_sentences):
                row = sampled[indx,:].tolist()
                if 2 in row:
                    idx = row.index(2)+1
                    mask[indx,idx:] = 0 
        else:
            mask = np.zeros((self.n_sentences,19))
            rewards = np.zeros((self.n_sentences,))
            score = np.zeros((self.n_sentences*2,))
        return rewards,mask,score,sampled


    def train(self):

            # train/val dataset
            captions = load_pickle('/data1/junjiaot/data/train/train.references.pkl')

            # build model
            loss = self.model.build_graph(is_training=True)
            print('build success')
            with tf.variable_scope(tf.get_variable_scope()):   
                tf.get_variable_scope().reuse_variables()
                sampled_captions2,_ = self.model.build_graph(mode='multinomial',is_training=False)
                argmax_captions2,alpha = self.model.build_graph(mode='argmax',is_training=False)
            print('build sampler success')

            print('computing df-idf')
            self.cider.init_docs(captions)
            print('Done computing')
            self.spice.init_docs(captions)
            # learning rate decay
            #train op
            train_op = self.train_op(loss,cfg.LR)

           
            print('Learning rate:{}'.format(cfg.LR))
            print ("The number of epoch: %d" %cfg.MAX_EPOCH)
            print ("Batch size: %d" %cfg.BATCH_SIZE)

            #self._print_trainable()
            #import ipdb;ipdb.set_trace()

            train_df = get_train_dataflow()
            eval_df = get_eval_dataflow()
            config = tf.ConfigProto(allow_soft_placement = True)
            #config.gpu_options.per_process_gpu_memory_fraction=0.9
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as self.sess:
                self.sess.run(tf.global_variables_initializer())
                self.max_saver = tf.train.Saver(max_to_keep=1)
                if cfg.LOAD is not None:
                    print ("Start with pretrained model..")
                    self.max_saver.restore(self.sess, cfg.LOAD)

                for self.e in range(cfg.START_EPOCH, cfg.MAX_EPOCH):
                    print('epoch:{}, learning_rate:{}'.format(self.e,cfg.LR))
                    #self.evaluation(argmax_captions2,alpha,eval_df)
                    start_t = time.time()
                    train_df.reset_state()
                    for i,datapoint in enumerate(train_df):
                        # datapoint: mean feature, region feature, box, ref captions
                        [mean_feature,region_feature,box,hist,ref_key] = datapoint
                        
                        self.n_sentences = len(ref_key)

                        feed_dict = {self.model.mean_feature: mean_feature, self.model.region_feature:region_feature,
                                    self.model.box:box, self.model.histogram:hist}

                        sampled = np.array(self.sess.run(sampled_captions2,feed_dict))
                        sampled_decoded = decode_captions(sampled, self.model.idx_to_word)#(N,T)
                        if i%2 == 0:
                            rewards,mask,spice_score,sampled_actions2 = self.calculate_rewards(feed_dict,sampled,sampled_decoded,argmax_captions2,ref_key,metric='spice')
                            rewards2,mask2,cider_score,sampled_actions2 = self.calculate_rewards(feed_dict,sampled,sampled_decoded,argmax_captions2,ref_key,metric='cider') 
                        else:
                            rewards,mask,spice_score,sampled_actions2 = self.calculate_rewards(feed_dict,sampled,sampled_decoded,argmax_captions2,ref_key,compute=False)
                            rewards2,mask2,cider_score,sampled_actions2 = self.calculate_rewards(feed_dict,sampled,sampled_decoded,argmax_captions2,ref_key,metric='cider')
                        
                        feed_dict.update({self.model.rewards:rewards[:,np.newaxis],
                                          self.model.rewards2:rewards2[:,np.newaxis],
                                          #self.model.mask: mask,
                                          self.model.mask2: mask2,
                                          #self.model.actions:sampled_actions,
                                          self.model.actions2:sampled_actions2})
                        #import ipdb; ipdb.set_trace()
                        _,l = self.sess.run([train_op,loss],feed_dict=feed_dict)
                        cider_avg = np.mean(cider_score[self.n_sentences:])
                        spice_avg = np.mean(spice_score[self.n_sentences:])
                        #critic_avg = np.mean(cider_score[n_sentences:])
                        #import ipdb;ipdb.set_trace()
                        if i % cfg.PRINT_EVERY == 0:
                            #print ("\n Aerage Cider and Spice at epoch {} iteration {}: {} {} loss: {}" .format(self.e, i, cider_avg,spice_avg,l))
                            print ("\n Aerage Cider and Spice at epoch {} iteration {}: {} {} loss: {}" .format(self.e, i, cider_avg,spice_avg,l))
                            

                        # print out BLEU scores and file write
                        if i % cfg.EVAL_PERIOD == 0:
                            self.evaluation(argmax_captions2,alpha,eval_df)
                    print ("Elapsed time: ", time.time() - start_t)

    def evaluation(self,argmax_captions2,alpha,eval_df):
        eval_df.reset_state()
        all_gen_cap = []
        all_alpha = []
        for datapoint in eval_df:
            [mean_feature,region_feature,box,hist] = datapoint
            feed_dict = {self.model.mean_feature: mean_feature, self.model.region_feature:region_feature,
                                             self.model.box:box, self.model.histogram:hist}
            gen_cap,alpha_module= self.sess.run([argmax_captions2,alpha],feed_dict)
            all_gen_cap.extend(gen_cap)
            all_alpha.extend(alpha_module)

        all_gen_cap = np.array(all_gen_cap)
        all_alpha = np.array(all_alpha)
        all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)#(N,T)
        sample_index = np.random.randint(0,len(all_decoded),size=1)
        for index in sample_index:
            print(all_decoded[index])
            print(all_alpha[index,:10,:])
        save_pickle(all_decoded,os.path.join(cfg.DATA.BASEDIR,'val/val.'+cfg.EVAL_CAP))
        print('Calculating scores...')
        scores,spice_scores = evaluate(data_path=cfg.DATA.BASEDIR, split='val', eval_file=cfg.EVAL_CAP,get_scores=True)
        if scores['CIDEr'] > self.cider_max:
                self.cider_max = scores['CIDEr']
                self.max_saver.save(tf.get_default_session(),os.path.join(cfg.CHECKPOINT_DIR_MAX, 'model_max'),global_step=tf.train.get_global_step(),write_meta_graph=False)
                print('Model saved to ' + os.path.join(cfg.CHECKPOINT_DIR_MAX, 'model'))
        g=tf.get_default_graph()
        write_bleu(scores=scores,spice_scores=spice_scores, path=cfg.SCORE_DIR, epoch=self.e,lr=cfg.LR)

if __name__ == '__main__':
    trainer = CaptioningSolver()
    trainer.train()
