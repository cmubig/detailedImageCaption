#!/usr/bin/env python
from __future__ import division
import tensorflow as tf
slim = tf.contrib.slim
import pickle
import numpy as np
import sys
from .preprocessing import image_processing
from .nets import nets_factory
from utils import *
import rospkg



class model(object):
    def __init__(self, dim_feature=[64, 1536], dim_embed=300, dim_hidden=512, n_time_step=19):
        rospack = rospkg.RosPack()
        source_dir = rospack.get_path('captioner') + "/models"
        self.word_to_idx = load_pickle(source_dir+'/train/word_to_idx.pkl')
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        self.V = len(self.word_to_idx)
        self.R = dim_feature[0]
        self.DR = dim_feature[1]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.LSTM = dim_hidden
        self.T = n_time_step
        self.H = 512
        self.num_mod = 5 # number of modules
        self.MIL_sz = 616
        self.image_size = 299
        self.image = tf.placeholder(tf.uint8,[None,None,3],'image')


        self._start = self.word_to_idx['<START>']
        self._null = self.word_to_idx['<NULL>']
        self.CNN_MODEL = 'inception_v4'#resnet_v1_152'
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        self.pretrained_model = source_dir+'/model/model_lejeune/model_max'#source_dir+'/models/model_lejeune/model_max-113855'#
        self.pretrained_resnet = source_dir+'/Inception/inception_v4.ckpt' 

        self.MIL_list = np.array(load_pickle(source_dir+'/train/multi_class_labels_list.pkl'))
        #self.env_list = np.array(load_pickle(source_dir+'/train/landmark_list.pkl')) #(44)
        
        self.load_scopes_model = 'word_embedding, image_features,MIL, attribute_attention_layer,lstm, v_lstm2, decode_layer, attention_pivot,attention_map'
        self.checkpoint_exclude_scopes =  'InceptionV4/AuxLogits,InceptionV4/Logits,InceptionV4/Predictions,InceptionV4/PreLogits'+ self.load_scopes_model 

    def init(self,sess):
        #language model saver
        model_variables_to_train = self._get_variables_to_save(self.load_scopes_model)
        saver_model = tf.train.Saver(model_variables_to_train)
        if self.pretrained_model is not None:
            print ("Start with pretrained Model..")
        saver_model.restore(sess, self.pretrained_model)
        
        #resnet saver
        with tf.get_default_graph().as_default():
            print('Start with pretrained cnn')
            variables_to_restore = self.get_init_fn()
        saver_resnet = tf.train.Saver(variables_to_restore,max_to_keep = 1)
        saver_resnet.restore(sess,self.pretrained_resnet)
        
    def _get_variables_to_save(self,trainable_scopes):
        scopes = [scope.strip() for scope in trainable_scopes.split(',')]
        variables_to_save = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)#+tf.get_collection(tf.GraphKeys.MODEL_VARIABLES ,scope)
            variables_to_save.extend(variables)
            for variable in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES ,scope):
                if variable not in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope):
                    variables_to_save.append(variable)
        return variables_to_save
        
    def get_init_fn(self):
        """Returns a function run by the chief worker to warm-start the training."""
        #with G.as_default():
        exclusions = [scope.strip() for scope in self.checkpoint_exclude_scopes.split(',')]
        variables_to_restore = []
        for var in slim.get_model_variables():
        #for var in tf.trainable_variables(scope ='InceptionV3' ):
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
        return variables_to_restore
                
    def _image_features(self,mean_feature,region_feature,reuse=False):
        with tf.variable_scope('image_features',reuse=reuse):
            mean_feature = tf.layers.dense(mean_feature,self.LSTM,activation=tf.nn.relu,name='image_feature') #(N,D)
            region_feature = tf.reshape(region_feature,[-1,self.DR])
            region_feature = tf.layers.dense(region_feature,self.LSTM,activation=tf.nn.relu,name='region_feature')

            region_feature = tf.reshape(region_feature,[-1,self.R,self.LSTM]) #(N,R,LSTM)
        return mean_feature,region_feature
    
    def _init_image_feature(self,mean_feature,reuse=False):
        with tf.variable_scope('lstm_init',reuse=reuse):
            init = tf.layers.dense(mean_feature,2*self.LSTM,activation=tf.nn.relu,name='lstm_init') #(N,3*LSTM)
        return init

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            #w.assign(self.word_embedding)
            #w = tf.stop_gradient(w)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _word_embedding_softmx(self, inputs):
        '''
        inputs: N,V
        '''
        with tf.variable_scope('word_embedding', reuse=True):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)  
            w = tf.stop_gradient(w)
            return tf.matmul(inputs,w) #N,M

    def _attention_pivot(self, region_feature ,h,is_training=True,reuse=False):
        with tf.variable_scope('attention_pivot', reuse=reuse):
            w_v = tf.get_variable('w_v', [self.LSTM,self.H], initializer=self.weight_initializer)
            w_g = tf.get_variable('w_g', [self.LSTM, self.H], initializer=self.weight_initializer)
            w_h = tf.get_variable('w_h', [self.H, 1], initializer=self.weight_initializer)

            temp_v = tf.matmul(tf.reshape(region_feature,[-1,self.LSTM]),w_v)#(N*R,512)
            temp_v = tf.reshape(temp_v,[-1,self.R,512]) #(N,R,512)
            content_v = tf.nn.tanh(temp_v + tf.expand_dims(tf.matmul(h,w_g),1)) #(N,R,512)
            
            z_t = tf.reshape(tf.matmul(tf.reshape(content_v,[-1,512]),w_h),[-1,self.R]) #(N,R)
            alpha = tf.nn.softmax(z_t)
            c = tf.reduce_sum(region_feature * tf.expand_dims(alpha, 2), 1, name='pivot') #(N,H)
            return c,alpha

    def _attention_module(self,h_m,vs,h,is_training=True,reuse=False):
        # W_v * V + W_g * h_t
        # h (N,LSTM); h_m (N,num_mod,LSTM); vs (N,LSTM)
        # c (N,LSTM)
                
        with tf.variable_scope('attention_module', reuse=reuse):
            w_v = tf.get_variable('w_v', [self.M, self.H], initializer=self.weight_initializer)
            w_g = tf.get_variable('w_g', [self.LSTM, self.H], initializer=self.weight_initializer)
            w_s = tf.get_variable('w_s', [self.M, self.H], initializer=self.weight_initializer)
            w_h = tf.get_variable('w_h', [self.H, 1], initializer=self.weight_initializer)

            temp_v = tf.matmul(tf.reshape(h_m,[-1,self.M]),w_v)   #(N*5,512)
            temp_v = tf.reshape(temp_v,[-1,self.num_mod,512]) #(N,5,512)
            content_v = tf.nn.tanh(temp_v + tf.expand_dims(tf.matmul(h,w_g),1)) #(N,5,512)

            z_t = tf.reshape(tf.matmul(tf.reshape(content_v,[-1,self.H]),w_h),[-1,self.num_mod]) #(N,5)
            #z_t_stopped = tf.stop_gradient(z_t)
            alpha = tf.nn.softmax(z_t) #(N,5)
            c = tf.reduce_sum(h_m * tf.expand_dims(alpha, 2), 1, name='h_m') #(N,H)
            
            content_s = tf.nn.tanh(tf.matmul(vs,w_s) + tf.matmul(h,w_g)) #(N,K)
            z_t_extended = tf.matmul(content_s,w_h) #(N,1)
            extended = tf.concat([z_t,z_t_extended],1)
            alpha_hat = tf.nn.softmax(extended) #(N,K+1)
            beta = tf.reshape(alpha_hat[:,-1],[-1,1],name = 'beta') #(N,1)
            c_hat = tf.multiply(vs,beta) + tf.multiply(c,(1-beta)) # (N,H)
            
            return c_hat,extended,alpha,beta

    def _attention_map(self,features,reuse = False):
      with tf.variable_scope('attention_map',reuse = reuse):
        w_a = tf.get_variable('w_a', [1024, 1], initializer=self.weight_initializer)
        b_a = tf.get_variable('b_a', initializer=0.0)
        v = tf.get_variable('v', [1536, 1024], initializer=self.weight_initializer) #self.Hself.weight_initializertf.contrib.layers.xavier_initializer()) #self.H
        u = tf.get_variable('u', [1536, 1024], initializer=self.weight_initializer) #self.H

        fc_flattened = tf.reshape(features,[-1,1536]) #(c*324,2048)
        temp_v = tf.nn.tanh(tf.matmul(fc_flattened,v)) # (c*64,1024)
        temp_u = tf.nn.sigmoid(tf.matmul(fc_flattened,u)) # (c*64,1024)
        alpha_logit = tf.reshape(tf.matmul(tf.multiply(temp_v,temp_u),w_a)+b_a,[-1,self.R,1])
        alpha = tf.nn.softmax(alpha_logit,dim=1) #(c,64,1)
        z = tf.reduce_sum(alpha * tf.reshape(fc_flattened,[-1,64,1536]),1) #(c,512)
        return z

    def MIL_classifier(self,features,mean):
      '''
      features: (n,8,8,1536)
      '''
      with tf.variable_scope('MIL'):
            mean_activation = tf.layers.dense(mean,1024,activation=tf.nn.relu) #(c,1024)
            mean_activation = tf.layers.dense(mean_activation,self.MIL_sz,activation=tf.nn.relu) #(c,616)

            fc_flattened = tf.reshape(features,[-1,self.DR],'reshape_orig') #(c*36,2048)
            fc_flattened = tf.layers.dense(fc_flattened,1024,activation=tf.nn.relu) #(c*36,1024)
            fc_flattened = tf.layers.dense(fc_flattened,2048,activation=tf.nn.relu) #(c*36,512)
            fc_flattened = tf.layers.dense(fc_flattened,self.MIL_sz,activation=tf.nn.sigmoid) #(c*36,616)

            fc8 = tf.reshape(fc_flattened,[-1,self.R,self.MIL_sz]) #(c,36,44)
            mil_1 = 1 - fc8 #(c,25,44)
            max_prob = tf.reduce_max(fc8, axis = 1) #(c,44)
            prob = 1 - tf.reduce_prod(mil_1,[1])
            final_prob = tf.maximum(max_prob,prob)

            leak_prob = tf.nn.sigmoid(mean_activation)
            max_prob2 = tf.maximum(tf.reduce_max(fc8, axis = 1),leak_prob)
            prob2 = 1 - tf.multiply(tf.reduce_prod(mil_1,[1]),1-leak_prob) #(c,44)
            final_prob2 = tf.maximum(max_prob2,prob2)
            return final_prob,mean_activation,final_prob2

    def env_classifier(self,features,z):
      '''
      features: (n,8,8,1536)
      '''
      with tf.variable_scope('env'):
        mean_activation = tf.layers.dense(z,1024,activation=tf.nn.relu) #(c,1024)
        mean_activation = tf.layers.dropout(mean_activation,rate=0.5,training=is_training)#(N,D)
        mean_activation = tf.layers.dense(mean_activation,self.env_sz,activation=tf.nn.relu) #(c,616)

        fc6 = tf.layers.conv2d(features,filters = 2048,kernel_size = 3,activation = tf.nn.relu,padding = 'VALID') #(c,6,6,4096)
        #fc6 = slim.batch_norm(fc6, is_training = is_training,activation_fn=tf.nn.relu)
        #fc6 = tf.layers.batch_normalization(fc6,training=is_training)
        fc7 = tf.layers.conv2d(fc6,filters = 1024,kernel_size = 1,activation = tf.nn.relu)#(c,6,6,4096)
        #fc7 = slim.batch_norm(fc7, is_training = is_training,activation_fn=tf.nn.relu)          
        #fc7 = tf.layers.batch_normalization(fc7,training=is_training)
        fc8 = tf.layers.conv2d(fc7,filters = self.env_sz,kernel_size = 1,activation = tf.nn.sigmoid,
                           bias_initializer=tf.constant_initializer(-6.8),
                           kernel_initializer=tf.truncated_normal_initializer(stddev = 0.001))#(c,6,6,44)
        #import ipdb;ipdb.set_trace()
        fc8 = tf.reshape(fc8,[-1,36,self.env_sz]) #(c,36,44)
        mil_1 = 1 - fc8 #(c,25,44)
        max_prob = tf.reduce_max(fc8, axis = 1) #(c,44)
        prob = 1 - tf.reduce_prod(mil_1,[1])
        final_prob = tf.maximum(max_prob,prob)

        leak_prob = tf.nn.sigmoid(mean_activation)
        max_prob2 = tf.maximum(tf.reduce_max(fc8, axis = 1),leak_prob)
        prob2 = 1 - tf.multiply(tf.reduce_prod(mil_1,[1]),1-leak_prob) #(c,44)
        final_prob2 = tf.maximum(max_prob2,prob2)
        return final_prob,mean_activation,final_prob2

    def _attribute_gate_layer(self,h,c,prob,reuse=False):
        with tf.variable_scope('attribute_attention_layer',reuse=reuse):
            w_z = tf.get_variable('w_z', [self.LSTM,2048 ], initializer=self.weight_initializer)
            b_z = tf.get_variable('b_z', [2048], initializer=self.const_initializer)
            w_z3 = tf.get_variable('w_z3', [2048, self.MIL_sz], initializer=self.weight_initializer)
            b_z3 = tf.get_variable('b_z3', [self.MIL_sz], initializer=self.const_initializer)
            w_h = tf.get_variable('w_h', [self.LSTM, 2048], initializer=self.weight_initializer)
            prob = tf.stop_gradient(prob)
            logit_z = tf.nn.relu(tf.matmul(c,w_z) + tf.matmul(h,w_h)+b_z)
            logit_z = tf.matmul(logit_z,w_z3)+b_z3#(N,H)
            attention = tf.multiply(tf.nn.sigmoid(logit_z),prob)
        return attention
        

    def _attribute_vector_op(self,batch_size,attention_logits,attribute_embedding_expand):
        # =  tf.nn.sigmoid(attention_logits)
        num = tf.reduce_sum(attention_logits,axis = 1,keep_dims = True)
        mask = tf.to_float(tf.equal(num, 0))        
        attribute_attention = attention_logits/(num+mask)#(N,1113)
        attribute_vector = tf.multiply(tf.expand_dims(attribute_attention,2),tf.tile(attribute_embedding_expand,[batch_size,1,1])) #(N,1113,M)
        attribute_vector = tf.reduce_sum(attribute_vector,1) #(N,M)
        return attribute_vector,attribute_attention

    def _decode_layer_copy(self,h):
        with tf.variable_scope('decode_layer', reuse=True):
            w = tf.get_variable('w', [self.LSTM, self.V], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.V], initializer=self.const_initializer)
            w = tf.stop_gradient(w)
            b = tf.stop_gradient(b)
            logits = tf.matmul(h,w)+b
        return logits

    def _decode_layer(self,h,reuse=False):
        with tf.variable_scope('decode_layer', reuse=reuse):
            w = tf.get_variable('w', [self.LSTM, self.V], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.V], initializer=self.const_initializer)
            logits = tf.matmul(h,w)+b
        return logits

    def color_module(self,pivot,h,dim,is_dropout=True,reuse=True):
        with tf.variable_scope('color_module', reuse=reuse): 
            l =  tf.layers.dense(tf.concat([pivot,h],axis=1), 512, activation=tf.nn.relu, name='h_cl')
            l = tf.layers.dense(tf.concat([pivot,h],axis=1), dim, activation=tf.nn.relu, name='h_cl2')
        return l

    def size_module(self,pivot,h,dim,is_dropout=True,reuse=True):
        with tf.variable_scope('size_module', reuse=reuse):
            #l =  tf.layers.dense(tf.concat([pivot,reference],axis=1),dim, activation=tf.nn.relu, name='h_sz') # N x 512
            l =  tf.layers.dense(tf.concat([pivot,h],axis=1),512, activation=tf.nn.relu, name='h_sz') # N x 512tf.truediv
            l =  tf.layers.dense(tf.concat([pivot,h],axis=1),dim, activation=tf.nn.relu, name='h_sz2') # N x 512tf.truediv
        return l


    def count_module(self,pivot,h,dim,is_dropout=True,reuse=True):
        with tf.variable_scope('count_module', reuse=reuse):
            l = tf.layers.dense(tf.concat([pivot,h],axis=1), 512, activation=tf.nn.relu, name='h_ct')
            l = tf.layers.dense(tf.concat([pivot,h],axis=1), dim, activation=tf.nn.relu, name='h_ct2')
        return l

    def spatial_module(self,pivot,h,dim,is_dropout=True,reuse=True):
        with tf.variable_scope('spatial_module', reuse=reuse):
            l = tf.layers.dense(tf.concat([pivot,h],axis=1), 512, activation=tf.nn.relu, name='h_ct')
            l = tf.layers.dense(tf.concat([pivot,h],axis=1), dim, activation=tf.nn.relu, name='h_ct2')
        return l


    def semantic_module(self,pivot,h,dim,is_dropout=True,reuse=True):
        with tf.variable_scope('semantic_module', reuse=reuse):
            l = tf.layers.dense(tf.concat([pivot,h],axis=1), 512, activation=tf.nn.relu, name='h_ct')
            l = tf.layers.dense(tf.concat([pivot,h],axis=1), dim, activation=tf.nn.relu, name='h_ct2')
        return l


    def build_sampler(self,max_len=20):
        sampled_word_list = []
        image = tf.image.convert_image_dtype(self.image, dtype=tf.float32)
        image_val = image_processing.process_image(image,is_training=False, height = self.image_size, width =self.image_size )
        network_fn = nets_factory.get_network_fn(
                      self.CNN_MODEL,
                      num_classes=1001,
                      weight_decay=0.0001,
                      is_training=False)
        logits,end_points = network_fn(tf.expand_dims(image_val,0))
        region_feature = end_points['Mixed_7d'] #(N,8,8,1536)
        mean_feature = tf.reduce_mean(tf.reshape(region_feature,[-1,self.R,1536]),axis=1) #(N,1536)
        batch_size = tf.shape(mean_feature)[0]

        mean_proj,region_proj = self._image_features(mean_feature,region_feature,reuse=False) # (N,H);(N,R,LSTM) 
        lstm_init = self._init_image_feature(mean_feature,reuse=False)
        
        MIL_embedding = tf.expand_dims(self._word_embedding(self.MIL_list,reuse = False),0) #(1,N,616)


        lstm_cell_v = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.LSTM)
        (c_v,h_v) = lstm_cell_v.zero_state(batch_size=batch_size, dtype=tf.float32)

        lstm_cell_v2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.LSTM)
        (c_v2,h_v2) = lstm_cell_v2.zero_state(batch_size=batch_size, dtype=tf.float32)

        z = self._attention_map(region_feature,reuse = False)
        final_prob,mean_activation,final_prob2 = self.MIL_classifier(region_feature,z)
        MIL_attribute,_ = self._attribute_vector_op(batch_size,final_prob2,MIL_embedding)

        with tf.variable_scope('lstm', reuse=False):
            _, (c_v, h_v) = lstm_cell_v(inputs=tf.concat((lstm_init,MIL_attribute),axis=1), state= (c_v, h_v))

        with tf.variable_scope('v_lstm2', reuse=False):
            _, (c_v2, h_v2) = lstm_cell_v2(inputs=tf.concat((lstm_init,MIL_attribute),axis=1), state=(c_v2, h_v2)) 

        for t in range(self.T):
            if t == 0:
                previous_word = self._word_embedding(inputs=tf.fill([batch_size], self._start),reuse=True)
            else: 
                previous_word = self._word_embedding(inputs=sampled_word, reuse=True)
            x_new_v = tf.concat([h_v2,mean_proj,previous_word],1)
            with tf.variable_scope('lstm', reuse=True):
                _, (c_v, h_v) = lstm_cell_v(inputs=x_new_v, state=(c_v, h_v))

            pivot,alpha_pivot = self._attention_pivot(region_proj,h_v,reuse=(t!=0)) #(N,H) (N,R)
            pivot_stopped = tf.stop_gradient(pivot)

            final_prob2_gated = self._attribute_gate_layer(h_v,pivot,final_prob2,reuse=(t!=0))
            MIL_attribute,_ = self._attribute_vector_op(batch_size,final_prob2_gated,MIL_embedding)

            x_new_v2 = tf.concat([pivot,h_v,MIL_attribute],1) #(2*LSTM)
            with tf.variable_scope('v_lstm2', reuse=True):
                _, (c_v2, h_v2) = lstm_cell_v2(inputs=x_new_v2, state=(c_v2, h_v2))

            logits2 = self. _decode_layer(h_v2,reuse=(t!=0))        

            sampled_word = tf.argmax(logits2, 1,output_type=tf.int32)
            sampled_word_list.append(sampled_word)
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
        return sampled_captions
