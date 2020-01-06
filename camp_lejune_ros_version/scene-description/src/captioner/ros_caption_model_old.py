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
    def __init__(self, dim_feature=[64, 2048], dim_embed=512, dim_hidden=512, n_time_step=19):
        rospack = rospkg.RosPack()
        source_dir = rospack.get_path('captioner') + "/models"
        self.word_to_idx = load_pickle(source_dir+'/train/word_to_idx.pkl')
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        self.V = len(self.word_to_idx)
        self.K = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = self.word_to_idx['<START>']
        self._null = self.word_to_idx['<NULL>']
        self.CNN_MODEL = 'inception_v3'#resnet_v1_152'
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        self.pretrained_model = source_dir+'/model/4000-21'
        self.pretrained_resnet = source_dir+'/resnet/4000-20' 
        self.pretrained_classifier = source_dir+'/classifier/4000-21'
        self.pretrained_human_classifier = source_dir+'/human_classifier/42-8'

        self.image_size = 299
        self.trainable_scopes_human_classifier = 'wilder'
        self.load_scopes_model = 'lstm,initial_lstm,word_embedding,image_features,attention_layer,the_second_attention_layer,Visual_Sentinel,decode_layer,attribute_attention_layer,decision_layer'
        self.trainable_scopes_classifier = 'MIL'#'MIL'
        self.checkpoint_exclude_scopes = 'InceptionV3/AuxLogits,InceptionV3/Logits,InceptionV3/Predictions,InceptionV3/PreLogits,' + self.load_scopes_model + ',' + self.trainable_scopes_classifier +',' +self.trainable_scopes_human_classifier
        self.attribute_list = np.array(load_pickle(source_dir+'/train/multi_class_labels_list.pkl'))
        self.num_attributes = 4
        self.image = tf.placeholder(tf.uint8,[None,None,3],'image')

    def init(self,sess):
        #language model saver
        model_variables_to_train = self._get_variables_to_save(self.load_scopes_model)
        saver_model = tf.train.Saver(model_variables_to_train)
        if self.pretrained_model is not None:
            print ("Start with pretrained Model..")
        saver_model.restore(sess, self.pretrained_model)
        #Attribute classifier saver
        classifier_variables_to_train = self._get_variables_to_save(self.trainable_scopes_classifier)
        saver_classifier = tf.train.Saver(classifier_variables_to_train)
        if self.pretrained_classifier is not None:
            print('Start with pretrained classifier')
            saver_classifier.restore(sess,self.pretrained_classifier)
        #resnet saver
        with tf.get_default_graph().as_default():
            print('Start with pretrained cnn')
            variables_to_restore = self.get_init_fn()
        saver_resnet = tf.train.Saver(variables_to_restore,max_to_keep = 1)
        saver_resnet.restore(sess,self.pretrained_resnet)
        #Attribute classifier saver
        human_classifier_variables_to_train = self._get_variables_to_save(self.trainable_scopes_human_classifier)
        saver_human_classifier = tf.train.Saver(human_classifier_variables_to_train)
        if self.pretrained_classifier is not None:
            print('Start with pretrained human classifier')
            saver_human_classifier.restore(sess,self.pretrained_human_classifier)

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
                
    def _get_initial_lstm(self, features):
        '''
        features: (N,K,H)
        '''
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1) #(N,H)
            w_h = tf.get_variable('w_h', [self.H, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)
            w_c = tf.get_variable('w_c', [self.H, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h


    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x


    def _image_features(self, features ,global_feature = True,reuse = False):
        '''
        # features (N,K,H)
        # v_i = ReLu(W_a * a_i),V_g = ReLu(W_b * a_g)
        #I do not want to noisy global image representation instead
        # I would like to use high level concepts to initialize the network
        '''
        with tf.variable_scope('image_features',reuse=reuse):
            w_a = tf.get_variable('w_a', [self.D, self.H], initializer=self.weight_initializer)
            b_a = tf.get_variable('b_a', [self.H], initializer=self.const_initializer)
            features_flat = tf.reshape(features, [-1, self.D])      #(N*K, D)
            features_proj = tf.nn.relu(tf.matmul(features_flat, w_a)+b_a) #(N*K, H)
            features_proj = tf.reshape(features_proj, [-1, self.K, self.H])     #(N,K,H)
            if global_feature == True:
                w_b = tf.get_variable('w_b', [self.D, self.M], initializer=self.weight_initializer)
                b_b = tf.get_variable('b_b', [self.M], initializer=self.const_initializer)
                features_global = tf.reduce_mean(features,axis=1) #(N,D) average pooling
                features_global = tf.nn.relu(tf.matmul(features_global, w_b)+b_b,name = 'global_feature') #(N,M)
                return features_proj,features_global
        return features_proj


    def _attention_layer(self, features_proj, features_orig ,h, vs,reuse=False):
        '''
        # W_v * V + W_g * h_t
        # h (N,H); features_proj (N,K,H); vs (N,H)
        # c (N,H)
        '''
        with tf.variable_scope('attention_layer', reuse=reuse):
            w_v = tf.get_variable('w_v', [self.H, 512], initializer=self.weight_initializer)
            w_g = tf.get_variable('w_g', [self.H, 512], initializer=self.weight_initializer)
            w_s = tf.get_variable('w_s', [self.H, 512], initializer=self.weight_initializer)
            w_h = tf.get_variable('w_h', [512, 1], initializer=self.weight_initializer)

            temp_v = tf.matmul(tf.reshape(features_proj,[-1,self.H]),w_v)   #(N*K,512)
            temp_v = tf.reshape(temp_v,[-1,self.K,512]) #(N,K,512)
            content_v = tf.nn.tanh(temp_v + tf.expand_dims(tf.matmul(h,w_g),1)) #(N,K,512)

            z_t = tf.reshape(tf.matmul(tf.reshape(content_v,[-1,512]),w_h),[-1,self.K]) #(N,K)
            alpha = tf.nn.softmax(z_t) #(N,K)
            c = tf.reduce_sum(features_proj * tf.expand_dims(alpha, 2), 1, name='context') #(N,H)
            context_vector_full = tf.reduce_sum(features_orig * tf.expand_dims(alpha, 2), 1, name='context_full')

            content_s = tf.nn.tanh(tf.matmul(vs,w_s) + tf.matmul(h,w_g)) #(N,K)
            z_t_extended = tf.matmul(content_s,w_h) #(N,1)
            extended = tf.concat([z_t,z_t_extended],1)
            alpha_hat = tf.nn.softmax(extended) #(N,K+1)
            beta = tf.reshape(alpha_hat[:,-1],[-1,1],name = 'beta') #(N,1)
            c_hat = tf.multiply(vs,beta) + tf.multiply(c,(1-beta)) # (N,H)

            return context_vector_full,c_hat, alpha, beta


    def _attention_layer2(self, features_proj, features_orig ,h,alpha_in, vs,reuse=False):
        '''
        # W_v * V + W_g * h_t
        # h (N,H); features_proj (N,K,H); vs (N,H)
        # c (N,H)
        what about inverse feature (??????)
        '''
        with tf.variable_scope('the_second_attention_layer', reuse=reuse):
            w_v = tf.get_variable('w_v', [self.H, 512], initializer=self.weight_initializer)
            w_g = tf.get_variable('w_g', [self.H, 512], initializer=self.weight_initializer)
            w_s = tf.get_variable('w_s', [self.H, 512], initializer=self.weight_initializer)
            w_h = tf.get_variable('w_h', [512, 1], initializer=self.weight_initializer)

            features_proj_alpha = features_proj * tf.expand_dims(alpha_in, 2)
            temp_v = tf.matmul(tf.reshape(features_proj_alpha,[-1,self.H]),w_v)   #(N*K,512)
            temp_v = tf.reshape(temp_v,[-1,self.K,512]) #(N,K,512)
            content_v = tf.nn.tanh(temp_v + tf.expand_dims(tf.matmul(h,w_g),1)) #(N,K,512)

            z_t = tf.reshape(tf.matmul(tf.reshape(content_v,[-1,512]),w_h),[-1,self.K]) #(N,K)
            alpha = tf.nn.softmax(z_t) #(N,K)

            #alpha_mask = tf.to_float(tf.greater_equal(alpha,alpha_in))
            #alpha = tf.nn.softmax(tf.multiply(alpha_mask,z_t))

            c = tf.reduce_sum(features_proj * tf.expand_dims(alpha, 2), 1, name='context') #(N,H)
            context_vector_full = tf.reduce_sum(features_orig * tf.expand_dims(alpha, 2), 1, name='context_full')

            content_s = tf.nn.tanh(tf.matmul(vs,w_s) + tf.matmul(h,w_g)) #(N,K)
            z_t_extended = tf.matmul(content_s,w_h) #(N,1)
            extended = tf.concat([z_t,z_t_extended],1)
            alpha_hat = tf.nn.softmax(extended) #(N,K+1)
            beta = tf.reshape(alpha_hat[:,-1],[-1,1],name = 'beta') #(N,1)
            c_hat = tf.multiply(vs,beta) + tf.multiply(c,(1-beta)) # (N,H)

            return context_vector_full,c_hat, alpha, beta

    def WILDER_attention_map(self,features,reuse = False):
        with tf.variable_scope('wilder',reuse = reuse):
          w_a = tf.get_variable('w_a', [1024, 1], initializer=tf.contrib.layers.xavier_initializer())
          b_a = tf.get_variable('b_a', [1], initializer=tf.constant_initializer(0.0))
          v = tf.get_variable('v', [2048, 1024], initializer=tf.contrib.layers.xavier_initializer())
          u = tf.get_variable('u', [2048, 1024], initializer=tf.contrib.layers.xavier_initializer())
        fc_flattened = tf.reshape(features,[-1,2048]) #(c*324,2048)
        temp_v = tf.nn.tanh(tf.matmul(fc_flattened,v)) # (c*49,1024)
        temp_u = tf.nn.sigmoid(tf.matmul(fc_flattened,u)) # (c*49,1024)
        alpha_logit = tf.reshape(tf.matmul(tf.multiply(temp_v,temp_u),w_a)+b_a,[-1,64,1])
        alpha = tf.nn.softmax(alpha_logit,dim=1) #(c,49,1)
        z = tf.reduce_sum(alpha * tf.reshape(fc_flattened,[-1,64,2048]),1) #(c,2048)
        return z,alpha

    def WILDER_attention(self,z,reuse = False):
        with tf.variable_scope('wilder',reuse=reuse):
          w = tf.get_variable('w', [2048, self.num_attributes], initializer=tf.contrib.layers.xavier_initializer())
          b = tf.get_variable('b', [self.num_attributes], initializer=tf.constant_initializer(0.0))
          w_2 = tf.get_variable('w_2', [2048, 2048], initializer=tf.contrib.layers.xavier_initializer())
          b_2 = tf.get_variable('b_2', [2048], initializer=tf.constant_initializer(0.0))
        temp_logit = tf.nn.relu(tf.matmul(z,w_2)+b_2) #()
        logits = tf.matmul(temp_logit,w)+b #(N,14)
        return logits

    def WILDER_Noisy_or(self,features,attention_logit,is_training,reuse=False):
      with tf.variable_scope('wilder',reuse=reuse):
        fc6 = tf.layers.conv2d(features,filters = 4096,kernel_size = 3,activation = None, padding ='VALID') #(c,5,5,4096)
        fc6 = slim.batch_norm(fc6, is_training = is_training,activation_fn=tf.nn.relu)
        fc7 = tf.layers.conv2d(fc6,filters = 4096,kernel_size = 1,activation = None)#(c,5,5,4096)
        fc7 = slim.batch_norm(fc7, is_training = is_training,activation_fn=tf.nn.relu)

        fc8 = tf.layers.conv2d(fc7,filters = self.num_attributes,kernel_size = 1,activation = tf.nn.sigmoid,
                             bias_initializer=tf.constant_initializer(-6.58),
                             kernel_initializer=tf.truncated_normal_initializer(stddev = 0.001))#(c,5,5,14)
        fc8 = tf.reshape(fc8,[-1,36,self.num_attributes]) #(c,25,14)
        mil_1 = 1 - fc8 #(c,25,14)
        leak_prob = tf.nn.sigmoid(attention_logit)
        max_prob2 = tf.maximum(tf.reduce_max(fc8, axis = 1),leak_prob)
        prob2 = 1 - tf.multiply(tf.reduce_prod(mil_1,[1]),1-leak_prob) #(c,14)
        final_prob2 = tf.maximum(max_prob2,prob2)

        max_prob = tf.reduce_max(fc8, axis = 1) #(c,14)
        prob = 1 - tf.reduce_prod(mil_1,[1])
        final_prob = tf.maximum(max_prob,prob)
      return final_prob,final_prob2


    def _visual_sentinel(self,h,c,x,reuse = False):
        '''
        # h (N,H); c (N,H); x (N,M+H)
        '''
        input_size = self.M+self.M # if use word embedding plus global feauture
        with tf.variable_scope('Visual_Sentinel', reuse=reuse):
            w_x = tf.get_variable('w_x', [input_size, self.H], initializer=self.weight_initializer)
            w_h = tf.get_variable('w_h', [self.H, self.H], initializer=self.weight_initializer)
            gate_t = tf.matmul(x,w_x) + tf.matmul(h,w_h) #(N,H)
            vs = tf.nn.sigmoid(gate_t)*tf.nn.tanh(c) #(N,H)
        return vs


    def _decode_layer(self, h, c_hat,stop_gradient = True,reuse=False):
        '''
        detected_prob attriubtes probability: (Vc,N)
        h: (N,H); c_hat: (N,H)
        '''
        with tf.variable_scope('decode_layer', reuse=reuse):
            w_g = tf.get_variable('w_g',[self.H,self.V],initializer=self.weight_initializer)
            b_g = tf.get_variable('b_g', [self.V], initializer=self.const_initializer)
            logits = tf.matmul(c_hat+h,w_g) + b_g
        return logits

    def _static_attribute_attention_layer(self,features,reuse = False):
      with tf.variable_scope('MIL',reuse = reuse):
        w_a = tf.get_variable('w_a', [1024, 1], initializer=tf.contrib.layers.xavier_initializer())
        b_a = tf.get_variable('b_a', [1], initializer=tf.constant_initializer(0.0))
        v = tf.get_variable('v', [2048, 1024], initializer=tf.contrib.layers.xavier_initializer()) #self.H
        u = tf.get_variable('u', [2048, 1024], initializer=tf.contrib.layers.xavier_initializer()) #self.H

        fc_flattened = tf.reshape(features,[-1,2048]) #(c*324,2048)
        temp_v = tf.nn.tanh(tf.matmul(fc_flattened,v)) # (c*64,1024)
        temp_u = tf.nn.sigmoid(tf.matmul(fc_flattened,u)) # (c*64,1024)
        alpha_logit = tf.reshape(tf.matmul(tf.multiply(temp_v,temp_u),w_a)+b_a,[-1,64,1])
        alpha = tf.nn.softmax(alpha_logit,dim=1) #(c,64,1)
        z = tf.reduce_sum(alpha * tf.reshape(fc_flattened,[-1,64,2048]),1) #(c,512)
        return z,tf.squeeze(alpha,2)

    def _MIL_attention(self,beta,c,z,inital = False,reuse = False):
      with tf.variable_scope('MIL', reuse=reuse):
         w = tf.get_variable('w', [2048, 1113], initializer=tf.contrib.layers.xavier_initializer())
         b = tf.get_variable('b', [1113], initializer=tf.constant_initializer(0.0))
         w_2 = tf.get_variable('w_2', [2048, 2048], initializer=tf.contrib.layers.xavier_initializer())
         b_2 = tf.get_variable('b_2', [2048], initializer=tf.constant_initializer(0.0))
         z_hat = tf.multiply(z,beta) + tf.multiply(c,(1-beta)) # (N,H)
         temp_logit = tf.nn.relu(tf.matmul(z_hat,w_2)+b_2) #()
         logits = tf.matmul(temp_logit,w)+b #(N,1113)
         return logits

    def _MIL_Leaky_noisy_or(self,features,attention_logit,is_training,reuse=False):
        '''
        features: original resenet conv feature
        features :#(N,K,H) -> #(N,7,7,H)
        '''
        with tf.variable_scope('MIL',reuse=reuse):
            fc6 = tf.layers.conv2d(features,filters = 4096,kernel_size = 3,activation = None,padding = 'VALID') #(c,5,5,4096)
            fc6 = slim.batch_norm(fc6, is_training = is_training,activation_fn=tf.nn.relu)

            fc7 = tf.layers.conv2d(fc6,filters = 4096,kernel_size = 1,activation = None)#(c,5,5,4096)
            fc7 = slim.batch_norm(fc7, is_training = is_training,activation_fn=tf.nn.relu)          

            fc8 = tf.layers.conv2d(fc7,filters = 1113,kernel_size = 1,activation = tf.nn.sigmoid,
                               bias_initializer=tf.constant_initializer(-6.8),
                               kernel_initializer=tf.truncated_normal_initializer(stddev = 0.001))#(c,5,5,1113)

            fc8 = tf.reshape(fc8,[-1,36,1113]) #(c,25,1113)
            mil_1 = 1 - fc8 #(c,25,1113)
            max_prob = tf.reduce_max(fc8, axis = 1) #(c,1113)
            prob = 1 - tf.reduce_prod(mil_1,[1])
            final_prob = tf.maximum(max_prob,prob)
            leak_prob = tf.nn.sigmoid(attention_logit)
            max_prob2 = tf.maximum(tf.reduce_max(fc8, axis = 1),leak_prob)
            prob2 = 1 - tf.multiply(tf.reduce_prod(mil_1,[1]),1-leak_prob) #(c,1113)
            final_prob2 = tf.maximum(max_prob2,prob2)

        
        return final_prob,final_prob2

    def _attribute_gate_layer(self,beta,c,z,h,x,prob,reuse=False):
        with tf.variable_scope('attribute_attention_layer',reuse=reuse):
            w_z = tf.get_variable('w_z', [self.M,2048 ], initializer=tf.contrib.layers.xavier_initializer())
            b_z = tf.get_variable('b_z', [2048], initializer=tf.constant_initializer(0.0))
            w_z3 = tf.get_variable('w_z3', [2048, 1113], initializer=tf.contrib.layers.xavier_initializer())
            b_z3 = tf.get_variable('b_z3', [1113], initializer=tf.constant_initializer(0.0))
            w_h = tf.get_variable('w_h', [self.H, 2048], initializer=tf.contrib.layers.xavier_initializer())
            logit_z = tf.nn.relu(tf.matmul(x,w_z) + tf.matmul(h,w_h)+b_z)
            logit_z = tf.matmul(logit_z,w_z3)+b_z3#(N,H)
            attention = tf.multiply(tf.nn.sigmoid(logit_z),prob)
        return attention

    def _attribute_vector_op(self,batch_size,attention_logits,attribute_embedding_expand):   
        values_top10,_ = tf.nn.top_k(attention_logits,10) #(N,10)
        values_10 = tf.gather(values_top10,9,axis = 1) #(N,1)
        attention_mask = tf.to_float(tf.greater_equal(attention_logits,tf.expand_dims(values_10,1)))
        attribute_attention = attention_logits*attention_mask/tf.reduce_sum(attention_logits*attention_mask,axis = 1,keep_dims = True)#(N,1113)
        attribute_vector = tf.multiply(tf.expand_dims(attribute_attention,2),tf.tile(attribute_embedding_expand,[batch_size,1,1])) #(N,1113,M)
        attribute_vector = tf.reduce_sum(attribute_vector,1) #(N,M)
        return attribute_vector,attribute_attention

    def decision(self,c,h,reuse = False):
        with tf.variable_scope('decision_layer',reuse = reuse):
            w_c = tf.get_variable('w_c', [self.M, 1024], initializer=self.weight_initializer)
            b_z = tf.get_variable('b_z',[1024],initializer=tf.constant_initializer(0.0))
            w = tf.get_variable('w', [1024,1], initializer=self.weight_initializer)
            b = tf.get_variable('b',[1],initializer=tf.constant_initializer(0.0))
            logit = tf.nn.relu(tf.matmul(c+h,w_c)+b_z)
            logit = tf.matmul(logit,w)+b
        return logit

    def init_human_attributes(self,features):
        human_attributes_im,_ = self.WILDER_attention_map(features)
        human_attributes_logit = self.WILDER_attention(human_attributes_im,reuse=False)
        _,human_attributes_prob = self.WILDER_Noisy_or(features,human_attributes_logit,is_training=False,reuse=False) #(N,14)
        short_prob = tf.expand_dims(tf.gather(human_attributes_prob,indices=0,axis = 1),1)
        long_prob = tf.expand_dims(tf.gather(human_attributes_prob,indices=1,axis = 1),1)
        plaid_prob = tf.expand_dims(tf.gather(human_attributes_prob,indices=2,axis = 1),1)
        striped_prob = tf.expand_dims(tf.gather(human_attributes_prob,indices=3,axis = 1),1)
        human_attributes_prob_extended = tf.concat([short_prob,long_prob,tf.maximum(short_prob,long_prob),
                                                plaid_prob,striped_prob,tf.maximum(plaid_prob,striped_prob)],axis=1)
        self.attribute_threshold_greater = [0.5,0.5,0.5,0.5,0.5,0.5]
        self.attribute_threshold_smaller = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.greater_attributes_list = [373,148,1701,1124,810,278] #short long hair plaid striped shirt
        self.smaller_attributes_list =  [0,0,0,0,0,0]
        return human_attributes_prob_extended

    def human_attributes_op(self,x,human_attributes_prob,batch_size):        
        mask_greater =tf.to_int32(tf.greater_equal(human_attributes_prob,tf.tile([self.attribute_threshold_greater],[batch_size,1]))) #(N,14)
        mask_smaller = tf.to_int32(tf.greater_equal(tf.tile([self.attribute_threshold_smaller],[batch_size,1]),human_attributes_prob)) #(N,14)
        greater = self.greater_attributes_list * mask_greater
        smaller = self.smaller_attributes_list * mask_smaller
        human_attributes_list = greater + smaller
        used_mask = tf.to_int32(tf.equal(tf.expand_dims(x,1),human_attributes_list))
        for used_word in self.used_list:
            used_mask = used_mask+tf.to_int32(tf.equal(tf.expand_dims(used_word,1),human_attributes_list))
        used_mask = tf.to_int32(tf.equal(used_mask,0))
        self.used_list.append(x)
        #import ipdb; ipdb.set_trace()
        human_attributes_list = human_attributes_list*used_mask
        empty_flag = tf.to_float(tf.not_equal(tf.reduce_sum(human_attributes_list,1,keep_dims=True),0))

        human_attributes_embedding = self._word_embedding(human_attributes_list,reuse = True)#(N,6,M) 
        human_attributes_embedding = tf.multiply(human_attributes_embedding,tf.cast(tf.expand_dims(
                                          used_mask*(mask_smaller+mask_greater),2),tf.float32)) #(N,14,M)
        human_attributes_mean = tf.reduce_sum(human_attributes_embedding,1)/tf.clip_by_value(tf.reduce_sum(
                                                            tf.cast(used_mask*(mask_greater+mask_smaller),tf.float32),keep_dims=True,axis=1),1.0,15.0) #(N,M)
        return human_attributes_mean,human_attributes_list,empty_flag

    def build_sampler(self,max_len=20):
        image = tf.image.convert_image_dtype(self.image, dtype=tf.float32)
        image_val = image_processing.process_image(image,is_training=False, height = self.image_size, width =self.image_size )
        network_fn = nets_factory.get_network_fn(
                      self.CNN_MODEL,
                      num_classes=1001,
                      weight_decay=0.0001,
                      is_training=False)
        logits,end_points = network_fn(tf.expand_dims(image_val,0))
        features = end_points['Mixed_7c']

        batch_size = tf.shape(features)[0]
        # batch normalize feature vectors
        with tf.variable_scope('image_features'): 
            features_orig = slim.batch_norm(features, is_training = False)

        features_orig = tf.reshape(features_orig,[-1,self.K,2048])   
        features_proj,features_global = self._image_features(features_orig)
        c, h = self._get_initial_lstm(features=features_proj)

        z,alpha = self._static_attribute_attention_layer(features)

        attribute_embedding = tf.expand_dims(self._word_embedding(self.attribute_list,reuse = False),0) #(1,1113,M)
        MIL_attention = self._MIL_attention(1.0,0.0,z,inital = True,reuse = False)
        _,prob2 = self._MIL_Leaky_noisy_or(features,MIL_attention,is_training=False,reuse=False)
        attribute_vector,attribute_attention = self._attribute_vector_op(batch_size,prob2,attribute_embedding)
        
        sampled_word_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)


        with tf.variable_scope('lstm', reuse=False):
            _, (c, h) = lstm_cell(inputs=tf.concat([features_global,attribute_vector],axis=1), state=[c, h])
        

        flag1=1.0
        human_attributes_prob_extended = self.init_human_attributes(features)
        self.used_list = []
        for t in range(max_len):
            if t == 0:
                sampled_word = tf.fill([tf.shape(features)[0]], self._start)
            x = self._word_embedding(inputs=sampled_word, reuse=True)     
            x_new = tf.concat([x,attribute_vector],1)

            with tf.variable_scope('lstm', reuse=True):
                h_old = h
                c_old = c
                _, (c, h) = lstm_cell(inputs=x_new, state=[c, h]) #(N,512)
            vs = self._visual_sentinel(h_old,c,x_new,reuse=(t!=0))
            
            c_context,c_hat_1, alpha,beta_1 = self._attention_layer(features_proj,features_orig ,h, vs,reuse=(t!=0))
            gamma = self.decision(c_hat_1,h,reuse = (t!=0)) #(N,1)
            trigger = tf.to_float(tf.greater_equal(tf.nn.sigmoid(gamma),0.3))

            MIL_attention = self._MIL_attention(0.0,c_context,z,inital = False,reuse=True)
            _,prob2 = self._MIL_Leaky_noisy_or(features,MIL_attention,is_training=False,reuse=True)
            attention_logits= self._attribute_gate_layer(beta_1,c_context,z,h,x,prob2,reuse=(t!=0))
            attribute_vector2,attribute_attention2 = self._attribute_vector_op(batch_size,attention_logits,attribute_embedding)
            
            human_attributes_mean,human_attributes_list,empty_flag =self.human_attributes_op(sampled_word,human_attributes_prob_extended,batch_size)
            hyperparameter = 0.9
            attribute_vector2 = (1-trigger*flag1*hyperparameter*empty_flag)*attribute_vector2 + trigger*flag1*hyperparameter*empty_flag*human_attributes_mean
            attribute_vector2 = (1-(1-trigger)*(1-flag1)*hyperparameter*empty_flag)*attribute_vector2 + (1-trigger)*(1-flag1)*hyperparameter*empty_flag*human_attributes_mean
            flag1 = 1-trigger

            x_new = tf.concat([x,attribute_vector2],1)

            with tf.variable_scope('lstm', reuse=True):
              _, (c, h) = lstm_cell(inputs=x_new, state=[c_old, h_old])
            vs = self._visual_sentinel(h_old,c,x_new,reuse=True)
            _,c_hat, alpha,beta = self._attention_layer2(features_proj, features_orig,h,alpha, vs,reuse=(t!=0))
            logits = self._decode_layer(h, c_hat ,stop_gradient = False,reuse=(t!=0))

            sampled_word = tf.argmax(logits, 1,output_type=tf.int32)
            sampled_word_list.append(sampled_word)
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
        return sampled_captions
