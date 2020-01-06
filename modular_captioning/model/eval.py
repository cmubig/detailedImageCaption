
from utils.utils import *
import numpy as np
from config import finalize_configs, config as cfg
from collections import namedtuple


DetectionResult = namedtuple(
    'DetectionResult',
    ['box', 'score', 'class_id', 'mask'])


def eval_coco(df,predict_func):
    df.reset_state()
    all_gen_cap = []
    all_alpha_module = []
    #all_beta_module  = []
    for data in df:
        gen_cap,alpha_module = predict_func(data)
        #gen_cap,alpha_module,beta_module = predict_func(data)
        all_gen_cap.extend(gen_cap)
        all_alpha_module.extend(alpha_module)
        #all_beta_module.extend(beta_module)
    return np.array(all_gen_cap),np.array(all_alpha_module)#,np.array(all_beta_module)
       
def predict_batch(data,model_func):
    mean_feature,region_feature,box,histogram = data
    captions = model_func(mean_feature,region_feature,box,histogram)
    #mean_feature,region_feature = data
    #captions = model_func(mean_feature,region_feature)
    return captions
'''
def visualization(self,e,file_name,image,decoded,gammas,betas,alphas):
                alphas: [1,T+1,R]
        box:[1,R]
                # Plot original image
        save_dir = cfg.EXAMPLE_DIR + str(e) + '/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        alpha_list = np.array(alphas)#[1,T+1,k]
        words = decoded.split(" ")
        fig = plt.figure(i,dpi = 500)
        ax = fig.add_subplot(19, 4, 1)
        ax.imshow(image, aspect='equal',extent = (-8,8,-8,8))
        ax.text(8.5, 0,file_name.split('/')[5].split('.')[0], color='black', fontsize=3)
        ax.axis('off')
        ax = fig.add_subplot(11, 4, 2)
        ax.imshow(image, aspect='equal',extent = (-8,8,-8,8))
        alp_curr = alpha_list[i][0,0].reshape(8,8)
        alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=37.375,sigma=20)
        ax.imshow(alp_img,alpha = 0.5, aspect='equal', extent = (-8,8,-8,8))
        ax.axis('off')
        ax = fig.add_subplot(11,4,3)
        for j,predict in enumerate(predictions[0].split(' ')):
            ax.text(0,(0.2*(4-j)),'%s(%.2f)'%(predict,values[0,0,j]),color='black', fontsize=3)
        ax.axis('off')

        # Plot images with first attention weights
        for t in range(len(words)): 
            if t > 18:
                break
            ax = fig.add_subplot(11, 4, 6+2*t-1)
            ax.text(8.5, 0, '%s(%.2f,%.2f)'%(words[t],1-beta_list[i][0,t],gammas[0,t]) , color='black', fontsize=3)
            ax.imshow(image, aspect='equal',extent = (-8,8,-8,8))
            alp_curr = alpha_list[i][0,t+1].reshape(8,8)
            alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=37.375,sigma=20)
            ax.imshow(alp_img,alpha = 0.5, aspect='equal',extent = (-8,8,-8,8))
            ax.axis('off')
            ax = fig.add_subplot(11, 4, 6+2*t)
            for j,predict in enumerate(predictions[t+1].split(' ')):
                ax.text(0,(0.2*(4-j)),'%s(%.2f)'%(predict,values[0,t+1,j]),color='black', fontsize=3)
            ax.axis('off')
        #import ipdb;ipdb.set_trace()
        save_path = save_dir + file_name.split('/')[5].split('.')[0] + mode+'.jpg'
        plt.savefig(save_path) 
        print('Example saved at ' + save_path)   
        plt.close('all')
        '''