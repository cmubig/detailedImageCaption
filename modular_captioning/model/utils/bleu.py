#import cPickle as pickle
import pickle
import os
import sys
sys.path.append('../coco-caption')
from pycocoevalcap_py3.bleu.bleu import Bleu
from pycocoevalcap_py3.rouge.rouge import Rouge
from pycocoevalcap_py3.cider.cider import Cider
from pycocoevalcap_py3.meteor.meteor import Meteor
from pycocoevalcap_py3.spice.spice import Spice

def score(ref, hypo):
    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(),"ROUGE_L"),
        (Cider(),"CIDEr"),
        (Spice(), "SPICE")
    ]
    final_scores = {}
    for scorer,method in scorers:
        score,scores = scorer.compute_score(ref,hypo)
        #import ipdb; ipdb.set_trace()
        if type(score)==list:
            for m,s in zip(method,score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores,scores
    

def evaluate(data_path='./data', split='val',eval_file ='candidate.captions.pkl' ,get_scores=False):
    reference_path = os.path.join(data_path, "%s/%s.references.pkl" %(split, split))
    candidate_path = os.path.join(data_path, "%s/%s." %(split, split)+eval_file)
    
    # load caption data
    with open(reference_path, 'rb') as f:
        ref = pickle.load(f)
    with open(candidate_path, 'rb') as f:
        cand = pickle.load(f)
    # make dictionary
    hypo = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]
    
    # compute bleu score
    final_scores,spice_scores = score(ref, hypo)
    
    #with open('./model_v16_ref.txt','w') as f: 
    #    for k, v in ref.items():
    #        f.write(str(k) + ' >>> '+ str(v) + '\n\n')
    #with open('./model_v16_baseline.txt','w') as f: 
    #    for k, v in hypo.items():
    #        f.write(str(k) + ' >>> '+ str(v) + '\n\n')   
    #import ipdb;ipdb.set_trace() 
    # print out scores
    print ('Bleu_1:\t',final_scores['Bleu_1'])  
    print ('Bleu_2:\t',final_scores['Bleu_2']) 
    print ('Bleu_3:\t',final_scores['Bleu_3']) 
    print ('Bleu_4:\t',final_scores['Bleu_4']) 
    print ('METEOR:\t',final_scores['METEOR'])  
    print ('ROUGE_L:',final_scores['ROUGE_L'])  
    print ('CIDEr:\t',final_scores['CIDEr'])
    print ('SPICE:\t',final_scores['SPICE'])
    #import ipdb;ipdb.set_trace()
    if get_scores:
        return final_scores,spice_scores
    
   
    
    
    
    
    
    
    
    
    
    


