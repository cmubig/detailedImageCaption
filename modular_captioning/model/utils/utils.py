import numpy as np
import pickle
import time
import os


def decode_captions(captions, idx_to_word):
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape

    decoded = []
    for i in range(N):
        words = []
        for t in range(T):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            if word == '<END>':
                words.append('.')
                break
            if word != '<NULL>':
                words.append(word)
        decoded.append(' '.join(words))
    return decoded



def write_bleu(scores,spice_scores,path, epoch,lr):
    if epoch == 1:
        file_mode = 'w'
    else:
        file_mode = 'a'
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(path,'val.bleu.scores.txt'), file_mode) as f:
        f.write('Epoch %d\n' %(epoch))
        f.write('Learning rate: %f\n'%lr)
        f.write('Bleu_1: %f\n' %scores['Bleu_1'])
        f.write('Bleu_2: %f\n' %scores['Bleu_2'])
        f.write('Bleu_3: %f\n' %scores['Bleu_3'])  
        f.write('Bleu_4: %f\n' %scores['Bleu_4']) 
        f.write('ROUGE_L: %f\n' %scores['ROUGE_L'])  
        f.write('METEOR: %f\n' %scores['METEOR'])
        f.write('CIDEr: %f\n' %scores['CIDEr'])
        f.write('SPICE: %f\n' %scores['SPICE'])
        f.write('obj: %.3f, attri: %.3f, rela: %.3f, col: %.3f, cnt: %.3f, size: %.3f\n\n'
                             %(spice_scores[0]*100,spice_scores[1]*100,spice_scores[2]*100,
                             spice_scores[3]*100,spice_scores[4]*100, spice_scores[5]*100))
def load_pickle(path,p=True):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        if p:
            print ('Loaded %s..' %path)
        return file 

def save_pickle(data, path):
    with open(path, 'wb') as f:
        #pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(data, f)
        print ('Saved %s..' %path)