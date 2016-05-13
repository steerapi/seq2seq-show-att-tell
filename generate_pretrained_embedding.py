#!/usr/bin/env python

"""Glove word embedding preprocessor
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs
import collections
import string
import json


def get_vocab(jsondata):
    max_sent_len = 0
    word_to_idx = {}
    word_count = {}
    
    # Start at 5 (1 is <s>, 2 is </s>)
    word_to_idx['*blank*'] = 1
    word_to_idx['<unk>'] = 2
    word_to_idx['<s>'] = 3
    word_to_idx['</s>'] = 4
    
    idx = 5

    all_images = jsondata
    
    image_counter = 0
       
    for i in range(len(all_images)):
        current_image = all_images[i]
        
        sentences = current_image['sentences']
        
        for j, sentence in enumerate(sentences):
            max_sent_len = max(max_sent_len, len(sentence['tokens']))
            for k, word in enumerate(sentence['tokens']):
                word = word.lower()
                if word not in word_to_idx:
                    word_to_idx[word] = idx
                    idx += 1                            
                    word_count[word.lower()] = 1
    
    return max_sent_len, word_to_idx



def get_glove_weights(dataset, word_to_idx):
    weight_dict = {}
    
    # with codecs.open(dataset, "r", encoding="latin-1") as f:
    f = open(dataset, 'r')
    for line in f:
        line = line.strip()
        line = line.split(' ')
        line[1:] = [float(i) for i in line[1:]]
        weight_dict[line[0].lower()] = line[1:]
    
    weights = np.zeros((len(word_to_idx)+1, 50))
    weights[1] = list(np.random.uniform(low=-0.1, high=0.1, size=50))
    weights[2] = list(np.random.uniform(low=-0.1, high=0.1, size=50))
    weights[3] = list(np.random.uniform(low=-0.1, high=0.1, size=50))
    weights[4] = list(np.random.uniform(low=-0.1, high=0.1, size=50))
    
    insert_idx = 4
    
    for k in word_to_idx:
        if k in ['*blank*', '<unk>', '<s>', '</s>']:
            continue
        v = word_to_idx[k]
        if k in weight_dict:
            #print len(weight_dict[k])
            weights[v] = weight_dict[k]
        else:
            weights[v] = list(np.random.uniform(low=-0.1, high=0.1, size=50))
        insert_idx += 1
    
    return weights

def main(arguments):
    
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-traindatafile', help="Train data file",
                        type=str)
    parser.add_argument('-validdatafile', help="Validation data file",
                        type=str)
    parser.add_argument('-testdatafile', help="Test data file",
                        type=str)
    parser.add_argument('-pretrainedembeddingsfile', help="Pretrained word embeddings file",
                        type=str)
    parser.add_argument('-savefile', help="Save file",
                        type=str)
    
 
    args = parser.parse_args(arguments)
    traindatafile = args.traindatafile
    validdatafile = args.validdatafile
    testdatafile = args.testdatafile
    savefile = args.savefile
    pretrainedembeddingsfile = args.pretrainedembeddingsfile
    
    f = h5py.File(traindatafile, "r")
    sents = f.attrs['sents']
    trainjsondata = json.loads(sents)
    
    f = h5py.File(validdatafile, "r")
    sents = f.attrs['sents']
    validjsondata = json.loads(sents)
    
    f = h5py.File(testdatafile, "r")
    sents = f.attrs['sents']
    testjsondata = json.loads(sents)
    
    alljsondata = trainjsondata + validjsondata + testjsondata
    
    # get max_sent_len and word_to_idx dictionary
    max_sent_len, word_to_idx = get_vocab(alljsondata)
    max_sent_len += 1
    
    weights = get_glove_weights(pretrainedembeddingsfile, word_to_idx)

    filename = savefile
    with h5py.File(filename, "w") as f:
        f['weights'] = weights
    
        
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


