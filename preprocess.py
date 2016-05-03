#!/usr/bin/env python

"""NER Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs
import collections
import string
import joblib

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

# Your preprocessing, features construction, and word2vec code.
def line_to_words(line, dataset):
    # clean_line = clean_str(line.strip())
    # words = clean_line.split(' ')
    clean_line = line.strip()
    words = clean_line.split('\t')
    #words = words[1:]
    return words

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_vocab(datafile):
    max_sent_len = 0
    word_to_idx = {}
    word_count = {}
    
    # Start at 5 (1 is <s>, 2 is </s>)
    word_to_idx['*blank*'] = 1
    word_to_idx['<unk>'] = 2
    word_to_idx['<s>'] = 3
    word_to_idx['</s>'] = 4
    # word_to_idx['RARE'] = 3
    # word_to_idx['NUMBER'] = 4

    idx = 5
    
    with open(datafile) as f:
        data = json.load(f)
        
    all_images = data['images']
    
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
                # elif word in ['<s>', '<s>', 'RARE', 'NUMBER']:
                #     if word in word_count:
                #         word_count[word] += 1
                #     else:
                #         word_count[word] = 1
    
    return max_sent_len, word_to_idx



import json
def get_sentences(datafile, word_to_idx, max_sent_len, splits = [6000,1000,1000], num_sentences_per = 5):
    training_output_start = np.zeros((splits[0], num_sentences_per, max_sent_len))+word_to_idx['*blank*']
    dev_output_start = np.zeros((splits[1], num_sentences_per, max_sent_len))+word_to_idx['*blank*']
    test_output_start = np.zeros((splits[2], num_sentences_per, max_sent_len))+word_to_idx['*blank*']
    
    training_output_end = np.zeros((splits[0], num_sentences_per, max_sent_len))+word_to_idx['*blank*']
    dev_output_end = np.zeros((splits[1], num_sentences_per, max_sent_len))+word_to_idx['*blank*']
    test_output_end = np.zeros((splits[2], num_sentences_per, max_sent_len))+word_to_idx['*blank*']
    
    training_num_words = np.zeros((splits[0], num_sentences_per))
    dev_num_words = np.zeros((splits[1], num_sentences_per))
    test_num_words = np.zeros((splits[2], num_sentences_per))
    
    
    with open(datafile) as f:
        data = json.load(f)
        
    all_images = data['images']
    
    # print len(all_images)
    # assert False
    
    image_counter = 0
       
    for i in range(splits[0]):
        current_image = all_images[image_counter]
        
        sentences = current_image['sentences']
        
        for j, sentence in enumerate(sentences):
            training_num_words[i][j] = len(sentence['tokens'])
            
            training_output_start[i][j][0] = word_to_idx['<s>']
            for k, word in enumerate(sentence['tokens']):
                training_output_start[i][j][k+1] = word_to_idx[word]
                training_output_end[i][j][k] = word_to_idx[word]
            training_output_end[i][j][k+1] = word_to_idx['</s>']

        image_counter += 1
        
    for i in range(splits[1]):
        # print image_counter
        current_image = all_images[image_counter]
        
        sentences = current_image['sentences']
        
        for j, sentence in enumerate(sentences):
            dev_num_words[i][j] = len(sentence['tokens'])
            
            dev_output_start[i][j][0] = word_to_idx['<s>']
            for k, word in enumerate(sentence['tokens']):
                dev_output_start[i][j][k+1] = word_to_idx[word]
                dev_output_end[i][j][k] = word_to_idx[word]
            dev_output_end[i][j][k+1] = word_to_idx['</s>']

        image_counter += 1
        
    for i in range(splits[2]):
        current_image = all_images[image_counter]
        
        sentences = current_image['sentences']
        
        for j, sentence in enumerate(sentences):
            test_num_words[i][j] = len(sentence['tokens'])
            
            test_output_start[i][j][0] = word_to_idx['<s>']
            for k, word in enumerate(sentence['tokens']):
                test_output_start[i][j][k+1] = word_to_idx[word]
                test_output_end[i][j][k] = word_to_idx[word]
            test_output_end[i][j][k+1] = word_to_idx['</s>']

                
        image_counter += 1
        
    return training_output_start, training_output_end, dev_output_start, dev_output_end, test_output_start, test_output_end, training_num_words, dev_num_words, test_num_words
    
    
# def batchify(batchsize, train_output, valid_output, test_output):
#     train_output = np.reshape(-1, batchsize, train_output.shape[1], train_output.shape[2])
#     valid_output = np.reshape(-1, batchsize, valid_output.shape[1], valid_output.shape[2])
#     test_output = np.reshape(-1, batchsize, test_output.shape[1], test_output.shape[2])
    
#     return train_output, valid_output, test_output
    
def batchify(batchsize, data):
    data = data.reshape(-1, batchsize, data.shape[1], data.shape[2])
    return data
    
    
    
    

def write(conv_features, batchsize, max_sent_len, vocab_size, num_sentences_per, output_start, output_end, num_words, filename='flickr8k_train.hdf5'):
    
    with h5py.File(filename, "w") as f:
        #### training ####
        #conv_features = joblib.load('preprocessors/train/train_conv.pkl')
        
        # how many features in conv source
        # calculated as last dimension of conv features
        f['source_num_features'] = np.array([conv_features.shape[-1]], dtype=np.int32)
        
        # vocab size
        # calculated as len of word_to_idx dictionary
        f['target_num_vocabs'] = np.array([vocab_size], dtype=np.int32) 
        
        f['input_conv'] = conv_features
        for i in range(num_sentences_per):
            f['target_input_%i' % (i)] = output_start[:,:,i,:]
            f['target_output_%i' % (i)] = output_end[:,:,i,:]
            f['total_words_%i' % (i)] = num_words[:,:,i]
        
        # how many samples
        # calculated as (#batches x batchsize)
        f['num_samples'] = np.array([output_start.shape[0]*output_start.shape[1]], dtype=np.int32)
        
        # how many batches
        # calculated as (#batches)
        f['num_batches'] = np.array([output_start.shape[0]], dtype=np.int32)
        
        
        # maximum sentence len
        # calcuated as max_sent_len returned from get_vocab() function
        f['max_target_sent_l'] = np.array([max_sent_len], dtype=np.int32)
        
        # source sentence len
        # calculated as second dimension of conv features
        f['max_source_sent_l'] = np.array([conv_features.shape[1]], dtype=np.int32)
        
        # max batch len
        # calculated as batchsize
        f['max_batch_l'] = np.array([output_start.shape[1]], dtype=np.int32)

        f['source_l'] = np.array([conv_features.shape[1]]*output_start.shape[0], dtype=np.int32)#: train_num_batches (196,196,...)
        
        f['target_l'] = np.array([max_sent_len]*output_start.shape[0], dtype=np.int32)#train_num_batches (37,37,...)

        f['batch_l'] = np.array([output_start.shape[1]]*output_start.shape[0], dtype=np.int32)#train_num_batches (32,32,...,10)   
         
        f['source_input'] = conv_features.reshape(-1, batchsize, conv_features.shape[1], conv_features.shape[2])
        
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-batchsize', help="Batch size",
                        type=int)
    args = parser.parse_args(arguments)
    batchsize = args.batchsize or 100
    
    num_sentences_per = 5
    
    datafile = 'data/flickr8k/dataset.json'
    
    
    # get max_sent_len and word_to_idx dictionary
    max_sent_len, word_to_idx = get_vocab(datafile)
    max_sent_len += 1
    
    # get vocab size
    vocab_size = len(word_to_idx)
    print('Vocab size:', vocab_size)

    # get core sentence data
    train_output_start, train_output_end, valid_output_start, valid_output_end, test_output_start, test_output_end, train_num_words, valid_num_words, test_num_words = get_sentences(datafile, word_to_idx, max_sent_len, splits = [6000,1000,1000], num_sentences_per = 5)
    
    
    # train_output, valid_output, test_output = batchify(batchsize, train_output, valid_output, test_output)
    
    train_output_start = batchify(batchsize, train_output_start) 
    train_output_end = batchify(batchsize, train_output_end)
    valid_output_start = batchify(batchsize, valid_output_start) 
    valid_output_end = batchify(batchsize, valid_output_end) 
    test_output_start = batchify(batchsize, test_output_start) 
    test_output_end = batchify(batchsize, test_output_end)
    
    train_num_words = train_num_words.reshape(-1, batchsize, train_num_words.shape[1])
    valid_num_words = valid_num_words.reshape(-1, batchsize, valid_num_words.shape[1])
    test_num_words = test_num_words.reshape(-1, batchsize, test_num_words.shape[1])

    conv_features = joblib.load('preprocessors/train/train_conv.pkl')
    write(conv_features, batchsize, max_sent_len, vocab_size, num_sentences_per, train_output_start, train_output_end, train_num_words, filename='mydata/flickr8k_train.hdf5')
       
    conv_features = joblib.load('preprocessors/dev/dev_conv.pkl')
    write(conv_features, batchsize, max_sent_len, vocab_size, num_sentences_per, valid_output_start, valid_output_end, valid_num_words, filename='mydata/flickr8k_valid.hdf5')
    
    conv_features = joblib.load('preprocessors/test/test_conv.pkl')
    write(conv_features, batchsize, max_sent_len, vocab_size, num_sentences_per, test_output_start, test_output_end, test_num_words, filename='mydata/flickr8k_test.hdf5')
        
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
