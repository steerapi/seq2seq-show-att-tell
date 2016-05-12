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


def get_vocab(jsondata):
    max_sent_len = 0
    word_to_idx = {}
    idx_to_word = {}
    word_count = {}
    
    # Start at 5 (1 is <s>, 2 is </s>)
    word_to_idx['*blank*'] = 1
    word_to_idx['<unk>'] = 2
    word_to_idx['<s>'] = 3
    word_to_idx['</s>'] = 4
    # word_to_idx['RARE'] = 3
    # word_to_idx['NUMBER'] = 4
    
    idx_to_word[1] = '*blank*'
    idx_to_word[2] = '<unk>'
    idx_to_word[3] = '<s>'
    idx_to_word[4] = '</s>'

    idx = 5
    
    #with open(datafile) as f:
    #    data = json.load(f)
        
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
                    idx_to_word[idx] = word
                    idx += 1                            
                    word_count[word.lower()] = 1
                # elif word in ['<s>', '<s>', 'RARE', 'NUMBER']:
                #     if word in word_count:
                #         word_count[word] += 1
                #     else:
                #         word_count[word] = 1
    
    return max_sent_len, word_to_idx, idx_to_word



import json
def get_sentences(jsondata, word_to_idx, max_sent_len, num_sentences_per = 5):
    numm = len(jsondata)
    
    training_output_start = np.zeros((numm, num_sentences_per, max_sent_len))+word_to_idx['*blank*']
    
    training_output_end = np.zeros((numm, num_sentences_per, max_sent_len))+word_to_idx['*blank*']
    
    training_num_words = np.zeros((numm, num_sentences_per))
    
    
    #with open(datafile) as f:
    #    data = json.load(f)
        
    all_images = jsondata
    
    image_counter = 0
       
    for i, current_image in enumerate(all_images):
        
        sentences = current_image['sentences']
        
        for j, sentence in enumerate(sentences[:5]):
            training_num_words[i][j] = len(sentence['tokens'])
            
            training_output_start[i][j][0] = word_to_idx['<s>']
            for k, word in enumerate(sentence['tokens']):
                training_output_start[i][j][k+1] = word_to_idx[word]
                training_output_end[i][j][k] = word_to_idx[word]
            training_output_end[i][j][k+1] = word_to_idx['</s>']

        image_counter += 1
        
    return training_output_start, training_output_end, training_num_words
    
    
# def batchify(batchsize, train_output, valid_output, test_output):
#     train_output = np.reshape(-1, batchsize, train_output.shape[1], train_output.shape[2])
#     valid_output = np.reshape(-1, batchsize, valid_output.shape[1], valid_output.shape[2])
#     test_output = np.reshape(-1, batchsize, test_output.shape[1], test_output.shape[2])
    
#     return train_output, valid_output, test_output
    
def batchify(batchsize, data):
    print "SHAPE: ", data.shape
    #assert False
    
    numover = data.shape[0] % batchsize
    padding = np.zeros((batchsize - numover, data.shape[1], data.shape[2]))
    data = np.vstack((data, padding))
    
    data = data.reshape(-1, batchsize, data.shape[1], data.shape[2])
    return data, numover
    
    
    
    

def write(conv_features, batchsize, max_sent_len, vocab_size, num_sentences_per, output_start, output_end, num_words, numover, filename='flickr8k_train.hdf5'):
    
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
        print "NB: ", output_start.shape[0]
        
        
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

        if numover == 0:
            f['batch_l'] = np.array([output_start.shape[1]]*output_start.shape[0], dtype=np.int32)#train_num_batches (32,32,...,10)  
        else:
            f['batch_l'] = np.array([output_start.shape[1]]*(output_start.shape[0]-1)+[numover], dtype=np.int32)
            
        if numover == 0: 
            f['source_input'] = conv_features.reshape(-1, batchsize, conv_features.shape[1], conv_features.shape[2])
        else:
            padding = np.zeros((batchsize - numover, conv_features.shape[1], conv_features.shape[2]))
            print "F1: ", conv_features.shape
            conv_features = np.vstack((conv_features, padding))
            print "F2: ", conv_features.shape
            f['source_input'] = conv_features.reshape(-1, batchsize, conv_features.shape[1], conv_features.shape[2])
            print "F3: ", conv_features.reshape(-1, batchsize, conv_features.shape[1], conv_features.shape[2]).shape
            
        
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-batchsize', help="Batch size",
                        type=int)
    parser.add_argument('-traindatafile', help="Data file",
                        type=str)
    parser.add_argument('-validdatafile', help="Data file",
                        type=str)
    parser.add_argument('-testdatafile', help="Data file",
                        type=str)
    parser.add_argument('-trainsavefile', help="Save file",
                        type=str)
    parser.add_argument('-validsavefile', help="Save file",
                        type=str)
    parser.add_argument('-testsavefile', help="Save file",
                        type=str)
    args = parser.parse_args(arguments)
    batchsize = args.batchsize or 100
    traindatafile = args.traindatafile
    validdatafile = args.validdatafile
    testdatafile = args.testdatafile
    trainsavefile = args.trainsavefile
    validsavefile = args.validsavefile
    testsavefile = args.testsavefile
    
    num_sentences_per = 5
    
    f = h5py.File(traindatafile, "r")
    sents = f.attrs['sents']
    trainjsondata = json.loads(sents)
    train_conv_features = np.array(f['feats_conv'])
    
    f = h5py.File(validdatafile, "r")
    sents = f.attrs['sents']
    validjsondata = json.loads(sents)
    valid_conv_features = np.array(f['feats_conv'])
    
    f = h5py.File(testdatafile, "r")
    sents = f.attrs['sents']
    testjsondata = json.loads(sents)
    test_conv_features = np.array(f['feats_conv'])
    
    alljsondata = trainjsondata + validjsondata + testjsondata
    
    
    # get max_sent_len and word_to_idx dictionary
    max_sent_len, word_to_idx, idx_to_word  = get_vocab(alljsondata)
    max_sent_len += 1
    
    ###################################
    
    f = open('idx_to_word_coco.txt','w')
    
    idx_to_word_list = ['DONOTUSE']
    for i in range(len(word_to_idx)):
        idx_to_word_list.append(idx_to_word[i+1])
        f.write('%i %s\n' % (i+1, idx_to_word[i+1]))
    f.close()
    '''
    assert False
    idx_to_word_list = np.array(idx_to_word_list)
    
    filename = 'idx_to_word_flickr30k.hdf5'
    with h5py.File(filename, "w") as f:
        f['idx_to_word'] = np.array(idx_to_word_list)
    
    #assert False
    '''
    
    ###################################
    
    # get vocab size
    vocab_size = len(word_to_idx)
    print('Vocab size:', vocab_size)

    # get core sentence data
    train_output_start, train_output_end, train_num_words = get_sentences(trainjsondata, word_to_idx, max_sent_len, num_sentences_per = 5)
    
    valid_output_start, valid_output_end, valid_num_words = get_sentences(validjsondata, word_to_idx, max_sent_len, num_sentences_per = 5)
    
    test_output_start, test_output_end, test_num_words = get_sentences(testjsondata, word_to_idx, max_sent_len, num_sentences_per = 5)
    
    
    # train_output, valid_output, test_output = batchify(batchsize, train_output, valid_output, test_output)
    
    train_output_start, numover_train = batchify(batchsize, train_output_start) 
    train_output_end, _ = batchify(batchsize, train_output_end)
    valid_output_start, numover_valid = batchify(batchsize, valid_output_start) 
    valid_output_end, _ = batchify(batchsize, valid_output_end) 
    test_output_start, numover_test = batchify(batchsize, test_output_start) 
    test_output_end, _ = batchify(batchsize, test_output_end)
    
    padding = np.zeros((batchsize - numover_train, train_num_words.shape[1]))
    train_num_words = np.vstack((train_num_words, padding))
    
    padding = np.zeros((batchsize - numover_valid, valid_num_words.shape[1]))
    valid_num_words = np.vstack((valid_num_words, padding))
    
    padding = np.zeros((batchsize - numover_test, test_num_words.shape[1]))
    test_num_words = np.vstack((test_num_words, padding))
    
    train_num_words = train_num_words.reshape(-1, batchsize, train_num_words.shape[1])
    valid_num_words = valid_num_words.reshape(-1, batchsize, valid_num_words.shape[1])
    test_num_words = test_num_words.reshape(-1, batchsize, test_num_words.shape[1])

    write(train_conv_features, batchsize, max_sent_len, vocab_size, num_sentences_per, train_output_start, train_output_end, train_num_words, numover_train, filename=trainsavefile)
    
    write(valid_conv_features, batchsize, max_sent_len, vocab_size, num_sentences_per, valid_output_start, valid_output_end, valid_num_words, numover_valid, filename=validsavefile)
    
    write(test_conv_features, batchsize, max_sent_len, vocab_size, num_sentences_per, test_output_start, test_output_end, test_num_words, numover_test, filename=testsavefile)

        
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
