# seq2seq-show-att-tell
We present Sequenced Show, Attend, and Tell: Natural Language from Natural Images, a machine translation-inspired framework to perform automatic captioning from images. Given an input image of a scene, our algorithm outputs a fully formed sentence explaining the contents and actions of the scene. Our method uses an LSTM-based sequence-to-sequence algorithm with global attention for generating the captions. The input to our algorithm is a set of convolution features extracted from the lower layers of a convolutional neural network, each corresponding to a particular portion of the input image. Following this, using a global attention model, the inputs are used to generate the caption one word at a time with the LSTM “focusing” on a portion of the image as dictated by the attention model.

We compare our proposed method with a number of different methods, including the attention- based method of Xu et al. (2015) as well as the attention-less method of Vinyals et al. (2015). Addi- tionally, we present results both with and without the use of pretrained word embeddings, with the use of different CNNs for feature extraction, the use of reverse ordering of the source input into the LSTM, and the use of residual connections. We find that our proposed method is com- parable with the state of the art. Further, we find that the use of pretrained word embeddings, different CNNs, reversing the ordering of the input, and the use of residual connections do not have a large impact on system performance.

A full example gallery can be seen at: <a href="http://steerapi.github.io/seq2seq-show-att-tell/flickr8k/pages/index.html">http://steerapi.github.io/seq2seq-show-att-tell/flickr8k/pages/index.html</a>

Preprocessed features for the Flickr-8K, Flickr-30K, and Microsoft COCO datasets can be found at <a href="https://drive.google.com/folderview?id=0Byyuc5LmNmJPQmJzVE5GOEJOdzQ&usp=sharing">https://drive.google.com/folderview?id=0Byyuc5LmNmJPQmJzVE5GOEJOdzQ&usp=sharing</a>.  This directory contains features extracted for these datasets using the VGG-16 Convolutional Neural Network (CNN).   The features are extracted from the last convolutional layer before the fully connected layers, where each image generate 196 features (14x14 unrolled) of length 512.  The provided preprocessing script (see below) automatically extracts the convolutional features and other relevant data from these files.  However, if you want to access portions of the data manually, you can do so through the following:

To read out the reference sentences in JSON format for each dataset, in Python execute:
```
f = h5py.File(<file>, "r")
sentences = f.attrs['sents']
```

To access the convolutional features for each dataset, in Python execute:
```
f = h5py.File(<file>, "r")
convolution_features = np.array(f['feats_conv'])
```

If you use this model or codebase, please cite:

    @misc{seqshowatttell,
      author = {Comiter, Marcus and Teerapittayanon, Surat},
      title = {Sequenced Show, Attend, and Tell: Natural Language from Natural Images},
      year = {2016},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {https://github.com/steerapi/seq2seq-show-att-tell/}
    }
    
### Dependencies

#### Python Dependencies
* h5py (for saving data)
* numpy

#### Lua Dependencies
You will need the following packages:
* hdf5 (for saving/reading data)
* nngraph

If you use a GPU for running the model, you will also need:
* cutorch
* cunn


### Instructions for Using the Codebase

#### Preprocessing and Data Acquisition
First download the features extracted from the convolutional neural network for each of the datasets.  Once these features have been downloaded, run the preprocess script by executing the following in the code/ directory:

```
bash scripts/preprocess.sh
```
This will generate hdf5 data files for each of the datasets for use with the algorithm.  If you only wish to generate data files for a subset of the available datasets, comment out the appropriate lines of scripts/preprocess.sh

#### Training the Model

Following the generation of the hdf5 files, you can train the model by executing the following in the code/ directory:

```
bash scripts/run_seq2seq_att.sh
```

By default, you must have both a training hdf5 file located at 'mydata/flickr8k_train.hdf5' and a validation hdf5 file located at 'mydata/flickr8k_train.hdf5'.  If you wish to use other training and validation files, you can pass this in as an option by altering the scripts/run_seq2seq_att.sh script, or by running the Lua file project.lua directly and passing in the following options:
```
-trainfile <training file from output of preprocessing> -validfile <validation file from output of preproessing>
```

#### Using the Model

Once the model has been trained, it can be used to generate captions for the test portion of the dataset by executing the following in the code/ directory:

```
bash scripts/test_seq2seq_att.sh
```

By default, you must have an ``index to word'' file located at 'mydata/idx_to_word.txt'.  If you wish to use another index to word file, you can pass this in as an option by altering the scripts/test_seq2seq_att.sh script, or by running the Lua file project.lua directly and passing in the following option:
```
-targ_dict' <index to word file>
```

#### Scoring the Output of the Model
Once the model has been used to generate captions, the resulting captions can be scored by executing the following in the code/directory:

```
bash scripts/score.sh <resulting file from running scripts/test_seq2seq_att.sh>
```

This will score the results in terms of BLEU score (specifically, BLEU-1, BLEU-2, BLEU-3, and BLEU-4).

#### Acknowledgments
Our Sequenced Show, Attend, and Tell implementation utilizes code from the following:
* [Yoon Kim's s Sequence-to-Sequence Learning with Attentional Neural Networks](https://github.com/harvardnlp/seq2seq-attn)
* [Andrej Karpathy's char-rnn repo](https://github.com/karpathy/char-rnn)
* [Wojciech Zaremba's lstm repo](https://github.com/wojzaremba/lstm)
* [Element rnn library](https://github.com/Element-Research/rnn)
