# seq2seq-show-att-tell
We present Sequenced Show, Attend, and Tell: Natural Language from Natural Images, a machine translation-inspired framework to perform automatic captioning from images. Given an input image of a scene, our algorithm outputs a fully formed sentence explaining the contents and actions of the scene. Our method uses an LSTM-based sequence-to-sequence algorithm with global attention for generating the captions. The input to our algorithm is a set of convolution features extracted from the lower layers of a convolutional neural network, each corresponding to a particular portion of the input image. Following this, using a global attention model, the inputs are used to generate the caption one word at a time with the LSTM “focusing” on a portion of the image as dictated by the attention model.

We compare our proposed method with a number of different methods, including the attention- based method of Xu et al. (2015) as well as the attention-less method of Vinyals et al. (2015). Addi- tionally, we present results both with and without the use of pretrained word embeddings, with the use of different CNNs for feature extraction, the use of reverse ordering of the source input into the LSTM, and the use of residual connections. We find that our proposed method is com- parable with the state of the art. Further, we find that the use of pretrained word embeddings, different CNNs, reversing the ordering of the input, and the use of residual connections do not have a large impact on system performance.

A full example gallery can be seen at: <a href="http://steerapi.github.io/seq2seq-show-att-tell/flickr8k/pages/index.html">http://steerapi.github.io/seq2seq-show-att-tell/flickr8k/pages/index.html</a>

Preprocessed features for the Flickr-8K, Flickr-30K, and Microsoft COCO datasets can be found at <a href="https://drive.google.com/folderview?id=0Byyuc5LmNmJPQmJzVE5GOEJOdzQ&usp=sharing">https://drive.google.com/folderview?id=0Byyuc5LmNmJPQmJzVE5GOEJOdzQ&usp=sharing</a>


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

GPU usage will additionally require:
* cutorch
* cunn


#### Acknowledgments
Our implementation utilizes code from the following:
* [Yoon Kim's s Sequence-to-Sequence Learning with Attentional Neural Networks](https://github.com/harvardnlp/seq2seq-attn)
* [Andrej Karpathy's char-rnn repo](https://github.com/karpathy/char-rnn)
* [Wojciech Zaremba's lstm repo](https://github.com/wojzaremba/lstm)
* [Element rnn library](https://github.com/Element-Research/rnn)
