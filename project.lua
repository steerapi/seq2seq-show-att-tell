require("hdf5")
require("nn")
require("optim")

require("seq2seq_att")
-- require("hmm")
-- require("memm")
-- require("structure")

function run_fold(opt)
    local model

    -- Train.
    if opt.classifier == 'seq2seq_att' then
        model = seq2seq_att(opt)
    end

    -- Test. Output test file to evaluate BLUE score
    if opt.test==1 then
        model.test(model,opt)
    end

end

function run(opt)
    if opt.gpuid >= 0 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        if opt.gpuid2 >= 0 then
            print('using CUDA on second GPU ' .. opt.gpuid2 .. '...')
        end      
        require 'cutorch'
        require 'cunn'
        if opt.cudnn == 1 then
            print('loading cudnn...')
            require 'cudnn'
        end      
        cutorch.setDevice(opt.gpuid)
        cutorch.manualSeed(opt.seed)      
    end

    -- Reporting training speed, training set loss, training set predictive accuracy, and validation predictive accuracy
    run_fold(opt)

end

cmd = torch.CmdLine()

-- Cmd Args


cmd:option('-trainfile', 'mydata/flickr8k_train.hdf5', 'train data')
cmd:option('-validfile', 'mydata/flickr8k_valid.hdf5', 'valid data')
cmd:option('-testfile', 'mydata/flickr8k_test.hdf5', 'test data')

cmd:option('-outfile', 'out.txt', 'output file')
cmd:option('-lossfile', 'loss.png', 'file to write loss plot png')

cmd:option('-classifier', 'seq2seq_att', 'classifier to use')
cmd:option('-optimizer', 'adam', 'classifier to use [adam|sgd]')

cmd:option('-train', 1, 'run training code')
cmd:option('-test', 1, 'run test')

cmd:option('-word_vec_size', 50, 'words embedded dimension')
cmd:option('-hop_attn', 0, [[If > 0, then use a *hop attention* on this layer of the decoder. For example, if num_layers = 3 and `hop_attn = 2`, then the model will do an attention over the source sequence on the second layer (and use that as input to the third layer) and the penultimate layer]])
cmd:option('-res_net', 0, [[Use residual connections between LSTM stacks whereby the input to the l-th LSTM layer if the hidden state of the l-1-th LSTM layer  added with the l-2th LSTM layer. We didn't find this to help in our  experiments]])
cmd:option('-curriculum', 0, [[For this many epochs, order the minibatches based on source sequence length. Sometimes setting this to 1 will increase convergence speed.]])

cmd:option('-source_size', 512, [[Source vocab size]])
cmd:option('-target_size', 8387, [[Target vocab size]])
cmd:option('-rnn_size', 500, [[Size of LSTM hidden states]])
cmd:option('-num_layers', 2, [[Number of layers in the LSTM encoder/decoder]])

cmd:option('-savefile', 'seq2seq_att.t7', [[Savefile name (model will be saved as 
    savefile.t7 where X is the X-th epoch and PPL is 
    the validation perplexity]])
cmd:option('-train_from', 'seq2seq_att.t7', [[If training from a checkpoint then this is the path to the pretrained model.]])


-- GPU
cmd:option('-gpuid', 1, [[Which gpu to use >0, -1 = use CPU]])
cmd:option('-gpuid2', -1, [[If this is >= 0, then the model will use two GPUs whereby the encoder is on the first GPU and the decoder is on the second GPU. This will allow you to train with bigger batches/models.]])
cmd:option('-cudnn', 1, [[Whether to use cudnn or not for convolutions (for the character model). cudnn has much faster convolutions so this is highly recommended if using the character model]])

-- optimization
cmd:option('-epochs', 13, [[Number of training epochs]])

cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]])
cmd:option('-learning_rate', 1, [[Starting learning rate]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this, renormalize it to have the norm equal to max_grad_norm]])
cmd:option('-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]])
cmd:option('-lr_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-start_decay_at', 9, [[Start decay after this epoch]])

cmd:option('-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load pretrained word embeddings (hdf5 file) on the decoder side. See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_dec', 0, [[If = 1, fix word embeddings on the decoder side]])

-- bookkeeping
cmd:option('-save_every', 1, [[Save every this many epochs]])
cmd:option('-print_every', 50, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])

-- beam
cmd:option('-beam', 5, [[Beam size]])
cmd:option('-targ_dict', 'mydata/idx_to_word.txt', [[Path to target vocabulary, "id word" per line]])
cmd:option('-max_sent_l', 196, [[Maximum sentence length. If any sequences in srcfile are longer than this then it will error out]])
cmd:option('-simple', 0, [[If = 1, output prediction is simply the first time the top of the beam ends with an end-of-sentence token. If = 0, the model considers all hypotheses that have been generated so far that ends with end-of-sentence token and takes the highest scoring of all of them.]])

function main()
    -- Parse input params
    opt = cmd:parse(arg)
    torch.manualSeed(opt.seed)

    run(opt)   
end

main()
