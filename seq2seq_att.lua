require("nn")
require('nnx')
require('dpnn')
require('nngraph')

require('seqcode.models')
require('seqcode.train')
require("helper_nn")

function seq2seq_att(opt)

    local buildmodel = function(opt)

        local encoder,decoder,generator, criterion


        if not path.exists(opt.train_from) then
            encoder = make_lstm(opt, 'enc')
            decoder = make_lstm(opt, 'dec')
            generator, criterion = make_generator(opt)
        else
            print('loading ' .. opt.train_from .. '...')
            local checkpoint = torch.load(opt.train_from)
            local model, model_opt = checkpoint[1], checkpoint[2]
            opt.source_size = model_opt.source_size
            opt.target_size = model_opt.target_size
            opt.num_layers = model_opt.num_layers
            opt.rnn_size = model_opt.rnn_size
            encoder = model[1]
            decoder = model[2]
            generator = model[3]
            criterion = model[4]
        end
        
        local layers = {encoder, decoder, generator}

        if opt.gpuid >= 0 then
            for i = 1, #layers do	 
                if opt.gpuid2 >= 0 then 
                    if i == 1 then
                        cutorch.setDevice(opt.gpuid) --encoder on gpu1
                    else
                        cutorch.setDevice(opt.gpuid2) --decoder/generator on gpu2
                    end
                end	 
                layers[i]:cuda()
            end
            if opt.gpuid2 >= 0 then
                cutorch.setDevice(opt.gpuid2) --criterion on gpu2
            end      
            criterion:cuda()      
        end

        encoder:apply(get_layer)
        decoder:apply(get_layer)

        return layers,criterion
    end

    return model_train(buildmodel, opt)
end