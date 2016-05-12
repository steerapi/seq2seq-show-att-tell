require("hdf5")
require("nn")
require("optim")
require("gnuplot")
require('nnx')

require("nngraph")
require("seqcode.train")
require("seqcode.beam")

-- Test. Output test file to evaluate BLUE score
local model_test = function(model_bundle, opt)
    local model = model_bundle.model
    
    for i = 1, #model do
        model[i]:evaluate()
    end
    
    local f = hdf5.open(opt.testfile, 'r')
    local source = f:read('source_input'):all()
    
    if opt.reverse_src == 1 then
        local source_l_rev = torch.ones(opt.max_sent_l):long()
        for i = 1, opt.max_sent_l do
            source_l_rev[i] = opt.max_sent_l - i + 1
        end
        for i = 1, num_batches do
            local source_input_i = source[i]:transpose(1,2)
            source_input_i = source_input_i:index(1, source_l_rev[{{1, opt.max_sent_l}}])
        end
    end
        
    if opt.gpuid > 0 then
        source = source:cuda()
    end
    
    run_beam(model, source, opt)
end

function model_train(buildmodel, opt)

    local layers,criterion = buildmodel(opt)
    
    local train_data = data.new(opt, opt.trainfile)
    local valid_data = data.new(opt, opt.validfile)
    
    if opt.train == 1 then
        train(train_data, valid_data, layers, criterion)
    end
    
    local model_bundle = {}

    model_bundle.model = layers
    model_bundle.criterion = criterion
    model_bundle.test = model_test
    
    return model_bundle
end
