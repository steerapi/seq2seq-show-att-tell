require 'nn'
require 'nngraph'
require 'hdf5'

require 'seqcode.data'
require 'seqcode.util'
require 'seqcode.models'
require 'seqcode.model_utils'

function zero_table(t)
    for i = 1, #t do
        if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
            if i == 1 then
                cutorch.setDevice(opt.gpuid)
            else
                cutorch.setDevice(opt.gpuid2)
            end
        end
        t[i]:zero()
    end
end

function plotpng(filename, plepochs, pllosses, title, ylabel, xlabel)
    gnuplot.pngfigure(filename)
    gnuplot.title(title or 'PPL over time')
    gnuplot.xlabel(xlabel or 'Epoch')
    gnuplot.ylabel(ylabel or 'PPL')
    gnuplot.plot(torch.Tensor(plepochs), torch.Tensor(pllosses))
    gnuplot.plotflush()
end

function train(train_data, valid_data, layers, criterion)
    local encoder = layers[1]
    local decoder = layers[2]
    local generator = layers[3]
    
    local timer = torch.Timer()
    local num_params = 0
    local start_decay = 0
    params, grad_params = {}, {}
    opt.train_perf = {}
    opt.val_perf = {}

    for i = 1, #layers do
        if opt.gpuid2 >= 0 then
            if i == 1 then
                cutorch.setDevice(opt.gpuid)
            else
                cutorch.setDevice(opt.gpuid2)
            end
        end
        local p, gp = layers[i]:getParameters()
        if not path.exists(opt.train_from) then
            p:uniform(-opt.param_init, opt.param_init)
        end
        num_params = num_params + p:size(1)
        params[i] = p
        grad_params[i] = gp
    end
    
    if opt.pre_word_vecs_dec:len() > 0 then      
        local f = hdf5.open(opt.pre_word_vecs_dec)     
        local pre_word_vecs = f:read('word_vecs'):all()
        for i = 1, pre_word_vecs:size(1) do
            word_vecs_dec.weight[1]:copy(pre_word_vecs[i])
        end      
    end

    print("Number of parameters: " .. num_params)

    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
        cutorch.setDevice(opt.gpuid)
        word_vecs_enc.weight[1]:zero()      
        cutorch.setDevice(opt.gpuid2)
        word_vecs_dec.weight[1]:zero()
    else
        word_vecs_enc.weight[1]:zero()            
        word_vecs_dec.weight[1]:zero()
    end         

    -- prototypes for gradients so there is no need to clone
    local encoder_grad_proto = torch.zeros(valid_data.batch_l:max(), valid_data.max_source_sent_l, opt.rnn_size)
    local encoder_grad_proto2 = torch.zeros(valid_data.batch_l:max(), valid_data.max_source_sent_l, opt.rnn_size)
    context_proto = torch.zeros(valid_data.batch_l:max(), valid_data.max_source_sent_l, opt.rnn_size)
    context_proto2 = torch.zeros(valid_data.batch_l:max(), valid_data.max_source_sent_l, opt.rnn_size)

    -- clone encoder/decoder up to max source/target length
    decoder_clones = clone_many_times(decoder, valid_data.max_target_sent_l)
    encoder_clones = clone_many_times(encoder, valid_data.max_source_sent_l)
    for i = 1, valid_data.max_target_sent_l do
        --attn_clones_idx = i
        --decoder_clones[i]:apply(get_layer)
        if decoder_clones[i].apply then
            decoder_clones[i]:apply(function(m) m:setReuse() end)
        end
    end
    for i = 1, valid_data.max_source_sent_l do
        --attn_clones_idx = i
        if encoder_clones[i].apply then
            encoder_clones[i]:apply(function(m) m:setReuse() end)
        end
    end   

    local h_init_dec = torch.zeros(valid_data.batch_l:max(), opt.rnn_size)
    local h_init_enc = torch.zeros(valid_data.batch_l:max(), opt.rnn_size)      
    if opt.gpuid >= 0 then
        h_init_enc = h_init_enc:cuda()      
        h_init_dec = h_init_dec:cuda()
        cutorch.setDevice(opt.gpuid)
        if opt.gpuid2 >= 0 then
            cutorch.setDevice(opt.gpuid)
            encoder_grad_proto2 = encoder_grad_proto2:cuda()
            context_proto = context_proto:cuda()	 
            cutorch.setDevice(opt.gpuid2)
            encoder_grad_proto = encoder_grad_proto:cuda()
            context_proto2 = context_proto2:cuda()	 
        else
            context_proto = context_proto:cuda()
            encoder_grad_proto = encoder_grad_proto:cuda()	 
        end
    end

    init_fwd_enc = {}
    init_bwd_enc = {}
    init_fwd_dec = {h_init_dec:clone()} -- initial context
    init_bwd_dec = {h_init_dec:clone()} -- just need one copy of this

    for L = 1, opt.num_layers do
        table.insert(init_fwd_enc, h_init_enc:clone())
        table.insert(init_fwd_enc, h_init_enc:clone())
        table.insert(init_bwd_enc, h_init_enc:clone())
        table.insert(init_bwd_enc, h_init_enc:clone())
        table.insert(init_fwd_dec, h_init_dec:clone()) -- memory cell
        table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden state
        table.insert(init_bwd_dec, h_init_dec:clone())
        table.insert(init_bwd_dec, h_init_dec:clone())      
    end      

    function reset_state(state, batch_l, t)
        local u = {[t] = {}}
        for i = 1, #state do
            state[i]:zero()
            table.insert(u[t], state[i][{{1, batch_l}}])
        end
        if t == 0 then
            return u
        else
            return u[t]
        end      
    end

    function clean_layer(layer)
        if opt.gpuid >= 0 then
            layer.output = torch.CudaTensor()
            layer.gradInput = torch.CudaTensor()
        else
            layer.output = torch.DoubleTensor()
            layer.gradInput = torch.DoubleTensor()
        end
        if layer.modules then
            for i, mod in ipairs(layer.modules) do
                clean_layer(mod)
            end
        elseif torch.type(self) == "nn.gModule" then
            layer:apply(clean_layer)
        end      
    end

    -- decay learning rate if val perf does not improve or we hit the opt.start_decay_at limit
    function decay_lr(epoch)
        --print(opt.val_perf)
        if epoch >= opt.start_decay_at then
            start_decay = 1
        end

        if opt.val_perf[#opt.val_perf] ~= nil and opt.val_perf[#opt.val_perf-1] ~= nil then
            local curr_ppl = opt.val_perf[#opt.val_perf]
            local prev_ppl = opt.val_perf[#opt.val_perf-1]
            if curr_ppl > prev_ppl then
                start_decay = 1
            end
        end
        if start_decay == 1 then
            opt.learning_rate = opt.learning_rate * opt.lr_decay
        end
    end   

    function train_batch(data, epoch)
        local train_nonzeros = 0
        local train_loss = 0	       
        local batch_order = torch.randperm(data.length) -- shuffle mini batch order     
        local start_time = timer:time().real
        local num_words_target = 0
        local num_words_source = 0


        for i = 1, data:size() do
            zero_table(grad_params, 'zero')
            local d
            if epoch <= opt.curriculum then
                d = data[i]
            else
                d = data[batch_order[i]]
            end
            local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
            local batch_l, target_l, source_l = d[5], d[6], d[7]

            local encoder_grads = encoder_grad_proto[{{1, batch_l}, {1, source_l}}]

            local rnn_state_enc = reset_state(init_fwd_enc, batch_l, 0)
            local context = context_proto[{{1, batch_l}, {1, source_l}}]
            if opt.gpuid >= 0 then
                cutorch.setDevice(opt.gpuid)
            end

            -- forward prop encoder
            for t = 1, source_l do
                encoder_clones[t]:training()
                local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
                local out = encoder_clones[t]:forward(encoder_input)
                rnn_state_enc[t] = out
                context[{{},t}]:copy(out[#out])
            end

            if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
                cutorch.setDevice(opt.gpuid2)	    
                local context2 = context_proto2[{{1, batch_l}, {1, source_l}}]
                context2:copy(context)
                context = context2
            end

            -- forward prop decoder
            local rnn_state_dec = reset_state(init_fwd_dec, batch_l, 0)
            if opt.init_dec == 1 then
                for L = 1, opt.num_layers do
                    rnn_state_dec[0][L*2]:copy(rnn_state_enc[source_l][L*2-1])
                    rnn_state_dec[0][L*2+1]:copy(rnn_state_enc[source_l][L*2])
                end
            end	 

            local preds = {}
            local decoder_input
            for t = 1, target_l do
                decoder_clones[t]:training()
                local decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
                local out = decoder_clones[t]:forward(decoder_input)
                local next_state = {}
                table.insert(preds, out[#out])
                table.insert(next_state, out[#out])
                for j = 1, #out-1 do
                    table.insert(next_state, out[j])
                end
                rnn_state_dec[t] = next_state
            end

            -- backward prop decoder
            encoder_grads:zero()	 
            local drnn_state_dec = reset_state(init_bwd_dec, batch_l, 1)
            local loss = 0
            for t = target_l, 1, -1 do
                local pred = generator:forward(preds[t])
                loss = loss + criterion:forward(pred, target_out[t])/batch_l
                local dl_dpred = criterion:backward(pred, target_out[t])
                dl_dpred:div(batch_l)
                local dl_dtarget = generator:backward(preds[t], dl_dpred)
                drnn_state_dec[#drnn_state_dec]:add(dl_dtarget)
                local decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
                local dlst = decoder_clones[t]:backward(decoder_input, drnn_state_dec)
                -- accumulate encoder/decoder grads
                encoder_grads:add(dlst[2])
                drnn_state_dec[#drnn_state_dec]:zero()
                drnn_state_dec[#drnn_state_dec]:add(dlst[3])
                for j = 4, #dlst do
                    drnn_state_dec[j-3]:copy(dlst[j])
                end	    
            end
            word_vecs_dec.gradWeight[1]:zero()
            if opt.fix_word_vecs_dec == 1 then
                word_vecs_dec.gradWeight:zero()
            end

            local grad_norm = 0
            grad_norm = grad_norm + grad_params[2]:norm()^2 + grad_params[3]:norm()^2

            -- backward prop encoder
            if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
                cutorch.setDevice(opt.gpuid)
                local encoder_grads2 = encoder_grad_proto2[{{1, batch_l}, {1, source_l}}]
                encoder_grads2:zero()
                encoder_grads2:copy(encoder_grads)
                encoder_grads = encoder_grads2 -- batch_l x source_l x rnn_size
            end

            local drnn_state_enc = reset_state(init_bwd_enc, batch_l, 1)
            if opt.init_dec == 1 then
                for L = 1, opt.num_layers do
                    drnn_state_enc[L*2-1]:copy(drnn_state_dec[L*2-1])
                    drnn_state_enc[L*2]:copy(drnn_state_dec[L*2])
                end	    
            end

            for t = source_l, 1, -1 do
                local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
                drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{},t}])
                local dlst = encoder_clones[t]:backward(encoder_input, drnn_state_enc)
                for j = 1, #drnn_state_enc do
                    drnn_state_enc[j]:copy(dlst[j+1])
                end	    
            end

            word_vecs_enc.gradWeight[1]:zero()
            if opt.fix_word_vecs_enc == 1 then
                word_vecs_enc.gradWeight:zero()
            end

            grad_norm = (grad_norm + grad_params[1]:norm()^2)^0.5

            -- Shrink norm and update params
            local param_norm = 0
            local shrinkage = opt.max_grad_norm / grad_norm
            for j = 1, #grad_params do
                if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
                    if j == 1 then
                        cutorch.setDevice(opt.gpuid)
                    else
                        cutorch.setDevice(opt.gpuid2)
                    end
                end
                if shrinkage < 1 then
                    grad_params[j]:mul(shrinkage)
                end	    
                params[j]:add(grad_params[j]:mul(-opt.learning_rate))
                param_norm = param_norm + params[j]:norm()^2
            end	    
            param_norm = param_norm^0.5

            -- Bookkeeping
            num_words_target = num_words_target + batch_l*target_l
            num_words_source = num_words_source + batch_l*source_l
            train_nonzeros = train_nonzeros + nonzeros
            train_loss = train_loss + loss*batch_l
            local time_taken = timer:time().real - start_time
            if i % opt.print_every == 0 then
                local stats = string.format('Epoch: %d, Batch: %d/%d, Batch size: %d, LR: %.4f, ',
                    epoch, i, data:size(), batch_l, opt.learning_rate)
                stats = stats .. string.format('PPL: %.2f, |Param|: %.2f, |GParam|: %.2f, ',
                    math.exp(train_loss/train_nonzeros), param_norm, grad_norm)
                stats = stats .. string.format('Training: %d/%d/%d total/source/target tokens/sec',
                    (num_words_target+num_words_source) / time_taken,
                    num_words_source / time_taken,
                    num_words_target / time_taken)			   
                print(stats)
            end
            if i % 200 == 0 then
                collectgarbage()
            end
        end
        return train_loss, train_nonzeros
    end   
    
    local start_time = os.clock()
    
    local plepochs = {}
    local pltrainPPLs = {}
    local plvalidPPLs = {}

    local start = 1
    local endepochs = opt.epochs
    local timetook = 0
    
    local lossdata
    
    if opt.lossfile ~= nil and path.exists(opt.lossfile.. '.th') then
        lossdata = torch.load(opt.lossfile .. '.th')
        plepochs = lossdata.plepochs
        pltrainPPLs = lossdata.pltrainPPLs
        plvalidPPLs = lossdata.plvalidPPLs
        
        start = start+#plepochs
        --endepochs = start-1+epochs
        
        timetook = lossdata.timetook
    end
    
    local savefile = string.format('%s', opt.savefile)      
        
    local total_loss, total_nonzeros, batch_loss, batch_nonzeros
    for epoch = start, endepochs do
        generator:training()
        total_loss, total_nonzeros = train_batch(train_data, epoch)
        local train_score = math.exp(total_loss/total_nonzeros)
        print('Train', train_score)
        opt.train_perf[#opt.train_perf + 1] = train_score
        local score = eval(valid_data,generator,criterion,{encoder_clones,decoder_clones,context_proto,context_proto2})
        
        opt.val_perf[#opt.val_perf + 1] = score
        decay_lr(epoch)
        
        table.insert(plepochs, epoch)
        table.insert(pltrainPPLs, train_score)
        table.insert(plvalidPPLs, score)
        plotpng('trainPPL_' .. opt.lossfile, plepochs, pltrainPPLs)
        plotpng('validPPL_' .. opt.lossfile, plepochs, plvalidPPLs)
        
        -- clean and save models
        if epoch % opt.save_every == 0 then
            print('saving checkpoint to ' .. savefile)
            --clean_layer(encoder); clean_layer(decoder); clean_layer(generator)
            torch.save(savefile, {{encoder, decoder, generator, criterion}, opt})
            
            lossdata = {
                plepochs=plepochs,
                pltrainPPLs=pltrainPPLs,
                plvalidPPLs=plvalidPPLs,
                timetook=timetook
            }
            torch.save(opt.lossfile .. '.th', lossdata)
            print('done')

        end
    end
    
    local end_time = os.clock()    
    timetook = timetook + end_time-start_time
    print("Training Time",timetook,"s")    

    print('saving final model to ' .. savefile)   
    lossdata = {
        plepochs=plepochs,
        pltrainPPLs=pltrainPPLs,
        plvalidPPLs=plvalidPPLs,
        timetook=timetook
    }
    torch.save(opt.lossfile .. '.th', lossdata)

    -- save final model
    local savefile = string.format('%s', opt.savefile)
    --clean_layer(encoder); clean_layer(decoder); clean_layer(generator)
    torch.save(savefile, {{encoder, decoder, generator, criterion}, opt})
    print('done')
end

function eval(data,generator,criterion,states)
    
    local encoder_clones = states[1]
    local decoder_clones = states[2]
    local context_proto = states[3]
    local context_proto2 = states[4]
    
    encoder_clones[1]:evaluate()   
    decoder_clones[1]:evaluate() -- just need one clone
    generator:evaluate()
    local nll = 0
    local total = 0
    for i = 1, data:size() do
        local d = data[i]
        local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
        local batch_l, target_l, source_l = d[5], d[6], d[7]
        local rnn_state_enc = reset_state(init_fwd_enc, batch_l, 1)
        local context = context_proto[{{1, batch_l}, {1, source_l}}]
        -- forward prop encoder
        for t = 1, source_l do
            local encoder_input = {source[t], table.unpack(rnn_state_enc)}
            local out = encoder_clones[1]:forward(encoder_input)
            rnn_state_enc = out
            context[{{},t}]:copy(out[#out])
        end

        if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
            cutorch.setDevice(opt.gpuid2)
            local context2 = context_proto2[{{1, batch_l}, {1, source_l}}]
            context2:copy(context)
            context = context2
        end

        local rnn_state_dec = reset_state(init_fwd_dec, batch_l, 1)
        if opt.init_dec == 1 then
            for L = 1, opt.num_layers do
                rnn_state_dec[L*2]:copy(rnn_state_enc[L*2-1])
                rnn_state_dec[L*2+1]:copy(rnn_state_enc[L*2])
            end	 
        end      
        local loss = 0
        for t = 1, target_l do
            local decoder_input = {target[t], context, table.unpack(rnn_state_dec)}
            local out = decoder_clones[1]:forward(decoder_input)
            rnn_state_dec = {}
            table.insert(rnn_state_dec, out[#out])
            for j = 1, #out-1 do
                table.insert(rnn_state_dec, out[j])
            end
            local pred = generator:forward(out[#out])
            loss = loss + criterion:forward(pred, target_out[t])
        end
        nll = nll + loss
        total = total + nonzeros
    end
    local valid = math.exp(nll / total)
    print("Valid", valid)
    return valid
end


function get_layer(layer)
    if layer.name ~= nil then
        if layer.name == 'word_vecs_dec' then
            word_vecs_dec = layer	 
        elseif layer.name == 'word_vecs_enc' then
            word_vecs_enc = layer
        elseif layer.name == 'decoder_attn' then	 
            decoder_attn = layer
        end
    end
end
