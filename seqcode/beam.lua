require 'nn'
require 'string'
require 'hdf5'
require 'nngraph'

require 'seqcode.models'
require 'seqcode.util'

local stringx = require('pl.stringx')

local function copy(orig)
   local orig_type = type(orig)
   local copy
   if orig_type == 'table' then
      copy = {}
      for orig_key, orig_value in pairs(orig) do
         copy[orig_key] = orig_value
      end
   else
      copy = orig
   end
   return copy
end

local StateAll = torch.class("StateAll")

function StateAll.initial(start)
   return {start}
end

function StateAll.advance(state, token)
   local new_state = copy(state)
   table.insert(new_state, token)
   return new_state
end

function StateAll.disallow(out)
   local bad = {1, 3} -- 1 is PAD, 3 is BOS
   for j = 1, #bad do
      out[bad[j]] = -1e9
   end
end

function StateAll.same(state1, state2)
   for i = 2, #state1 do
      if state1[i] ~= state2[i] then
         return false
      end
   end
   return true
end

function StateAll.next(state)
   return state[table.getn(state)]
end

function StateAll.heuristic(state)
   return 0
end

function StateAll.print(state)
   for i = 1, #state do
      io.write(state[i] .. " ")
   end
   print()
end

local START_WORD = 3
local END_WORD = 4

-- Convert a flat index to a row-column tuple.
local function flat_to_rc(v, flat_index)
   local row = math.floor((flat_index - 1) / v:size(2)) + 1
   return row, (flat_index - 1) % v:size(2) + 1
end

local function generate_beam(model, initial, K, max_sent_l, source, init_fwd_enc, init_fwd_dec, context_proto, context_proto2, opt)
   --reset decoder initial states
   if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid)
   end
   local n = max_sent_l
  -- Backpointer table.
   local prev_ks = torch.LongTensor(n, K):fill(1)
   -- Current States.
   local next_ys = torch.LongTensor(n, K):fill(1)
   -- Current Scores.
   local scores = torch.FloatTensor(n, K)
   scores:zero()
   local source_l = math.min(source:size(1), opt.max_sent_l)
   local attn_argmax = {}   -- store attn weights
   attn_argmax[1] = {}

   local attn_max = {}   -- store attn weights
   attn_max[1] = {}
    
   local states = {} -- store predicted word idx
   states[1] = {}
   for k = 1, 1 do
      table.insert(states[1], initial)
      table.insert(attn_argmax[1], initial)
      table.insert(attn_max[1], {torch.zeros(opt.max_sent_l)})
      next_ys[1][k] = StateAll.next(initial)
   end

   local source_input = source:view(source_l, opt.source_size)

   local rnn_state_enc = {}
   for i = 1, #init_fwd_enc do
      table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
   end   
   local context = context_proto[{{}, {1,source_l}}]:clone() -- 1 x source_l x rnn_size
   
   for t = 1, source_l do
      local encoder_input = {source_input[t]:view(1,-1), table.unpack(rnn_state_enc)}
      local out = model[1]:forward(encoder_input)
      rnn_state_enc = out
      context[{{},t}]:copy(out[#out])
   end
   context = context:expand(K, source_l, opt.rnn_size)
   
   if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid2)
      local context2 = context_proto2[{{1, K}, {1, source_l}}]
      context2:copy(context)
      context = context2
   end

   rnn_state_dec = {}
   for i = 1, #init_fwd_dec do
      table.insert(rnn_state_dec, init_fwd_dec[i]:zero())
   end

   out_float = torch.FloatTensor()
   
   local i = 1
   local done = false
   local max_score = -1e9
   local found_eos = false
   while (not done) and (i < n) do
      i = i+1
      states[i] = {}
      attn_argmax[i] = {}
      attn_max[i] = {}
        
      local decoder_input1
      decoder_input1 = next_ys:narrow(1,i-1,1):squeeze()
      if opt.beam == 1 then
        decoder_input1 = torch.LongTensor({decoder_input1})
      end	
        
      local decoder_input = {decoder_input1, context, table.unpack(rnn_state_dec)}
      local out_decoder = model[2]:forward(decoder_input)
      local out = model[3]:forward(out_decoder[#out_decoder]) -- K x vocab_size
      
      rnn_state_dec = {} -- to be modified later
      table.insert(rnn_state_dec, out_decoder[#out_decoder])
      for j = 1, #out_decoder - 1 do
         table.insert(rnn_state_dec, out_decoder[j])
      end
      out_float:resize(out:size()):copy(out)
      for k = 1, K do
         StateAll.disallow(out_float:select(1, k))
         out_float[k]:add(scores[i-1][k])
      end
      -- All the scores available.

       local flat_out = out_float:view(-1)
       if i == 2 then
          flat_out = out_float[1] -- all outputs same for first batch
       end

       
       for k = 1, K do
            
          while true do
             local score, index = flat_out:max(1)
             local score = score[1]
             local prev_k, y_i = flat_to_rc(out_float, index[1])
                

             states[i][k] = StateAll.advance(states[i-1][prev_k], y_i)
             local diff = true
             for k2 = 1, k-1 do
                if StateAll.same(states[i][k2], states[i][k]) then
                   diff = false
                end
             end
	     
             if i < 2 or diff then		
                local max_attn, max_index = decoder_softmax.output[prev_k]:max(1)
                attn_argmax[i][k] = StateAll.advance(attn_argmax[i-1][prev_k],max_index[1])
                attn_max[i][k] = StateAll.advance(attn_max[i-1][prev_k],decoder_softmax.output[prev_k])
                    
                prev_ks[i][k] = prev_k
                next_ys[i][k] = y_i
                scores[i][k] = score
                flat_out[index[1]] = -1e9
                break -- move on to next k 
             end
             flat_out[index[1]] = -1e9
          end
       end
       for j = 1, #rnn_state_dec do
          rnn_state_dec[j]:copy(rnn_state_dec[j]:index(1, prev_ks[i]))
       end
       end_hyp = states[i][1]
       end_score = scores[i][1]
       end_attn_argmax = attn_argmax[i][1]
       if end_hyp[#end_hyp] == END_WORD then
          done = true
          found_eos = true
       else
          for k = 1, K do
             local possible_hyp = states[i][k]
             if possible_hyp[#possible_hyp] == END_WORD then
                found_eos = true
                if scores[i][k] > max_score then
                   max_hyp = possible_hyp
                   max_k = k
                   max_score = scores[i][k]
                   max_attn_argmax = attn_argmax[i][k]
                end
             end	     
          end	  
       end       
   end
   
   if opt.simple == 1 or end_score > max_score or not found_eos then
      max_hyp = end_hyp
      max_k = 1
      max_score = end_score
      max_attn_argmax = end_attn_argmax
   end
   --local attns = torch.zeros(#max_hyp-2, opt.max_sent_l)
   --local m = 1
   --for j = 2,#max_hyp-1 do
   --     attns[m] = attn_max[j][max_k]:double()
   --     m = m + 1
   --end
   --print('states',states)
   --print('scores',scores)
   --print('attn_argmax',attn_argmax[i])
   return max_hyp, max_score, max_attn_argmax, states[i], scores[i], attn_argmax[i], attn_max[i][max_k]
end

local function idx2key(file)   
   local f = io.open(file,'r')
   local t = {}
   for line in f:lines() do
      local c = {}
      for w in line:gmatch'([^%s]+)' do
        table.insert(c, w)
      end
      t[tonumber(c[1])] = c[2]
   end
   return t
end

local function wordidx2sent(sent, idx2word)
    local t = {}
    local start_i, end_i
    for i = 2, #sent-1 do -- skip START and END
        table.insert(t, idx2word[sent[i]])	 
    end
    return table.concat(t, ' ')
end

function run_beam(model, source, opt)

    local idx2word_targ = idx2key(opt.targ_dict)
   
    local decoder = model[2]
    local softmax_layers = {}

    local function get_layer(layer)
       if layer.name ~= nil then
          if layer.name == 'decoder_attn' then
             decoder_attn = layer
          elseif layer.name:sub(1,3) == 'hop' then
             hop_attn = layer
          elseif layer.name:sub(1,7) == 'softmax' then
             table.insert(softmax_layers, layer)
          elseif layer.name == 'word_vecs_enc' then
             word_vecs_enc = layer
          elseif layer.name == 'word_vecs_dec' then
             word_vecs_dec = layer
          end       
       end
    end

    decoder:apply(get_layer)
    decoder_attn:apply(get_layer)
    decoder_softmax = softmax_layers[1]
    local attn_layer = torch.zeros(opt.beam, opt.max_sent_l)
    
    local context_proto = torch.zeros(1, opt.max_sent_l, opt.rnn_size)
    local h_init_dec = torch.zeros(opt.beam, opt.rnn_size)
    local h_init_enc = torch.zeros(1, opt.rnn_size) 

    if opt.gpuid >= 0 then
        h_init_enc = h_init_enc:cuda()      
        h_init_dec = h_init_dec:cuda()
        cutorch.setDevice(opt.gpuid)
        if opt.gpuid2 >= 0 then
            cutorch.setDevice(opt.gpuid)
            context_proto = context_proto:cuda()	 
            cutorch.setDevice(opt.gpuid2)
            context_proto2 = torch.zeros(opt.beam, opt.max_sent_l, opt.rnn_size):cuda()
        else
            context_proto = context_proto:cuda()
        end
        attn_layer = attn_layer:cuda()
    end

    local init_fwd_enc = {}
    local init_fwd_dec = {h_init_dec:clone()} -- initial context   
    for L = 1, opt.num_layers do
        table.insert(init_fwd_enc, h_init_enc:clone())
        table.insert(init_fwd_enc, h_init_enc:clone())
        table.insert(init_fwd_dec, h_init_dec:clone()) -- memory cell
        table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden state      
    end      

    local pred_score_total = 0
    local pred_words_total = 0

    local sent_id = 0

    local out_file = io.open(opt.outfile,'w')
    
    local attns_vectors = torch.zeros(source:size(1)*source:size(2), opt.max_sent_l, opt.max_sent_l)
    local j = 1
    for b = 1,source:size(1) do
        for i = 1,source:size(2) do
            local source_input = source[b][i]

            sent_id = sent_id + 1
            
            local state = StateAll.initial(START_WORD)

            local pred, pred_score, attn, all_sents, all_scores, all_attn, attns = generate_beam(model, state, opt.beam, opt.max_sent_l, source_input, init_fwd_enc, init_fwd_dec, context_proto, context_proto2, opt)        
            -- saving attention
            -- print("#attns",#attns)
            -- print(attns[1]:size())
            -- print(attns_vectors:size())
            for m = 1,(#attns-1) do
                attns_vectors[{{j},{m},{}}] = attns[m+1]:double()
            end
            
            pred_score_total = pred_score_total + pred_score
            pred_words_total = pred_words_total + #pred - 1

            local pred_sent = wordidx2sent(pred, idx2word_targ)
            out_file:write(pred_sent .. '\n')
            print('PRED ' .. sent_id .. ': ' .. pred_sent)
            j = j + 1
        end
    end
    
    print("WRITE ATTENTION WEIGHTS")
    local attn_file = hdf5.open(opt.outfile..'_attns_vectors.hdf5', 'w')
    attn_file:write('attns_vectors', attns_vectors)
    attn_file:close()
    print("DONE")

    print(string.format("PRED AVG SCORE: %.4f, PRED PPL: %.4f", pred_score_total / pred_words_total, math.exp(-pred_score_total/pred_words_total)))

    out_file:close()
end
