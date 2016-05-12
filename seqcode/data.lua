--
-- Manages encoder/decoder data matrices.
--
require('hdf5')

local data = torch.class("data")

function data:__init(opt, data_file)
    
    local f = hdf5.open(data_file, 'r')
       
    self.source_input = f:read('source_input'):all()

    self.target_input_table = {}
    self.target_output_table = {}
    self.total_words_table = {}
    
    local num_sentences = opt.num_sentences
    for i=0,num_sentences-1 do
        self.target_input_table[i+1] = f:read('target_input_'..i):all()
        self.target_output_table[i+1] = f:read('target_output_'..i):all()
        self.total_words_table[i+1] = f:read('total_words_'..i):all()
    end
    
    self.batch_l = f:read('batch_l'):all()
    self.target_l = f:read('target_l'):all()
    self.source_l = f:read('source_l'):all()

    -- local num_batches = f:read('num_batches'):all()[1]
    local num_batches = self.source_input:size(1)
    self.length = num_batches*num_sentences
    self.max_target_sent_l = f:read('max_target_sent_l'):all()[1]   
    self.max_source_sent_l = f:read('max_source_sent_l'):all()[1]   
    -- print("num_batches",num_batches,f:read('num_batches'):all()[1])

    f:close()
    self.batches = {}

    local source_l_rev = torch.ones(self.max_source_sent_l):long()
    for i = 1, self.max_source_sent_l do
        source_l_rev[i] = self.max_source_sent_l - i + 1
    end
    
    
    for i = 1, num_batches do
        --print("self.source_input",self.source_input:size())
        local source_input_i = self.source_input[i]:transpose(1,2)
        
        if opt.reverse_src == 1 then
            source_input_i = source_input_i:index(1, source_l_rev[{{1, self.max_source_sent_l}}])
        end
        
        for j = 1, num_sentences do
            local target_input = self.target_input_table[j]
            local target_output = self.target_output_table[j]
            local total_words = self.total_words_table[j]
            
            local target_input_i = target_input[i]:transpose(1,2)
            local target_output_i = target_output[i]:transpose(1,2)
            local total_words_i = total_words[i]:sum()

            table.insert(self.batches, { target_input_i,                
                    target_output_i,
                    total_words_i,  -- sent len for each batch
                    source_input_i,
                    self.batch_l[i], -- length of the batch
                    self.target_l[i], -- max sent len target
                    self.source_l[i] })
        end
    end
end

function data:size()
    return self.length
end

function data.__index(self, idx)
    if type(idx) == "string" then
        return data[idx]
    else
        local target_input = self.batches[idx][1]
        local target_output = self.batches[idx][2]
        local nonzeros = self.batches[idx][3]
        local source_input = self.batches[idx][4]      
        local batch_l = self.batches[idx][5]
        local target_l = self.batches[idx][6]
        local source_l = self.batches[idx][7]

        if opt.gpuid >= 0 then --if multi-gpu, source lives in gpuid1, rest on gpuid2
            cutorch.setDevice(opt.gpuid)
            source_input = source_input:cuda()
            if opt.gpuid2 >= 0 then
                cutorch.setDevice(opt.gpuid2)
            end	 
            target_input = target_input:cuda()
            target_output = target_output:cuda()
        end
        return {target_input, target_output, nonzeros, source_input,
            batch_l, target_l, source_l}
    end
end

return data
