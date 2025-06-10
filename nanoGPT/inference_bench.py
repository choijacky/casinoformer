"""
Inference with perplexity from a trained model
"""
import sys
import datetime
import os
from contextlib import nullcontext
import torch
import tiktoken
from tqdm import tqdm
import math
from model import GPT



def write_to_file (out_data_path, tot_log_probs, x_size):
    write_mode = "a" if os.path.exists(out_data_path) else "w"
    with open(out_data_path, write_mode) as logfile:
        logfile.write("\n==============================================================\n")
        logfile.write("# of tokens processed: " + str(x_size) + ", tot_log_probs = " + str(tot_log_probs))
        logfile.write(", perplexity: " + str(math.exp(-tot_log_probs / x_size)))


#accepts two arguments: attention_method of type string and number_of_terms_exponent of type float
if __name__ == '__main__':
    
    attention_method = str(sys.argv[1])
    number_of_terms_exponent = float(sys.argv[2])

    assert os.path.split(os.getcwd())[1] == "casinoformer-deep-learning-project-hs23", "current working directory not set to casinoformer"
    torch.manual_seed(0)
    
    # Config from nanoGPT
    # -----------------------------------------------------------------------------
    init_from = 'gpt2'
    out_dir = 'nanoGPT/out'
    start = "FILE:nanoGPT/data/wikitext-2-v1/wiki.test.tokens"#\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    max_new_tokens = 1 # number of tokens generated in each sample
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = None  #Was 200 initally. Retain only the top_k most likely tokens, clamp others to have 0 probability
    device = torch.device('cpu') # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = torch.float16 # 'float32' or 'bfloat16' or 'float16'
    compile = False # use PyTorch 2.0 to compile the model to be faster

    #exec(open(os.path.join(os.getcwd(), 'nanoGPT/configurator.py')).read()) # overrides from command line or config file
    very_verbose = False 
    # -----------------------------------------------------------------------------
    
    # Config for benchmarking
    extension_param = {
        "attention_method": attention_method, # one of "tree-less", "alignment", "FAVOR+ReLU", "FAVOR+", "positive_alignment", "exponentially_decaying_horizon" or "uniform"
        "number_of_terms_exponent": number_of_terms_exponent, # in [0.2,0.9], computes l^number_of_terms_exponent terms instead of l, only relevant if attention_method is not "tree-less"
        #"num_processes": 32, #between 1 and block_size, set to 32 for all test runs
    }
    percentage_test_set = float(sys.argv[3]) #percentage of test set used for benchmarking
    offset = float(sys.argv[4]) #offset applied to original test datset
    # -----------------------------------------------------------------------------

    print("percentage of test set: ", percentage_test_set, " with offset ", offset)


    device_type = 'cpu'
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=dtype)

    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, extension_param)

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    print("loading file")
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            data = f.read()

    print("start encoding")
    data_ids = encode(data)
    n_dataset_start_block = math.ceil(offset * len(data_ids)/model.config.block_size)
    n_dataset_end_block = math.ceil(percentage_test_set * len(data_ids)/model.config.block_size) + n_dataset_start_block
    x = (torch.tensor(data_ids[n_dataset_start_block*model.config.block_size:n_dataset_end_block*model.config.block_size], dtype=torch.long, device=device)[None, ...])
    print(n_dataset_start_block*model.config.block_size)
    print("size of x: ", x.size())
    
    
    
    if extension_param["attention_method"] != "tree-less":
        out_data_path = os.path.join(out_dir, 'perplexity_' + extension_param["attention_method"] + '_' + str(extension_param["number_of_terms_exponent"]) + '_fast_sampling.txt')
    else:
        out_data_path = os.path.join(out_dir, 'perplexity_score__' + extension_param["attention_method"] + '.txt')

    write_mode = "a" if os.path.exists(out_data_path) else "w"    
    with open(out_data_path, write_mode) as logfile:
        logfile.write("\n==============================================================\n")
        logfile.write(str(datetime.datetime.now()) + "\n")
        logfile.write("Configuration used: \n")
        logfile.write(str(model.config) + "\n")
        logfile.write("Dataset used: " +start[5:]+ " with dataset ratio " + str(percentage_test_set) + " and offset " + str(offset))

    ## run generation   
    with torch.no_grad():
        with ctx:
            tot_log_probs = 0
            #after 1024 tokens, jump to the next block and start again, increasing context size for every iteration
            for j in tqdm(range(max(1, x.shape[1]//model.config.block_size + 1))):
                if extension_param["attention_method"] != "tree-less":
                    model = GPT.from_pretrained(init_from, extension_param)  
                model.delete_cache()
                #for tokens in block of size 1024, compute perplexity
                for i in tqdm(range(j*model.config.block_size+1, min(x.shape[1], (j+1)*model.config.block_size))):
                    y, log_probs = model.generate_with_probs(x[:,j*model.config.block_size:i], max_new_tokens, temperature=temperature, top_k=top_k)
                    decoded_y = decode(y[0].tolist())
                    tot_log_probs += log_probs.item()
                    if very_verbose:
                        print("INPUT: ", decode(x[:,j*model.config.block_size:i].tolist()), "\nGENERATED: ", decoded_y, "| logprobs: ", log_probs)
                write_to_file(out_data_path, tot_log_probs, min(x.shape[1], (j+1)*model.config.block_size))
                print("current perplexity: ", math.exp(-tot_log_probs / min(x.shape[1], (j+1)*model.config.block_size)))
            perplexity = math.exp(-tot_log_probs / x.shape[1])
    print(perplexity)