"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
Code adopted from: https://github.com/karpathy/nanoGPT
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from transformers import GPT2LMHeadModel, GPT2ForSequenceClassification


################################################################################
# GPT Model classes
################################################################################

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    LoRA_rank: int = 4


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention, but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(
                torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size))
        
        # TODO: Inject LoRA layers here
        # 1. add the auxiliary LoRA linear layer here corresponding to c_attn
        # 2. add the auxiliary LoRA linear layer here corresponding to c_proj
        # 3. remove the gradients from the c_attn and c_proj layers
        
        # added by me 
        self.c_attn_lora = LoRALinear(config.n_embd, 3 * config.n_embd, config.LoRA_rank)
        self.c_proj_lora = LoRALinear(config.n_embd, config.n_embd, config.LoRA_rank)
        self.c_attn.weight.requires_grad = False # remove the gradients from the c_attn layer because we are using the LoRA layer instead of this
        self.c_proj.weight.requires_grad = False # remove the gradients from the c_proj layer because we are using the LoRA layer instead of this
        self.c_attn.bias.requires_grad = False 
        self.c_proj.bias.requires_grad = False
        
        # until here 
        
        

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)


        attn_out = self.c_attn(x)

        # TODO: send x through the auxiliary c_attn and add them back to attn_out
        x = self.c_attn_lora(x)
        attn_out = attn_out + x
        # print(f"shape of x in CausalSelfAttention: {attn_out.shape}")

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = attn_out.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        proj_out = self.c_proj(y)

        # TODO: send y through the auxiliary c_proj and add them back to proj_out
        # added by me
        proj_out = proj_out + self.c_proj_lora(y)
        # added by me 

        y = self.resid_dropout(proj_out)
        
        # print(f"shape of y after residual dropout: {y.shape}")
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        # TODO: Inject LoRA layers here
        # 1. add the auxiliary LoRA linear layer here corresponding to c_fc
        # 2. add the auxiliary LoRA linear layer here corresponding to c_proj
        # 3. remove the gradients from the c_fc and c_proj layers
        
        # added by me 
        self.c_fc_lora = LoRALinear(config.n_embd, 4 * config.n_embd, config.LoRA_rank)
        self.c_proj_lora = LoRALinear(4 * config.n_embd, config.n_embd, config.LoRA_rank)
        self.c_fc.weight.requires_grad = False
        self.c_proj.weight.requires_grad = False
        self.c_fc.bias.requires_grad = False
        self.c_proj.bias.requires_grad = False
        # added by me 
        

    def forward(self, x):
        # TODO: modify the forward pass to pass the input through the auxiliary LoRA layers and add them back to the compute graph for automatic backpropagation
       
        x = self.c_fc_lora(x) + self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj_lora(x) + self.c_proj(x)
        x = self.dropout(x)
        
        return x
        
        
        
        # commented by me which they have given
        # x = self.c_fc(x)
        # x = self.gelu(x)
        # x = self.c_proj(x)
        # x = self.dropout(x)
        
        
        # return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, model_type='gpt2', LoRA_rank=4, is_gen=False):
        super(GPT, self).__init__()
        
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M (1.3B) params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True        # always True for GPT model checkpoints
        config_args['LoRA_rank'] = LoRA_rank # decomposition rank for LoRA

        self.config = GPTConfig(**config_args)
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            drop = nn.Dropout(self.config.dropout),
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
            ln_f = LayerNorm(self.config.n_embd, bias=self.config.bias),
        ))
        if is_gen:
            self.lm_head = nn.Linear(
                self.config.n_embd, self.config.vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying
            self.score = None
        else:
            self.score = nn.Linear(self.config.n_embd, 2, bias=False)

        # TODO: Remove gradients from the embedding layers
        # added by me below
        self.transformer.wte.weight.requires_grad = False
        self.transformer.wpe.weight.requires_grad = False
        # added by me above 
        sd = self.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init a huggingface/transformers model
        print("loading weights from pretrained gpt2 model")
        if is_gen:
            model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        else:
            model_hf = GPT2ForSequenceClassification.from_pretrained(model_type)

        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        # print the total number of parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {total_params / 1e6:.2f}M")
        if not is_gen:
            # print the number of trainable parameters
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Number of trainable parameters: {num_params / 1e6:.2f}M")
            # calculate the reduction in parameters
            reduction = 100 * (total_params - num_params) / total_params
            print(f"Reduction: {reduction:.2f}%")

    def forward(self, idx, mask=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # if mask is provided, find the indices of the last tokens in each sequence
        if mask is not None:
            assert mask.size() == idx.size(), "Mask size must match input size"
            eos_idxs = mask.sum(1) - 1 # last non-pad token in each sequence

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if self.score is None:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        else: # no need to preserve the time dimension for classification task
            if mask is not None:
                # if mask is provided, only return the logits for the last token in each sequence
                logits = self.score(x[torch.arange(b, device=device), eos_idxs])
            else:
                
                logits = self.score(x[:, -1, :])
               
        return logits

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        
       
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            
            # print(f"shape of idx in generate: {idx.shape}") # added by me 
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    def save_trainable_params(self, path):
        trainable_params =\
            list(filter(lambda p: p.requires_grad, self.parameters()))
        torch.save(trainable_params, path)
    
    def load_trainable_params(self, path):
        trainable_params = torch.load(path)
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = trainable_params.pop(0)


################################################################################
# LoRA model
################################################################################

class LoRALinear(nn.Module):
    # TODO: Implement LoRA model. Add two matrics for LoRA decomposition
    # pass
    def __init__(self, in_features, out_features, rank):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_dropout = nn.Dropout(0.1)
        self.U = nn.Parameter(torch.Tensor(in_features, rank))
        self.V = nn.Parameter(torch.Tensor(rank, out_features))
        self.reset_parameters()
        
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        # nn.init.xavier_uniform_(self.U)
        # nn.init.xavier_uniform_(self.V)
        
    
    def forward(self, x):
        
        # print(f"shape of x in LoRALinear: {x.shape}")
        
        # perform low rank decomposition
        x = self.lora_dropout(x)
        x = torch.matmul(x, self.U)
        x = torch.matmul(x, self.V)
        
        # print(f"shape of x in LoRALinear: {x.shape}")
        
        return x 
        


################################################################################
# Distillation model
################################################################################

class DistilRNN(nn.Module):
    # TODO: Implement DistilRNN model
    def __init__(self):
        super(DistilRNN, self).__init__()
        
        self.embedding = nn.Embedding(50257, 768)
        self.rnn = nn.RNN(768, 768, num_layers=1, batch_first=True)
        # self.relu = nn.ReLU()
        self.fc = nn.Linear(768, 2)
        
    def forward(self, x,mask=None ):
        
        x = self.embedding(x)
        
        # h0 = torch.zeros(2, x.size(0), 768).to(x.device)
        x, _ = self.rnn(x)    
        # x = self.relu(x)
        x = self.fc(x[:, -1, :])
        return x
