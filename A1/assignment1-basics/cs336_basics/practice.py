import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    
    #subtract the largest elemennt for numerical stability
    new = logits - torch.max(logits)
    #exp to all logits
    exp_logits = torch.exp(new[targets])
    
    #softmax
    prob = exp_logits / exp_logits.sum()
    
    comp_logits = torch.exp(logits[targets])/exp_logits
    result = -torch.log(comp_logits)
    return result

def softmax(tensor: torch.tensor, i: int) -> int:
    up = torch.exp(tensor[i] - tensor.max()) 
    exp_tensor = torch.exp(tensor - tensor.max()) 
    down = exp_tensor.sum()
    res = up/down.sum()
    return res

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val = x.max(dim=dim, keepdim=True)[0]
    x_exp=torch.exp(x - max_val) 
    partition = x_exp.sum(dim,keepdim=True)
    return x_exp / partition

def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None
) -> torch.Tensor:
    dk = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    attn_weights = softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)
