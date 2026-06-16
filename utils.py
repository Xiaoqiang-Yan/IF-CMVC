import torch
import numpy as np
import torch.nn as nn

def UD_constraint(classer):
    CL = classer.detach().cpu().numpy()
    N, K = CL.shape
    CL = CL.T
    r = np.ones((K, 1)) / K
    c = np.ones((N, 1)) / N
    CL **= 10
    inv_K = 1. / K
    inv_N = 1. / N
    err = 1e3
    _counter = 0
    epsilon = 1e-10 
    while err > 1e-2 and _counter < 75:
        denominator = CL @ c
        r = inv_K / (denominator + epsilon) 
        c_new = inv_N / (r.T @ CL).T
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1
    CL = CL.astype(np.float64)
    c = c.astype(np.float64)
    c = np.clip(c, a_min=-1e10, a_max=1e10)
    CL *= np.squeeze(c) 
    # CL *= np.squeeze(c)
    CL = CL.T
    CL *= np.squeeze(r)
    CL = CL.T
    try:
        argmaxes = np.nanargmax(CL, 0)
    except:
        argmaxes = np.argmax(CL, 0)
    newL = torch.LongTensor(argmaxes)
    return newL

def unite(view,hs,attention_net,p_net,p_sample,adaptive_weight):
    hs_tensor = torch.tensor([]).cuda()  

    # obtain global view feature
    for v in range(view):
        hs_tensor = torch.cat((hs_tensor, torch.mean(hs[v], 1).unsqueeze(1)), 1)      # d * v  
    hs_tensor = hs_tensor.t()  
    # process by the attention
    hs_atten = attention_net(hs_tensor, hs_tensor, hs_tensor)                         # v * 1
    # learn the view sampling distribution
    p_learn = p_net(p_sample)                                                         # v * 1              
    # regulatory factor                              
    r = hs_atten * p_learn
    s_p = nn.Softmax(dim=0)
    r = s_p(r)
    # adjust adaptive weight
    adaptive_weight = r * adaptive_weight
    # obtain fusion feature          
    fusion_feature = torch.zeros([hs[0].shape[0], hs[0].shape[1]]).cuda()  # （256,512）
    for v in range(view):
        fusion_feature = fusion_feature + adaptive_weight[v].item() * hs[v]
    return fusion_feature