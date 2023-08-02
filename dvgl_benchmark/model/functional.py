
import math
import torch
import torch.nn.functional as F

def sare_ind(query, positive, negative):
    '''all 3 inputs are supposed to be shape 1xn_features'''
    dist_pos = ((query - positive)**2).sum(1)
    dist_neg = ((query - negative)**2).sum(1)
    
    dist = - torch.cat((dist_pos, dist_neg))
    dist = F.log_softmax(dist, 0)
    
    #loss = (- dist[:, 0]).mean() on a batch
    loss = -dist[0]
    return loss

def sare_joint(query, positive, negatives):
    '''query and positive have to be 1xn_features; whereas negatives has to be
    shape n_negative x n_features. n_negative is usually 10'''
    # NOTE: the implementation is the same if batch_size=1 as all operations
    # are vectorial. If there were the additional n_batch dimension a different
    # handling of that situation would have to be implemented here.
    # This function is declared anyway for the sake of clarity as the 2 should
    # be called in different situations because, even though there would be
    # no Exceptions, there would actually be a conceptual error.
    return sare_ind(query, positive, negatives)

def mac(x):
    return F.adaptive_max_pool2d(x, (1,1))

def spoc(x):
    return F.adaptive_avg_pool2d(x, (1,1))

def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

def rmac(x, L=3, eps=1e-6):
    ovr = 0.4 # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension
    W = x.size(3)
    H = x.size(2)
    w = min(W, H)
    # w2 = math.floor(w/2.0 - 1)
    b = (max(H, W)-w)/(steps-1)
    (tmp, idx) = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) # steps(idx) regions for long dimension
    # region overplus per dimension
    Wd = 0;
    Hd = 0;
    if H < W:  
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1
    v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)
    for l in range(1, L+1):
        wl = math.floor(2*w/(l+1))
        wl2 = math.floor(wl/2 - 1)
        if l+Wd == 1:
            b = 0
        else:
            b = (W-wl)/(l+Wd-1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l-1+Wd+1))*b) - wl2 # center coordinates
        if l+Hd == 1:
            b = 0
        else:
            b = (H-wl)/(l+Hd-1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l-1+Hd+1))*b) - wl2 # center coordinates
        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:,:,(int(i_)+torch.Tensor(range(wl)).long()).tolist(),:]
                R = R[:,:,:,(int(j_)+torch.Tensor(range(wl)).long()).tolist()]
                vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                v += vt
    return v

