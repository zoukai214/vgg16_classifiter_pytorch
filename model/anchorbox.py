import numpy as np


# class AnchorBox(nn.Module):
#     def __init__(self,)

def anchorbox(base_size=16,ratios=[0.5,1,2],anchor_scales=[8,16,32]):
    py = base_size/2.
    px = base_size/2.
    anchor_base = np.zeros((len(ratios)*len(anchor_scales),4),dtype = np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size*anchor_scales[j]*np.sqrt(ratios[i])
            w = base_size*anchor_scales[j]*np.sqrt(1./ratios[i])

            index = i*len(anchor_scales)+j
            anchor_base[index,0] = py-h/2.
            anchor_base[index,1] = px-w/2.
            anchor_base[index,2] = py+h/2.
            anchor_base[index,3] = px+w/2.
    return anchor_base

print(anchorbox(base_size=16,ratios=[0.5,1,2],anchor_scales=[8,16,32]))