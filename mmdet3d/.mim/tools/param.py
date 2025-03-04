import torch
from collections import defaultdict
import sys
def part(state_dict, keys=None):
    float32 = 0
    int64 = 0
    for k,v in state_dict.items():
        if keys is not None and k in keys:
            size = v.flatten().shape[0]
            if v.dtype == torch.float32:
                float32+=size
            if v.dtype == torch.int64:
                int64+=size
    return (float32*4 + int64*8)/1024/1024
def calc_param(state_dict):
    float32 = 0
    int64 = 0
    for k,v in state_dict.items():
        size = v.flatten().shape[0]
        if v.dtype == torch.float32:
            float32+=size
        if v.dtype == torch.int64:
            int64+=size
    return (float32*4 + int64*8)/1024/1024

def full_part(state_dict):
    float32 = 0
    int64 = 0
    each = defaultdict(int)
    for k,v in state_dict.items():
        size = v.flatten().shape[0]
        # if v.dtype == torch.float32:
        #     size*=4
        if v.dtype == torch.int64:
            size*=2
        each[k.split('.')[0]]+=size
    return each
    

if __name__=='__main__':
    m=torch.load(sys.argv[1])
    # p=full_part(m['state_dict'])
    p=full_part(m)
    # print(p)
    total = 0
    for k,v in p.items():
        print(k,':',v//1000000)
        total+=v
    print('total:',total//1000000)
    