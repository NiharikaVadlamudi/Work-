import torch
import collections
import os
from torch.utils.data import DataLoader, ConcatDataset

# Additional Imports.
from niharika_resnet import Resnet18
from niharika_dataloader import KittiLoader
from niharika_transforms import imageTransforms

def toDevice(input,device):

    if torch.is_tensor(input):
        return(input.to(device=device))
    
    elif isinstance(input,str):
        return(input)
    
    elif isinstance(input,collections.Mapping):
        return {k: toDevice(sample, device=device) for k, sample in input.items()}    
        
    elif isinstance(input,collections.Sequence):
        return [toDevice(sample,device=device) for sample in input ]
    
    else :
        raise TypeError(f'Input must contain tensor,list or dict, found {type(input)}')



def getModel(model,input_chns=3,pretrained=False):
    outModel=Resnet18(input_chns)
    return(outModel)



def prepareData(data_dir,mode,augParams,doAug,batchSize,size,numWorkers):

    data_dirs=os.listdir(data_dir)
    dataTransform=imageTransforms(
        mode=mode,
        augParams=augParams,
        doAug=doAug,
        size=size
    )

    # Making sure dataset is ready.
    datasets=[KittiLoader(os.path.join(data_dir,dir),mode,transform=dataTransform) for dir in data_dirs]
    dataset=ConcatDataset(datasets)
    nimg=len(dataset)

    if mode=='train':
        loader=DataLoader(dataset,batch_size=batchSize,shuffle=True,num_workers=numWorkers,pin_memory=True)
    else:
        loader=DataLoader(dataset,batch_size=batchSize,shuffle=True,num_workers=numWorkers,pin_memory=True)
    

    return nimg,loader