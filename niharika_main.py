# Main File , Parser statements and model loading .

import argparse
import time
import torch
import numpy as np
import torch.optim as optim

#Custom Modules

from niharika_utils import getModel,toDevice,prepareData
# Fill it up!
# from niharika_loss import MonodepthLoss

# Plot parameters.
import matplotlib.pyplot as plt
import matplotlib as mpl


def returnArgs():
    parser=argparse.ArgumentParser(description='Pytorch Monodepth Estimation')

    parser.add_argument('data_dir',help='directory of data')
    parser.add_argument('val_data_dir',help='Validation Data path.')
    parser.add_argument('model_path',help='Path to teh trained model')
    parser.add_argument('output_dir',help='Path where all images are stored.')
    parser.add_argument('--input_chs',default=3)
    parser.add_argument('--input_height',default=256,help='fixed resize height for all images')
    parser.add_argument('--input_width',default=512,help='fixed resizeable width')
    parser.add_argument('--model',default='Resnet18')
    parser.add_argument('--pretrained',default=False)
    parser.add_argument('--mode',default='test',help='Choose between -train or -test')
    parser.add_argument('--epochs',default=50)
    parser.add_argument('--learningrate',default=1e-4)
    parser.add_argument('--batchSize',default=256)
    parser.add_argument('--adjustlr',default=False)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--doAug',default=True,help='To do augmentations or not!')
    parser.add_argument('--augParams',default=[0.8,1.2,0.5,2.0,0.8,1.2])
    parser.add_argument('--printImages',default=True)
    parser.add_argument('--numWorkers',default=4)
    parser.add_argument('--usemultipleGPU',default=True)


    args=parser.parse_args()
    return(args)

def adjustlr(optimizer,epoch,learningRate):
    if epoch >= 30 and epoch < 40:
        lr = learningRate / 2
    elif epoch >= 40:
        lr = learningRate / 4
    else:
        lr = learningRate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


# Consists of all details.
class Model:

    def __init__(self,args):
        self.args=args
        # Set up model
        self.device=args.device
        self.model=getModel(args.model,input_channels=args.input_chs,pretrained=args.pretrained)
        self.model=self.model.to(self.device)

        if args.usemultipleGPU:
            self.model=torch.nn.DataParallel(self.model)
        
        if args.mode=='train':
            # Do something.
            print('We arent training here..so no issues')

        else:
            self.model.load_state_dict(torch.load(args.model_path))
            args.augParams=None
            args.doAug=False
            args.batch_size=1

        # Loading the data..
        self.op_dir=args.output_dir
        self.input_hght=args.input_height
        self.input_width=args.input_width

        self.nimg,self.loader=prepareData(args.data_dir,args.mode,args.augParams,args.doAug,args.batchSize,(args.input_hgt,args.input_wdth,args.numWorkers))


        if 'cuda' in self.device:
            torch.cuda.synchronize()
        

        def train(self)
        

        # Only testing kabatii..
        def test(self):
            self.model.eval()
            disparities=np.zeros((self.nimg,self.input_height),dtype=np.float32)
            disparities_pp=np.zeros((self.n_img,self.input_height, self.input_width),dtype=np.float32)

            with torch.no_grad():

                for(i,data) in enumerate(self.loader):
                    data=toDevice(data,self.device)
                    left=data.squeeze() # Changes the numpy part , got it .
                    disps=self.model(left)
                    disp=disps[0][:,0,:,:].unsqueeze(1)
                    disparities[i]=disp[0].squeeze().cpu().numpy()
                    disparities_pp[i]=post_process_disparity(disp[0][:,0,:,:].cpu().numpy())
                
                np.save(self.output_dir+'/disparities.npy',disparities)
                np.save(self.output_dir+'/disparities_pp.npy',disparities_pp)
                print('Done with Testing')


def main(args):
    args = returnArgs()
    if args.mode == 'test':
        model_test = Model(args)
        model_test.test()

if __name__ == '__main__':
    main()



