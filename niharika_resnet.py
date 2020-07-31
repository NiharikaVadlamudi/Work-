from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

class conv(nn.Module):
    def __init__(self,input_chn,output_chn,kernel_size,stride):
        super(conv,self).__init__()
        self.kernel_size=kernel_size
        self.conv_base=nn.Conv2d(input_chn,output_chn,kernel_size=kernel_size,stride=stride)
        self.normalize=nn.BatchNorm2d(output_chn)
    
    def forward(self,x):
        p=int((np.floor(self.kernel_size-1)/2))
        padding=(p,p,p,p)
        x=self.conv_base(F.pad(x,padding))
        x=self.normalize(x)
        return(F.elu(x,inplace=True))


class convBlock(nn.Module):
    def __init__(self,input_chn,output_chn,kernel_size,stride):
        super(convBlock,self).__init__()
        # Stride=1
        self.conv1=conv(input_chn,output_chn,kernel_size,1)
        # Stride = 2
        self.conv2=conv(output_chn,output_chn,kernel_size,2)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        return(x)

class maxPool(nn.Module):
    def __init__(self,kernel_size):
        super(maxPool,self).__init__()
        self.kernel_size=kernel_size
    
    def forward(self,x):
        p = int(np.floor((self.kernel_size-1)/2))
        padding=(p,p,p,p)
        return(F.max_pool2d(F.pad(x,padding)),self.kernel_size,2)


class resConvBasic(nn.Module):
    # Resnet 18 only .
    def __init__(self,input_chns,output_chns,stride):
        super(resConvBasic,self).__init__()
        self.output_chns=output_chns
        self.stride=stride

        self.conv1=conv(input_chns,output_chns,3,stride)
        self.conv2=conv(output_chns,output_chns,3,1)
        self.conv3=nn.Conv2d(input_chns,output_chns,kernel_size=kernel_size,stride=stride)
        self.normalise=nn.BatchNorm2d(output_chns)
    
    def forward(self,x):
        doProj=True
        shortcut=[]
        x_out=self.conv1(x)
        x_out=self.conv2(x_out)
        if doProj:
            shortcut=self.conv3(x)
        else:
            shortcut=self.conv3(x)
        
        return(F.elu(self.normalize(x_out+shortcut),inplace=True))


class upConv(nn.Module):
    def __init__(self,input_chns,output_chns,kernel_size,scale):
        super(upConv,self).__init__()
        self.scale=scale
        self.conv1=conv(input_chns,output_chns,kernel_size,1)
    
    def forward(self,x):
        x=nn.functional.interpolate(x,scale_factor=self.scale,mode='bilinear',align_corners=True)
        x=self.conv1(x)
        return(x)


class getDisparity(nn.Module):
    def __init__(self,input_chns):
        super(getDisparity,self).__init__()
        self.conv1=nn.Conv2d(input_chns,2,kernel_size=3,stride=1)
        self.normalize=nn.BatchNorm2d(2)
        self.sigmoid=torch.nn.Sigmoid()
    
    def forward(self,x):
        p =1 
        padding=(p,p,p,p)
        x=self.conv1(F.pad(x,padding))
        x=self.normalize(x)
        return(0.3*self.sigmoid(x))



def resBlockBasic(input_chns,output_chns,num_blocks,stride):
    layers=[]
    layers.append(resConvBasic(input_chns,output_chns,stride))
    for _ in range(1,num_blocks):
        layers.append(resConvBasic(input_chns,output_chns,1))
    return(nn.Sequential(*layers))



class Resnet18(nn.Module):

    def __init__(self,input_chns):
        super(Resnet18,self).__init__()
        #Encoder 
        self.conv1=conv(input_chns,64,7,2)
        self.pool1=maxPool(3)

        self.conv2=resBlockBasic(64,64,2,2)
        self.conv3=resBlockBasic(64,128,2,2)
        self.conv4=resBlockBasic(128,256,2,2)
        self.conv5=resBlockBasic(256,512,2,2)

        #Decoder
        self.upconv6=upConv(512,512,3,2)
        self.iconv6=conv(256+512,512,3,1)

        self.upconv5=upConv(512,256,3,2)
        self.iconv5=conv(256+128,256,3,1)

        # First Disparity Scale
        self.upconv4=upConv(256,128,3,2)
        self.iconv4=conv(64+128,128,3,1)
        self.disp_4layer=getDisparity(128)

        # Second Disparity Scale 
        self.upconv3=upConv(128,64,3,2)
        self.iconv3=conv(64+64+2,64,3,1)
        self.disp_3layer=getDisparity(64)

        # Third Disparity Scale
        self.upconv2=upConv(64,32,3,2)
        self.iconv2=conv(64+32+2,32,3,1)
        self.disp_2layer=getDisparity(32)

        # Fourth Disparity Scale
        self.upconv1=upConv(32,16,3,2)
        self.iconv2=conv(16+2,16,3,1)
        self.disp_1layer=getDisparity(16)


        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self,x):
        # Encoder 
        x1=self.conv1(x)
        x_pool1=self.pool1(x1)
        x2=self.conv2(x_pool1)
        x3=self.conv3(x2)
        x4=self.conv4(x3)
        x5=self.conv5(x4)

        # Skips .
        skip1=x1
        skip2=x_pool1
        skip3=x2
        skip4=x3
        skip5=x4


        #Decoder.
        upconv6=self.upconv6(x5)
        concat6=torch.cat((upconv6,skip5),1)
        iconv6=self.iconv(concat6)

        upconv5=self.upconv5(iconv6)
        concat5=torch.cat((upconv5,skip4),1)
        iconv5=self.iconv5(concat5)

        upconv4=self.upconv4(iconv5)
        concat4=torch.cat((upconv4,skip3),1)
        iconv4=self.iconv4(concat4)
        self.disp4=self.disp_4layer(iconv4)
        self.udisp4=nn.functional.interpolate(self.disp4,scale_factor=2,mode='bilinear',align_corners=True)

        
        
        upconv3=self.upconv3(iconv4)
        concat3=torch.cat((upconv3,skip2,self.udisp4),1)
        iconv3=self.iconv3(concat3)
        self.disp3=self.disp_3layer(iconv3)
        self.udisp3=nn.functional.interpolate(self.disp3,scale_factor=2,mode='bilinear',align_corners=True)


        upconv2=self.upconv2(iconv3)
        concat2=torch.cat((upconv2,skip1,self.udisp3),1)
        iconv2=self.iconv2(concat2)
        self.disp2=self.disp_2layer(iconv2)
        self.udisp2=nn.functional.interpolate(self.disp2,scale_factor=2,mode='bilinear',align_corners=True)

        upconv1=self.upconv1(iconv2)
        concat1=torch.cat((upconv1,self.udisp2),1)
        iconv1=self.iconv1(concat1)
        self.disp1=self.disp_1layer(iconv1)

        # 4 layers predicted 
        return(self.disp1,self.disp2,self.disp3,self.disp4)











        



        













