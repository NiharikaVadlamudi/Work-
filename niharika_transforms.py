import torch
import torchvision.transforms as transforms
import numpy as np

# Based on mode , the kind of transforms we apply will differ.

'''
4 types of transforms : 
1.Resizing the image (because image size shld be fixed)
2.DoTest (No idea)
3.ToTensor -Convert image to tensor..
4.Random Flip to the image.
5.Augmentation to the image.

'''

def imageTransforms(mode='train',augParams=[0.8,1.2,0.5,2.0,0.8,1.2],doAug=True,transformations=None,size=(256,512)):

    if mode=='train':
        dataTransform=transforms.Compose([
            resizeImage(train=True,size=size),
            randomFlip(doAug),
            toTensor(train=True),
            augmentImagePair(augParams,doAug)
        ])

    elif mode=='test':
        dataTransform=transforms.Compose([
            resizeImage(train=False,size=size),
            toTensor(train=False),
            doTest(),
        ])
    else:
        dataTransform=transforms.Compose([])    
    
    return(dataTransform)


# Defining each part seperately .

class resizeImage(object):

    def __init__(self,train=True,size=(256,512)):
        self.train=train
        # Calling the inbuilt func only , lil added features.
        self.transform=transforms.Resize(size) 

    def __call__(self,sample):
        if self.train:
            leftImage=sample['left']
            rightImage=sample['right']

            newleftImage=self.transform(leftImage)
            newrightImage=self.transform(rightImage)

            sample={'left':newleftImage,'right':newrightImage}
        
        else:
            # Not a dictionary any more .
            leftImage=sample
            newleftImage=self.transform(leftImage)
            sample=newleftImage

class doTest(object):
    def __call__(self,sample):
        # Didn't understand what fully is happening here..
        newSample=torch.stack((sample,torch.flip(sample,[2])))
        return(newSample)

class toTensor(object):

    def __init__(self,train):
        self.train=train
        self.transform=transforms.ToTensor()
    
    def __call__(self,sample):

        if self.train:
            leftImage=sample['left']
            rightImage=sample['right']
            
            newRightImage=self.transform(rightImage)
            newLeftImage=self.transform(leftImage)

            sample={'left':newLeftImage,'right':newRightImage}

            return(sample)

        else:
            leftImage=sample
            sample=self.transform(leftImage)
        
        return(sample)


# Sample is a class object..so object is an input laga.
class randomFlip(object):

    def __init__(self,doAug):
        self.transform=transforms.RandomHorizontalFlip(p=1)
        self.doAug=doAug
    

    def __call__(self,sample):
        leftImage=sample['left']
        rightImage=sample['right']

        k=np.random.uniform(0,1,1)

        if self.doAug:
            if(k>0.5):
                flippedLeft=self.transform(rightImage)
                flippedRight=self.transform(leftImage)
                sample={'left':flippedLeft,'right':flippedRight}
            
        else:
            sample={'left': leftImage,'right':rightImage}
            
        return(sample)


class augmentImagePair(object):

    def __init__(self,augParams,doAug):

        self.doAug=doAug
        self.augParams=augParams

        # Settings.
        self.gammaLow=augParams[0]
        self.gammaHigh=augParams[1]
        self.brightnessLow=augParams[2]
        self.brightnessHigh=augParams[3]
        self.colorHigh=augParams[5]
        self.colorLow=augParams[4]
    

    def __call__(self,sample):

        leftImg=sample['left']
        rightImg=sample['right']

        p = np.random.uniform(0,1,1)

        if self.doAug:
            if(p>0.5):
                # Randomly Shift Gamma
                rgamma=np.random(self.gammaLow,self.gammaHigh)
                leftImg=leftImg**rgamma
                rightImg=rightImg**rgamma

                # Randomly Shift Brightness
                rbrightness=np.random(self.brightnessLow,self.brightnessHigh)
                leftImg=leftImg*rbrightness
                rightImg=rightImg*rbrightness

                # Randomly Shift Color
                randomColor=np.uniform(self.colorLow,self.colorHigh)
                for i in range(3):
                    leftImg[i,:,:]*=randomColor
                    rightImg[i,:,:]*=randomColor

                # Randomly shift saturation level.
                leftImg=torch.clamp(leftImg,0,1)
                rightImg=torch.clamp(rightImg,0,1)

            sample={'left':leftImg,'right':rightImg}

        else:
            sample={'left':leftImg,'right':rightImg}
        
        return(sample)









