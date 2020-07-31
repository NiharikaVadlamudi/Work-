import os 
from  PIL import Image 
from torch.utils.data import Dataset


# How kitti data is structured and using it .
class KittiLoader(Dataset):
    def __init__(self,root,mode,transform=None):

        # Left-Right Images.
        leftDir=os.path.join(root,'image_02/data/')
        rightDir=os.path.join(root,'image_03/data/')

        # Corresponding paths.
        self.rightPaths=sorted([os.path.join(rightDir,fname) for fname in os.listdir(rightDir)])
        self.leftPaths=sorted([os.path.join(leftDir,fname) for fname in os.listdir(leftDir)])

        #Checking .
        assert(len(self.rightPaths)==len(self.leftPaths))

        self.transform=transform
        self.mode=mode


    # Len of the data.
    def __len__(self):
        return(len(self.leftPaths))

    # One particular image aquisition.
    def __getitem__(self,idx):
        leftImage=Image.open(self.leftPaths[idx]) #Its a list na , so simply pick.
        rightImage=Image.open(self.rightPaths[idx])

        if self.mode=='train':
            # One sample consists a left-right pair.
            sample={'left':leftImage,'right':rightImage}

            # Based on mode , ur transform will be chosen.
            if self.transform:
                sample=self.transform(sample)
                return(sample)
            else:
                return(sample)    
        
        else:
            if self.transform:
                leftImage=self.transform(leftImage)
            return(leftImage)

# Noice, I understood how to write a custom dataset today.





