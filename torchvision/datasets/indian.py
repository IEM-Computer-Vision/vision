from PIL import Image
import os
import os.path
import numpy as np
import pickle
import scipy


from .vision import VisionDataset
from .utils import  download_and_extract_archive

def padWithZeros(X, margin=2):

    ## From: https://github.com/gokriznastic/HybridSN/blob/master/Hybrid-Spectral-Net.ipynb
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):

     ## From: https://github.com/gokriznastic/HybridSN/blob/master/Hybrid-Spectral-Net.ipynb
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype=np.uint8)
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype=np.uint8)
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

class IndianPines(VisionDataset):
    
    data_url = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
    label_url = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
    file_data = "Indian_pines_corrected.mat"
    file_tgt = "Indian_pines_gt.mat"
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False,split=0.95):

        super(IndianPines, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        
        self.data = np.array(scipy.io.loadmat(root+data_url.split('/')[-1])[data_url.split('/')[-1].split('.')[0].lower()])
        self.targets = np.array(scipy.io.loadmat(root+label_url.split('/')[-1])[label_url.split('/')[-1].split('.')[0].lower()])
        self.data, self.targets = createImageCubes(self.data,self.targets, windowSize=7)
        self.data = self.data.transpose((0,3,1,2))

        length = self.data.shape[0]
        if self.train:
            self.data,self.targets = self.data[:int(np.ceil(split*length)),:,:,:],self.targets[:int(np.ceil(split*length))]
            
        else:
            self.data,self.targets = self.data[:int(-np.ceil(split*length)),:,:,:],
                                                    self.targets[:int(-np.ceil(split*length))]
            
            
            
            
            
        
    def __getitem__(self, idx):
      
      return self.data[idx,:,:,:] , self.targets[idx]
    
    def __len__(self):
        return self.data.shape[0]



    def download(self):
        download_and_extract_archive(self.data_url, self.root, filename=self.file_data)
        download_and_extract_archive(self.label_url, self.root, filename=self.file_tgt)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

