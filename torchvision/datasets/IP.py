from PIL import Image
import os
import os.path
import numpy as np
import pickle
import scipy


from .vision import VisionDataset
from .utils import  download_and_extract_archive


class IndianPines(VisionDataset):
    """`
    Args:
        root (string): Root directory of dataset where directory
            ```` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    data_url = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
    tgt_url = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
    file_data = "Indian_pines_corrected.mat"
    file_tgt = "Indian_pines_gt.mat"
   
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(IndianPines, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if self.train:
            
        else:
            
            
        self.data = np.array(scipy.io.loadmat('/content/'+data_url.split('/')[-1])[data_url.split('/')[-1].split('.')[0].lower()])
        self.targets = np.array(scipy.io.loadmat('/content/'+label_url.split('/')[-1])[label_url.split('/')[-1].split('.')[0].lower()])
        self.data, self.targets = createImageCubes(self.data,self.targets, windowSize=7)
        self.data = self.data.transpose((0,3,1,2))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


    def download(self):
        download_and_extract_archive(self.data_url, self.root, filename=self.file_data)
        download_and_extract_archive(self.tgt_url, self.root, filename=self.file_tgt)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

