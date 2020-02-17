import os
import collections
import torch
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt

from torch.utils import data
#from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate


class rootLoader(data.Dataset):
    def __init__( self, root, split="train", is_transform=False, img_size=None, augmentations=None, img_norm=True, test_mode=False,):
        self.root = root
        self.split = split
        #self.img_size = [360, 480]
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.mean = np.array([122.34146, 132.07745, 131.84017])
        self.n_classes = 2
        self.files = collections.defaultdict(list)

        file_list = os.listdir(root + "/images")
        self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + "/images/" + img_name
        lbl_path = self.root + "/segmentation8/" + img_name

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int8)
        # Turn from 0 & -1 to 0 & 1
        lbl[lbl == -1] = 1

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        #img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


if __name__ == "__main__":
    local_path = "/home/greg/datasets/rootset"
    #augmentations = Compose([RandomRotate(10), RandomHorizontallyFlip()])

    dst = rootLoader(local_path, is_transform=True)
    bs=1
    trainloader = data.DataLoader(dst, batch_size=bs)

    print("\n**DATA**\n")
    print("length of trainloader:", len(trainloader))
    for i, data_samples in enumerate(trainloader):
        print("Shape of 0th:", data_samples[0].shape)
        print("shape of 1st:", data_samples[1].shape)
        """ DISPLAY IMAGES
        imgs, labels = data_samples
        imgs = imgs.numpy()[:,::-1,:,:]
        imgs = np.transpose(imgs, [0,2,3,1])
        f, axarr = plt.subplots(bs, 2,squeeze=False)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(labels.numpy()[j])
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()
        """
