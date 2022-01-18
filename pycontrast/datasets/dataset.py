from __future__ import print_function

import numpy as np
import torch
from torchvision import datasets
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFilter

class ImageFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, root, transform=None, target_transform=None,
                 two_crop=False, jigsaw_transform=None):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop
        self.jigsaw_transform = jigsaw_transform
        self.use_jigsaw = (jigsaw_transform is not None)
        self.num = self.__len__()

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path, target = self.imgs[index]
        image = self.loader(path)

        # # image
        if self.transform is not None:
            img = self.transform(image)
            if self.two_crop:
                img2 = self.transform(image)
                img = torch.cat([img, img2], dim=0)
        else:
            img = image

        # # jigsaw
        if self.use_jigsaw:
            jigsaw_image = self.jigsaw_transform(image)

        if self.use_jigsaw:
            return img, index, jigsaw_image
        else:
            return img, index


class TrainWaferDataset(Dataset):
    def __init__(self, dt_x, transform=None):
        self.dt_x = dt_x
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.fromarray(self.dt_x[index]).resize((224,224))
        #denoise
        #img = img.filter(ImageFilter.MedianFilter(size=3))
        if self.transform is not None:
            #aug twice to get 2 version of the same img
            img1 = self.transform(img)
            img2 = self.transform(img)
        return torch.cat([img1, img2], dim=0)*255/2.0, index


    def __len__(self):
        return self.dt_x.shape[0]

class TrainWaferDataset_1(Dataset):
    def __init__(self, dt_x, transform=None):
        self.dt_x = dt_x
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.fromarray(self.dt_x[index]).resize((224,224))
        #denoise
        img_2 = img.filter(ImageFilter.MedianFilter(size=3))
        if self.transform is not None:
            #aug twice to get 2 version of the same img
            img1 = self.transform(img)
            img2 = self.transform(img_2)
        return torch.cat([img1, img2], dim=0)*255/2.0, index


    def __len__(self):
        return self.dt_x.shape[0]

class TestWaferDataset(Dataset):
    def __init__(self, dt_x, dt_y, transform=None):
        self.dt_x = dt_x
        self.dt_y = dt_y
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.fromarray(self.dt_x[index]).resize((224,224))
        #denoise
        #img = img.filter(ImageFilter.MedianFilter(size=3))
        if self.transform is not None:
            #aug twice to get 2 version of the same img
            img1 = self.transform(img)
        lbl = self.dt_y[index]
        return img1*255/2.0, torch.tensor(lbl, dtype=torch.int64)


    def __len__(self):
        return self.dt_x.shape[0]



class TrainWaferDataset_2(Dataset):
    def __init__(self, dt_x, transform=None):
        self.dt_x = dt_x
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img1 = Image.fromarray(self.dt_x[index]).resize((224,224))

        img = np.array(img1)
        #denoise
        # img_2 = img.filter(ImageFilter.MedianFilter(size=3))
        n_grid = 3
        img_size = 244
        crop_size = 64
        grid_size = int(img_size / n_grid)
        side = grid_size-crop_size
        yy, xx = np.meshgrid(np.arange(n_grid), np.arange(n_grid))
        yy_ = np.reshape(yy * self.grid_size, (n_grid * n_grid,))
        xx_ = np.reshape(xx * self.grid_size, (n_grid * n_grid,))
        r_x = np.random.randint(0, side + 1, n_grid * n_grid)
        r_y = np.random.randint(0, side + 1, n_grid * n_grid)
        crops = []
        for i in range(n_grid * n_grid):
            crops.append(img[xx_[i] + r_x[i]: xx_[i] + r_x[i] + crop_size,
                         yy_[i] + r_y[i]: yy_[i] + r_y[i] + crop_size])
        # crops = [Image.fromarray(crop) for crop in crops]
        
        img1 = torch.from_numpy(np.array(img1))
        img2 = torch.from_numpy(np.array(crops))
        img2 = img2.view(244,244)
        
        return torch.cat([img1, img2], dim=0)/2.0, index


    def __len__(self):
        return self.dt_x.shape[0]

    def __init__(self, n_grid=3, img_size=255, crop_size=64):
        self.n_grid = n_grid
        self.img_size = img_size
        self.crop_size = crop_size
        self.grid_size = int(img_size / self.n_grid)
        self.side = self.grid_size - self.crop_size

        yy, xx = np.meshgrid(np.arange(n_grid), np.arange(n_grid))
        self.yy = np.reshape(yy * self.grid_size, (n_grid * n_grid,))
        self.xx = np.reshape(xx * self.grid_size, (n_grid * n_grid,))

    def __call__(self, img):
        r_x = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        r_y = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        img = np.asarray(img, np.uint8)
        crops = []
        for i in range(self.n_grid * self.n_grid):
            crops.append(img[self.xx[i] + r_x[i]: self.xx[i] + r_x[i] + self.crop_size,
                         self.yy[i] + r_y[i]: self.yy[i] + r_y[i] + self.crop_size, :])
        crops = [Image.fromarray(crop) for crop in crops]
        return crops


