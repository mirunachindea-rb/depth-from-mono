import os
import cv2
import random
from random import uniform
import utils
import augmentations as augm

class Dataset:
    def __init__(
        self,
        mode,
        left_img_dir,
        right_img_dir=None,
        augmentation=False,
        ):

        """
        Dataset class
        Parameters:
            * mode: train or test
            * left_img_dir: left image directory
            * right_img_dir: right image directory
            * augmentation: whether or not to perform data augmentation (default: ``False``)
        """
    
        self.mode=mode

        self.ids_left=None
        self.ids_right=None
        self.left_fps=None
        self.right_fps=None

        if mode=='test':
            self.ids_left=self.get_ids(left_img_dir)
            self.left_fps=[os.path.join(left_img_dir, img_id) for img_id in self.ids_left]
        
        if mode=='train':
            self.ids_left=self.get_ids(left_img_dir)
            self.ids_right=self.get_ids(right_img_dir)
            self.left_fps=[os.path.join(left_img_dir, img_id) for img_id in self.ids_left]
            self.right_fps=[os.path.join(right_img_dir, img_id) for img_id in self.ids_right]

        self.augmentation=augmentation
        
    def __getitem__(self,i):
        # Test
        if self.mode=='test':
            # read data
            left=cv2.imread(self.left_fps[i])
            left=cv2.cvtColor(left,cv2.COLOR_BGR2RGB)
            left=utils.normalize_img(left)
            if self.augmentation:
                left=self.augment_img(left)
            return left,None
        
        # Train
        else:
            # read data
            left=cv2.imread(self.left_fps[i])
            left=cv2.cvtColor(left,cv2.COLOR_BGR2RGB)
            right=cv2.imread(self.right_fps[i])
            right=cv2.cvtColor(right,cv2.COLOR_BGR2RGB)
            left=utils.normalize_img(left)
            right=utils.normalize_img(right)
            
            # apply augmentations
            if self.augmentation:
                left,right=self.augment_pair(left,right)

            return left,right
    
    def __len__(self):
        return len(self.ids_left)
    
    def get_ids(self,directory):
        folders = os.listdir(directory)
        ids=[]
        for folder in folders:
            folder_directory=os.path.join(directory,folder)
            images=os.listdir(folder_directory)
            for image in images:
                image_in_dir=os.path.join(folder,image)
                ids.append(image_in_dir)
        return ids

    def augment_img(self, img):
        # randomly shift gamma
        img=augm.random_shift_gamma1(img)
        # randomly shift brightness
        img=augm.random_shift_brightness1(img)
        # randomly shift color
        img=augm.random_shift_color1(img)
        # random saturation and contrast
        img=augm.random_saturation1(img)
        # random scale
        img=augm.random_scale1(img)
        # random flip
        img=augm.random_flip1(img)
        return img
        
    def augment_pair(self, left_image, right_image):
        # randomly shift gamma
        left_image,right_image=augm.random_shift_gamma2(left_image,right_image)
        # randomly shift brightness
        left_image,right_image=augm.random_shift_brightness2(left_image,right_image)
        # randomly shift color
        left_image,right_image=augm.random_shift_color2(left_image,right_image)
        # random saturation and contrast
        left_image,right_image=augm.random_saturation2(left_image,right_image)
        # random scale
        left_image,right_image=augm.random_scale2(left_image,right_image)
        # random flip
        left_image,right_image=augm.random_flip2(left_image,right_image)
        return left_image, right_image
        