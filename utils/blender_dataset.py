from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from albumentations import BboxParams, Normalize, Resize, PadIfNeeded, Compose
import math

class BlenderDataset(Dataset):

    def __init__(self, opt, split, transforms):
        super(BlenderDataset, self).__init__()

        self._opt = opt
        self._split = split
        self._transforms = transforms
        self._images_path = []
        self._keypoints_path = []
        dirlist = os.listdir(os.path.join(self._opt['path'], self._split))

        for dirname in dirlist:
            imgs = [os.path.join(self._opt['path'], self._split, dirname, x) for x in os.listdir(os.path.join(self._opt['path'], self._split, dirname)) if 'background' in x]
            self._images_path += imgs
            self._keypoints_path += [os.path.join(self._opt['path'], self._split, dirname, '{}_keypoints.txt'.format(dirname))] * len(imgs)
    
    def __len__(self):
        return len(self._images_path)

    def __getitem__(self, idx):
        img = Image.open(self._images_path[idx])
        img = np.array(img)

        # read keypoints
        # type - x - y
        keypoints = []
        with open(self._keypoints_path[idx], 'r') as f:
            c, x, y = f.readlines().strip().split('\n')
            keypoints.append([self._opt['classes mapping'][c], x, y])

        # coco format -> yolo format
        bbox_size = 32 # parametro configurabile
        h, w = img.shape[:2]
        bboxes = [[kp[1], kp[2], bbox_size, bbox_size] for kp in keypoints]
        normalized_bboxes = [[bbox[0] / w,
                              bbox[1] / h,
                              bbox[2] / w,
                              bbox[3] / h]
                              for bbox in bboxes]
            
        old_h = img.shape[0]
        old_w = img.shape[1]
        new_h = img.shape[0]
        new_w = img.shape[1]
        if not img.shape[1] % 32 == 0:
            new_w = old_w + (32 - old_w % 32)
            ratio = new_w / old_w
            new_h = int(old_h * ratio)
            self._transforms.append(Resize(new_h, new_w))

        if not img.shape[0] % 32 == 0:
            new_h = new_h + (32 - new_h % 32)
            self._transforms.append(PadIfNeeded(new_h, new_w))

        self._transforms.append(Normalize(p=1.0))
        
        aug = Compose(self._transforms, bbox_params=BboxParams(format='yolo', label_fields=['category_id']))
        res = aug(image=img, bboxes=normalized_bboxes, category_id=keypoints[:,0])

        img = res['image']
        bboxes = res['bboxes']
        classes_id = res['category_id']

        img = img.transpose(2,0,1)

        labels = [[0,
                   classes_id[i],
                   bbox[0],
                   bbox[1],
                   bbox[2],
                   bbox[3]] for i, bbox in enumerate(bboxes)]
         
        return torch.from_numpy(img), torch.from_numpy(labels), self._images_path[idx], None

    # this is called after __getitem__ (after an entire batch)
    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()

        # img is a tuple -> return a single 4d tensor
        # label is a tuple -> return a single list of labels
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes