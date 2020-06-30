from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from albumentations import BboxParams, Normalize, Resize, PadIfNeeded, Compose
import math

class BlenderDataset(Dataset):
    # def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
    #              cache_images=False, single_cls=False, stride=32, pad=0.0):
    
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
            keypoints.append([c, x, y])

        # coco format -> yolo format
        bbox_size = 32 # parametro configurabile
        h, w = img.shape[:2]
        bboxes = [[kp[1], kp[2], bbox_size, bbox_size] for kp in keypoints]
        normalized_bboxes = [[bbox[0] / w,
                              bbox[1] / h,
                              bbox[2] / w,
                              bbox[3] / h]
                              for bbox in bboxes]
            
        self._transforms.append(Normalize(p=1.0))
        
        aug = Compose(self._transforms, bbox_params=BboxParams(format='yolo', min_area=0, min_visibility=0, label_fields=['category_id']))
        res = aug(image=img, bboxes=normalized_bboxes, category_id=keypoints[:,0])

        img = res['image']
        bboxes = res['bboxes']
        classes_id = res['category_id']

        img = img.transpose(2,0,1)

        labels = [[0,
                   #self._opt['classes mapping'][i],
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