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
        self._bbox_path = []
        self._class_path = []
        dirlist = os.listdir(os.path.join(self._opt['path'], self._split))

        for dirname in dirlist:
            imgs = [os.path.join(self._opt['path'], self._split, dirname, x) for x in os.listdir(os.path.join(self._opt['path'], self._split, dirname)) if 'background' in x]
            self._images_path += imgs
            self._bbox_path += [os.path.join(self._opt['path'], self._split, dirname, '{}_objectbb.txt'.format(dirname))] * len(imgs)
            self._class_path +=[os.path.join(self._opt['path'], self._split, dirname, '{}_objects.txt'.format(dirname))] * len(imgs)

    def __len__(self):
        return len(self._images_path)

    def __getitem__(self, idx):
        img = Image.open(self._images_path[idx])
        img = np.array(img)

        # # read bboxes gt
        # bboxes = []
        # with open(self._bbox_path[idx], 'r') as f:
        #     x, y, w, h = map(int, f.readline().strip().split(' '))
        #     bboxes.append([x, y, w, h])

        # # read classes gt
        # classes_id = []
        # with open(self._class_path[idx], 'r') as f:
        #     class_id = self._opt['classes mapping'][(f.readline().strip())]
        #     classes_id.append(class_id)

        # read keypoints
        # type - x - y
        keypoints = []

        # coco format -> yolo format
        bbox_size = 32 # parametro esterno---------------------------------------------
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


class BlenderDataBunch(object):

    def __init__(self, opt, transforms):
        self.train_dataset = BlenderDataset(opt, 'train', transforms)
        self.val_dataset = BlenderDataset(opt, 'val', transforms)
        self.test_dataset = BlenderDataset(opt, 'test', transforms)

        self.dataloaders = {
            'train': DataLoader(self.train_dataset, shuffle=True, batch_size=opt['batch size'], num_workers=opt['num workers']),
            'val': DataLoader(self.val_dataset, shuffle=False, batch_size=opt['batch size'], num_workers=opt['num workers']),
            'test': DataLoader(self.test_dataset, shuffle=False, batch_size=opt['batch size'], num_workers=opt['num workers']),
        }

# class DataBunch(object):

#     def __init__(self, opt, dataset, transforms, val=True, test=True):
#         self.dataloaders = dict()

#         self.train_dataset = dataset(opt, 'train', transforms)
#         self.dataloaders['train'] = DataLoader(self.train_dataset, shuffle=True, batch_size=opt['batch size'], num_workers=opt['num workers'])
        
#         if val:
#             self.val_dataset = dataset(opt, 'val', [])
#             self.dataloaders['val'] = DataLoader(self.val_dataset, shuffle=False, batch_size=opt['batch size'], num_workers=opt['num workers'])
        
#         if test:
#             self.test_dataset = dataset(opt, 'test', [])
#             self.dataloaders['test'] = DataLoader(self.test_dataset, shuffle=False, batch_size=opt['batch size'], num_workers=opt['num workers'])
            