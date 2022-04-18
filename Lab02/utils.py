import argparse
import errno
import glob
import os
import shutil

import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.transforms.functional import pil_to_tensor
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from labelme2coco import get_coco_from_labelme_folder, save_json
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

from PIL import Image
from typing import List, Dict, Tuple

def makedirs(path: str) -> None:
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass


def create_dataset_mrcnn(labelme_folder, dst_dir):
    files = glob.iglob(os.path.join(labelme_folder, "*.jpg"))
    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, dst_dir)


def argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_type', help='split type: wear_dev_split | train_test_split | type_split', type=str,
                        default='train_test_split')
    parser.add_argument('--epochs', help='Number of epochs which the model will train for', type=int,
                        default=2000)
    parser.add_argument('--output_dir', help='output_dir', type=str, default='coco_output')
    parser.add_argument('--fiftyone', help='Setup and show fiftyone', action='store_true')
    opt = parser.parse_args()
    return opt


def segmask_list(ann_list):
    tmp_list = []
    for ann in ann_list:
        tmp_list.append(ann['segmentation'])

    segmasks = torch.tensor(tmp_list, dtype=torch.float)
    return segmasks


def bbox_list(ann_list):
    tmp_list = []
    for ann in ann_list:
        tmp_list.append(ann['bbox'])

    bboxes = torch.tensor(tmp_list, dtype=torch.float)
    return bboxes


def labelme_to_coco_annjson(orig_path: str, dst_file: str) -> None:
    coco_file = get_coco_from_labelme_folder(orig_path)
    save_json(coco_file.json, dst_file)

def draw_box_and_segment(image_tensor: torch.TensorType, ann_list :List[Dict[str, object]], coco: COCO, show_image : bool = False):

    result = image_tensor

    for ann in ann_list:
        mask = coco.annToMask(ann)

        bool_mask = torch.tensor(mask == 1, dtype=torch.bool)

        bbox = masks_to_boxes(bool_mask.unsqueeze(0))
        result = draw_segmentation_masks(result, bool_mask, colors=["white"], alpha=0.5)
        result = draw_bounding_boxes(result, bbox, colors=["black"], width=5)

    if show_image:
        show(result)
        plt.show()

    return result

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def ann_to_target(ann_list: List[Dict[str, object]], coco: COCO):

    target = {'boxes': torch.tensor(()), 'labels': torch.tensor(()), 'scores': torch.tensor(()), 'masks': torch.tensor(())}

    for ann in ann_list:
        target_box_format = cocoann_to_targetformat(ann['bbox'])
        target['boxes'] = torch.cat((target['boxes'], torch.tensor(target_box_format, dtype=torch.float).unsqueeze(0)), 0)
        target['labels'] = torch.cat((target['labels'], torch.tensor([ann['category_id'] + 1])), 0)
        target['scores'] = torch.cat((target['scores'], torch.tensor([1.0])), 0)

        mask = coco.annToMask(ann)

        bool_mask = torch.tensor(mask == [ann['category_id'] + 1], dtype=torch.uint8)
        bool_mask = bool_mask.unsqueeze(0).unsqueeze(0)

        target['masks'] = torch.cat((target['masks'], bool_mask), 0)

    target['labels'] = target['labels'].type(dtype=torch.int64)

    return target

def ann_to_target_tensor(ann_list: List[Dict[str, object]], dataset):

    target = {'boxes': torch.tensor(()), 'labels': torch.tensor(()), 'scores': torch.tensor(()), 'masks': torch.tensor(())}

    for ann in ann_list:
        target_box_format = cocoann_to_targetformat(ann['bbox'])
        target['boxes'] = torch.cat((target['boxes'], torch.tensor(target_box_format, dtype=torch.float).unsqueeze(0)), 0)
        target['labels'] = torch.cat((target['labels'], torch.tensor([ann['category_id'] + 1])), 0)
        target['scores'] = torch.cat((target['scores'], torch.tensor([1.0])), 0)

        mask = dataset.annToMask(ann)

        bool_mask = torch.tensor(mask == (ann['category_id'].item() + 1), dtype=torch.uint8)
        bool_mask = bool_mask.unsqueeze(0).unsqueeze(0)

        target['masks'] = torch.cat((target['masks'], bool_mask), 0)

    target['labels'] = target['labels'].type(dtype=torch.int64)

    return target

def cocoann_to_targetformat(bbox: List[int]):
    target_format = [bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2, bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
    return target_format

def draw_target(image: torch.TensorType, target: Dict[str, torch.TensorType], show_image: bool = False):

    result = image

    for i in range(len(target['labels'])):
        bbox = masks_to_boxes(target['masks'][i])
        bool_mask = target['masks'][i] >= 0.5
        result = draw_segmentation_masks(result, bool_mask, alpha=0.5)
        result = draw_bounding_boxes(result, bbox, width=5)

    if show_image:
        show(result)
        plt.show()

def pass_through_collate(batch):
    image, target = batch[0]

    image = image.to("cuda")
    target['boxes'] = target['boxes'].to("cuda")
    target['labels'] = target['labels'].to("cuda")
    target['scores'] = target['scores'].to("cuda")
    target['masks'] = target['masks'].to("cuda")

    return [image, target]

class MaskRCNNLossManager(object):
    def __init__(self):
        self.classifier_losses = 0
        self.box_reg_losses = 0
        self.mask_losses = 0
        self.objectness_losses = 0
        self.rpn_box_reg_losses = 0
        self.total = 0

    def add(self, res: Dict[str, torch.TensorType], batch_size: int):
        self.classifier_losses += res['loss_classifier'].item()
        self.box_reg_losses += res['loss_box_reg'].item()
        self.mask_losses += res['loss_mask'].item()
        self.objectness_losses += res['loss_objectness'].item()
        self.rpn_box_reg_losses += res['loss_rpn_box_reg'].item()
        self.total += batch_size

    def get_averages(self):
        return np.array([self.classifier_losses, self.box_reg_losses, self.mask_losses, self.objectness_losses, self.rpn_box_reg_losses])/self.total

