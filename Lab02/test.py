from typing import Dict

from torchvision.transforms.functional import convert_image_dtype

import network
import utils
from dataset import BSDDataset


def test_drawing(folders_dict: Dict[str, str]):
    train_dataset = BSDDataset(folders_dict['train_dir'], folders_dict['coco_train_ann'])
    for i in range(3):
        image, target = train_dataset[i]
        utils.draw_target(image, target, show_image=True)


def test_network(folders_dict: Dict[str, str]):
    train_dataset = BSDDataset(folders_dict['train_dir'], folders_dict['coco_train_ann'])

    model = network.maskrcnn_resnet101_fpn()

    coco = train_dataset.get_coco()

    model.train()
    image, target = train_dataset[0]
    image_tensor = convert_image_dtype(image)
    # target = utils.ann_to_target(ann, coco)
    res = model([image_tensor], [target])

    model.eval()
    res = model([image_tensor])
    utils.draw_target(image, res[0], show_image=True)

    print("Done")
