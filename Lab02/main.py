import argparse
import os
from distutils.dir_util import copy_tree
from typing import Dict

import fiftyone as fo
import labelme2coco
import torch
from torch.utils.data import DataLoader

import utils
from Dataset.split_dataset import mogli
from dataset import BSDDataset
import network
from torchvision.transforms.functional import convert_image_dtype
from torch.utils.tensorboard import SummaryWriter


def main(args: argparse.Namespace):

    writer = SummaryWriter()

    folders_dict = preprocessing(args)
    # test_drawing(folders_dict)
    # test_network(folders_dict)

    train(args, folders_dict, writer)

def preprocessing(args: argparse.Namespace):
    # Defining a dictionary with all folders
    folders_dict = dataset_dict(args)
    folders_dict['labelme_train_folder']

    # Splitting dataset
    if not os.path.exists(folders_dict['output_dir']):
        mogli(args.split_type, args.output_dir)

    # Create directories
    utils.makedirs(folders_dict['train_dir'])
    utils.makedirs(folders_dict['val_dir'])

    # Creating training and validation datasets for mrcnn
    if not os.path.exists(folders_dict['train_dir']):
        utils.create_dataset_mrcnn(folders_dict['labelme_train_folder'], folders_dict['train_dir'])
    if not os.path.exists(folders_dict['val_dir']):
        utils.create_dataset_mrcnn(folders_dict['labelme_val_folder'], folders_dict['val_dir'])

    # Copy contents from the entirety of the folders
    if not os.path.exists(folders_dict['datalabel_dir']):
        copy_tree(folders_dict['labelme_train_folder'], folders_dict['datalabel_dir'])
        copy_tree(folders_dict['labelme_val_folder'], folders_dict['datalabel_dir'])

    # Create fiftyone dataset
    if args.fiftyone:
        utils.makedirs(folders_dict['datavis_dir'])

        # Save dataset annotations with standard name
        labelme2coco.convert(folders_dict['datalabel_dir'], folders_dict['fiftyone_dir'])
        os.rename(os.path.join(folders_dict['fiftyone_dir'], "dataset.json"),
                  os.path.join(folders_dict['fiftyone_dir'], "labels.json"))

        utils.create_dataset_mrcnn(folders_dict['datalabel_dir'], folders_dict['datavis_dir'])
        setup_fiftyone(folders_dict['fiftyone_dir'])

    # Create train coco object and export train coco json

    if not os.path.exists(folders_dict['coco_train_ann']):
        utils.labelme_to_coco_annjson(folders_dict['labelme_train_folder'], folders_dict['coco_train_ann'])
    if not os.path.exists(folders_dict['coco_val_ann']):
        utils.labelme_to_coco_annjson(folders_dict['labelme_val_folder'], folders_dict['coco_val_ann'])

    return folders_dict

def dataset_dict(args: argparse.Namespace):
    folders_dict = dict()
    folders_dict['dataset_dir'] = os.path.join(os.getcwd(), "Dataset")
    folders_dict['output_dir'] = os.path.join(folders_dict['dataset_dir'], args.output_dir)
    folders_dict['fiftyone_dir'] = os.path.join(folders_dict['dataset_dir'], "fiftyone")

    folders_dict['labelme_train_folder'] = os.path.join(folders_dict['output_dir'], "train")
    folders_dict['labelme_val_folder'] = os.path.join(folders_dict['output_dir'], "test")
    folders_dict['export_dir'] = os.path.join(folders_dict['output_dir'], "coco_json")

    folders_dict['train_dir'] = os.path.join(folders_dict['export_dir'], "images", "train")
    folders_dict['val_dir'] = os.path.join(folders_dict['export_dir'], "images", "val")
    folders_dict['datavis_dir'] = os.path.join(folders_dict['fiftyone_dir'], "data")

    folders_dict['datalabel_dir'] = os.path.join(folders_dict['dataset_dir'], "data_labels")
    folders_dict['cocoanns_dir'] = os.path.join(folders_dict['export_dir'], "annotations")

    folders_dict['coco_train_ann'] = os.path.join(folders_dict['cocoanns_dir'], "train.json")
    folders_dict['coco_val_ann'] = os.path.join(folders_dict['cocoanns_dir'], "val.json")

    return folders_dict


def setup_fiftyone(dataset_dir):
    name = "BSD"
    dataset_type = fo.types.COCODetectionDataset

    try:
        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=dataset_type,
            name=name,
        )
    except ValueError as e:
        dataset = fo.load_dataset(name)

    session = fo.launch_app(dataset)

    input("Press enter to continue")


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

def train(args: argparse.Namespace, folders_dict: Dict[str, str], writer: SummaryWriter):

    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = BSDDataset(folders_dict['train_dir'], folders_dict['coco_train_ann'])
    val_dataset = BSDDataset(folders_dict['val_dir'], folders_dict['coco_val_ann'])
    train_dataloder = DataLoader(train_dataset, batch_size=1, pin_memory=torch.cuda.is_available())
    val_dataloder = DataLoader(val_dataset, batch_size=1, pin_memory=torch.cuda.is_available())

    model = network.maskrcnn_resnet101_fpn()

    train_dataset = BSDDataset(folders_dict['train_dir'], folders_dict['coco_train_ann'])

    model.train()

    for epoch in range(args.epochs):
        loss_mgr = utils.MaskRCNNLossManager()
        for image_tensor, target in train_dataloder:
            # Collate adds another useless dimension, don't want to deal with this bs so this is the fastest solution possible
            target['boxes'] = target['boxes'][0]
            target['labels'] = target['labels'][0]
            target['scores'] = target['scores'][0]
            target['masks'] = target['masks'][0]
            image_tensor_float = convert_image_dtype(image_tensor)
            res = model(image_tensor_float, [target])
            loss_mgr.add(res, image_tensor.shape[0])

        class_loss_avg, box_loss_avg, mask_loss_avg, objns_loss_avg, rpn_box_reg_loss = loss_mgr.get_averages()
        writer.add_scalar('Loss/train/classifier', class_loss_avg, epoch)
        writer.add_scalar('Loss/train/box_reg', box_loss_avg, epoch)
        writer.add_scalar('Loss/train/mask', mask_loss_avg, epoch)
        writer.add_scalar('Loss/train/objectness', objns_loss_avg, epoch)
        writer.add_scalar('Loss/train/rpn_box_reg', rpn_box_reg_loss, epoch)

    writer.close()

if __name__ == "__main__":
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    os.environ.update({"QT_QPA_PLATFORM_PLUGIN_PATH":
                           "/home/foxtrot/.py_envs/MLA/lib/python3.8/site-packages/PyQt5/Qt5/plugins/xcbglintegrations/libqxcb-glx-integration.so"})
    args = utils.argparser()

    main(args)
