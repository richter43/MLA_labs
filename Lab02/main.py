import argparse
import logging
import os
from distutils.dir_util import copy_tree
from typing import Dict

import fiftyone as fo
import labelme2coco
import torch
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import convert_image_dtype

import network
import utils
from Dataset.split_dataset import mogli
from dataset import BSDDataset
import test

folders_dict = None
writer = None

def main(args: argparse.Namespace):
    global writer
    writer = SummaryWriter()

    folders_dict = preprocessing(args)
    logging.info("Successfully setup")
    # test.test_drawing(folders_dict)
    # test.test_network(folders_dict)

    train(args)


def preprocessing(args: argparse.Namespace):
    global folders_dict
    # Defining a dictionary with all folders
    dataset_dict(args)

    # Splitting dataset
    if not os.path.exists(folders_dict['output_dir']):
        mogli(args.split_type, args.output_dir)

    # Create directories
    utils.makedirs(folders_dict['train_dir'])
    utils.makedirs(folders_dict['val_dir'])

    # Creating training and validation datasets for mrcnn
    if len(os.listdir(folders_dict['train_dir'])) == 0:
        utils.create_dataset_mrcnn(folders_dict['labelme_train_folder'], folders_dict['train_dir'])
    if len(os.listdir(folders_dict['val_dir'])) == 0:
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
    global folders_dict
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


def train(args: argparse.Namespace):
    global folders_dict
    global writer

    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = BSDDataset(folders_dict['train_dir'], folders_dict['coco_train_ann'])
    val_dataset = BSDDataset(folders_dict['val_dir'], folders_dict['coco_val_ann'])
    train_dataloder = DataLoader(train_dataset, batch_size=1,
                                 collate_fn=utils.pass_through_collate if torch.cuda.is_available() else None)
    val_dataloder = DataLoader(val_dataset, batch_size=1,
                               collate_fn=utils.pass_through_collate if torch.cuda.is_available() else None)

    model = network.maskrcnn_resnet101_fpn().to(device)

    optim_fn = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    for epoch in range(args.epochs):
        loss_mgr = utils.MaskRCNNLossManager()
        for image_tensor, target in tqdm.tqdm(train_dataloder):
            image_tensor_float = convert_image_dtype(image_tensor)
            loss_dict = model([image_tensor_float], [target])
            loss_mgr.add(loss_dict, image_tensor.shape[0])

        if epoch % 2 == 0:
            logging.log(logging.INFO, "Epoch %d", epoch)

        optim_fn.zero_grad()
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optim_fn.step()


        class_loss_avg, box_loss_avg, mask_loss_avg, objns_loss_avg, rpn_box_reg_loss = loss_mgr.get_averages()
        writer.add_scalar('Loss/train/classifier', class_loss_avg, epoch)
        writer.add_scalar('Loss/train/box_reg', box_loss_avg, epoch)
        writer.add_scalar('Loss/train/mask', mask_loss_avg, epoch)
        writer.add_scalar('Loss/train/objectness', objns_loss_avg, epoch)
        writer.add_scalar('Loss/train/rpn_box_reg', rpn_box_reg_loss, epoch)
    writer.close()
    torch.save(model.state_dict(), args.model_output_path)



if __name__ == "__main__":
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    os.environ.update({"QT_QPA_PLATFORM_PLUGIN_PATH":
                           "/home/foxtrot/.py_envs/MLA/lib/python3.8/site-packages/PyQt5/Qt5/plugins/xcbglintegrations/libqxcb-glx-integration.so"})
    args = utils.argparser()

    main(args)
