import torchvision
from torchvision.transforms import PILToTensor, Compose, Resize
from pycocotools import mask as maskUtils
from torchvision.transforms.functional import InterpolationMode
import utils

class BSDDataset(torchvision.datasets.CocoDetection):
    def __init__(self, root: str, ann_file: str):
        transform = Compose([PILToTensor(), Resize((1024,1024))])
        # transform = Compose([PILToTensor()])
        transform_mask = Compose([Resize((1024,1024), interpolation=InterpolationMode.NEAREST)])
        # transform_mask = None

        target_transform = AnnToTarget(transform_mask)
        super().__init__(root, ann_file, transform=transform, target_transform=target_transform)
        target_transform.set_coco(self.coco)

    def get_coco(self):
        return self.coco

    # def __getitem__(self, index:int):
    #     image, target = super().__getitem__(index)
    #
    #     return pil_to_tensor(image), target

    def annToRLE(self, ann):
        """
        Modified version from pycocotools library due to the usage of a tensor instead of a
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        t = self.coco.imgs[ann['image_id'].item()]
        h, w = t['height'], t['width']
        segm = ann['segmentation']
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann)
        m = maskUtils.decode(rle)
        return m

class AnnToTarget:
    """Convert a ``PIL Image`` to a tensor of the same type. This transform does not support torchscript.

    Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
    """

    def __init__(self, transform_target=None):
        self.coco = None
        self.transform_target = transform_target


    def __call__(self, ann):

        target = utils.ann_to_target(ann, self.coco)
        if self.transform_target is not None:
            target['masks'] = self.transform_target(target['masks'])

        return target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def set_coco(self, coco):

        self.coco = coco
