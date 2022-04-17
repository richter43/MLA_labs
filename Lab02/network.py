from torchvision.models.detection.mask_rcnn import MaskRCNN

from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models.resnet import resnet101
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.ops import misc as misc_nn_ops


def maskrcnn_resnet101_fpn(
        progress=True, num_classes=2, pretrained_backbone=True, trainable_backbone_layers=None,
        **kwargs
):
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained_backbone, trainable_backbone_layers, 5, 3
    )

    anchor_sizes =((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    ag = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    backbone = resnet101(pretrained=pretrained_backbone, progress=progress, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = MaskRCNN(backbone, num_classes, rpn_anchor_generator=ag, min_size=1024, max_size=1024, **kwargs)

    return model
