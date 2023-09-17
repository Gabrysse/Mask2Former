import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import get_norm
from detectron2.modeling import BACKBONE_REGISTRY, Backbone

from timm.layers import convert_sync_batchnorm

from timm.models.layers import trunc_normal_, DropPath


class TimmBackbone(nn.Module):
    def __init__(self,
                model_name,
                pretrained,
                img_size,
                norm_layer='BN'
                ):
        super().__init__()

        assert model_name in timm.list_models(), f"{model_name} is not included in timm." \
                                                 f"Please use a model included in timm. " \
                                                 "Use timm.list_models() for the complete list."

        self.model = timm.create_model(model_name,
                                       pretrained=pretrained,
                                       features_only=True,
                                       img_size=img_size,
                                       pretrained_strict=False)

        if norm_layer == 'SyncBN':
            self.model = convert_sync_batchnorm(self.model)

        self.feature_stride = self.model.feature_info.reduction()
        self.feature_channels = self.model.feature_info.channels()

    def forward(self, x):
        o = self.model(x)
        out = {'res2': o[0],
               'res3': o[1],
               'res4': o[2],
               'res5': o[3]}

        return out


@BACKBONE_REGISTRY.register()
class D2timm(TimmBackbone, Backbone):
    def __init__(self, cfg, input_shape):
        model_name = cfg.MODEL.TIMMBACKBONE.MODEL_NAME
        pretrained = cfg.MODEL.TIMMBACKBONE.PRETRAINED
        norm = cfg.MODEL.TIMMBACKBONE.NORM
        img_size = (cfg.INPUT.CROP.SIZE[0], cfg.INPUT.CROP.SIZE[1])

        super().__init__(
            model_name,
            pretrained,
            img_size,
            norm,
        )

        self._out_features = cfg.MODEL.TIMMBACKBONE.OUT_FEATURES

        self._out_feature_strides = {
            "res2": self.feature_stride[0],
            "res3": self.feature_stride[1],
            "res4": self.feature_stride[2],
            "res5": self.feature_stride[3],
        }

        self._out_feature_channels = {
            "res2": self.feature_channels[0],
            "res3": self.feature_channels[1],
            "res4": self.feature_channels[2],
            "res5": self.feature_channels[3],
        }

