
import torch

from detectron2.layers import (
    ShapeSpec,
)
from detectron2.modeling import Backbone, BACKBONE_REGISTRY, FPN
from detectron2.modeling.backbone.fpn import LastLevelP6P7, LastLevelMaxPool

from layoutlmft.models.layoutlmv3 import LayoutLMv3Model
from transformers import AutoConfig

__all__ = [
    "layoutlmv3_vit_fpn_backbone",
]


class LMV3_Backbone(Backbone):
    """
    基于Lmv3的backbone
    """
    def __init__(self, name, out_features, drop_path, img_size, pos_type, model_kwargs,
                 config_path=None, image_only=False, cfg=None):
        super().__init__()
        # 可以理解为硬编码
        self._out_feature_strides = {"layer3": 4, "layer5": 8, "layer7": 16, "layer11": 32}
        self._out_feature_channels = {"layer3": 768, "layer5": 768, "layer7": 768, "layer11": 768}
        # 使用的transforms的自动加载config。
        config = AutoConfig.from_pretrained(config_path)
            # disable relative bias as DiT
        config.has_spatial_attention_bias = False
        config.has_relative_attention_bias = False
        self.backbone = LayoutLMv3Model(config, detection=True,
                                            out_features=out_features, image_only=image_only)# 这是执行了detection=True
        self.name=name
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        #论文中没有加入text 嵌入的输入，未来要尝试加上文本嵌入。
        return self.backbone.forward(
                input_ids=x["input_ids"] if "input_ids" in x else None,
                bbox=x["bbox"] if "bbox" in x else None,
                images=x["images"] if "images" in x else None,
                attention_mask=x["attention_mask"] if "attention_mask" in x else None,
                # output_hidden_states=True,
            )

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def build_VIT_backbone(cfg):
    """
    Create a VIT instance from config.

    Args:
        cfg: a detectron2 CfgNode

    Returns:
        A VIT backbone by lmv3w
    """
    # fmt: off
    name = cfg.MODEL.VIT.NAME
    out_features = cfg.MODEL.VIT.OUT_FEATURES
    drop_path = cfg.MODEL.VIT.DROP_PATH
    img_size = cfg.MODEL.VIT.IMG_SIZE
    pos_type = cfg.MODEL.VIT.POS_TYPE

    model_kwargs = eval(str(cfg.MODEL.VIT.MODEL_KWARGS).replace("`", ""))

    
    if cfg.MODEL.CONFIG_PATH != '':
        config_path = cfg.MODEL.CONFIG_PATH
    else:
        config_path = cfg.MODEL.WEIGHTS.replace('pytorch_model.bin', '')  # layoutlmv3 pre-trained models
        config_path = config_path.replace('model_final.pth', '')  # detection fine-tuned models


    return LMV3_Backbone(name, out_features, drop_path, img_size, pos_type, model_kwargs,
                        config_path=config_path, image_only=cfg.MODEL.IMAGE_ONLY, cfg=cfg)


@BACKBONE_REGISTRY.register()
def layoutlmv3_vit_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Create a VIT w/ FPN backbone.

    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_VIT_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS# 这个值应该是默认的。
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,# 对应要从bottom_up中取出的layers
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
