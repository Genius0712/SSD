from torch import nn

from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head
from ssd.modeling.weight_orthogonality.layer_orthogonality_loss import LayerOrthogonalityPullLoss


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = build_box_head(cfg)
        self.weight_loss = LayerOrthogonalityPullLoss(self.backbone, weight=100.)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        detector_losses['weight_loss'] = self.weight_loss()
        if self.training:
            return detector_losses
        return detections
