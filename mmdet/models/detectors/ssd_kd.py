from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn as nn


@DETECTORS.register_module()
class SSDKD(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 hint_adapt=dict(type='hint')
                 ):
        super(SSDKD, self).__init__(
            backbone,
            neck,
            bbox_head,
            train_cfg,
            test_cfg,
            pretrained,)
        if 'neck-adapt' in hint_adapt.type:
            self.neck_adapt = []
            in_channels = hint_adapt.neck_in_channels
            out_channels = hint_adapt.neck_out_channels
            for i in range(len(in_channels)):
                self.neck_adapt.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels[i], out_channels[i], kernel_size=3, padding=1),
                        # nn.ReLU(),
                        nn.Sequential()
                    )
                )
            self.neck_adapt = nn.ModuleList(self.neck_adapt)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        cls_score, bbox_pred = outs
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        bbox_list = self.bbox_head.get_bboxes(*outs, img_metas)
        head_det = dict()
        head_det['cls_score'] = cls_score
        head_det['bbox_pred'] = bbox_pred
        head_det['neck'] = x
        head_det['proposals'] = bbox_list

        return losses, head_det

