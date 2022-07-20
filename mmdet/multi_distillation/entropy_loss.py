import torch
import torch.nn.functional as F
import torch.nn as nn
from mmdet.ops import nms
from .entro_loss import entro_beta_loss, Mahalanobis


class proposal_dis():
    def __init__(self, dis, index):
        self.dis = dis
        self.index = index


class entropy_loss(nn.Module):
    def __init__(self,
                 img,
                 img_metas,
                 gt_bboxes,
                 gt_labels,
                 feat_t,
                 proposal_list_t,
                 feat_adapt_s,
                 proposal_list_s
                 ):
        """
            cls_s{tuple()}:tuple: num of rois=1024;
                           tensor: class=21;
            bbox_s{tuple()}: tuple: num of rois=1024;
                             tensor: class * place=80;
            feat_adapt_s{tuple(tensor)}: tuple: level=5;
                                         tensor: batch_size, channel, H, W (H, W get smaller while level get larger)
            proposal_list{tuple(tensor)}: tuple: len=2;
                                          tensor: rois_num, x1, y1, x2, y2, score
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]

            return
                mask_batch{tuple(tensor)}: tuple: level=5;
                                           tensor: batch_size, H=feat.H, W=feat.W
        """
        super(entropy_loss, self).__init__()
        self.img = img
        self.neck_entro_mask = self.get_fpn_entropy_mask(img, img_metas, gt_bboxes, gt_labels, feat_t, proposal_list_t,
                                                         feat_adapt_s, proposal_list_s)

    def treat_target_poposal(self, rois_target, feat_t, feat_s, b, rate=0.6):
        """
        return: index list contain which rois to keep
        """
        with torch.no_grad():
            img = self.img._data
            final_keep_mask = []
            delta_list = []
            # we check the rois target in final level of feat
            # b = feat_s[-1].size(2)
            c = feat_s[-1].size(1)
            h = feat_s[-1].size(2)
            w = feat_s[-1].size(3)
            for i in range(int(rois_target.size(0))):
                mask_per_img = torch.zeros([1, c, h, w], dtype=torch.double).cuda()
                x1 = rois_target[i][1] / img[0].size(2) * h
                y1 = rois_target[i][0] / img[0].size(3) * w
                x2 = rois_target[i][3] / img[0].size(2) * h
                y2 = rois_target[i][2] / img[0].size(3) * w
                if int(x2) > int(h) or int(y2) > int(w):
                    continue
                mask_per_img[:, :, int(x1):int(x2), int(y1):int(y2)] += 1
                norms = max(1.0, mask_per_img.sum())
                # delta_list.append((torch.pow(feat_t[-1][b:b+1, ...] - feat_s[-1][b:b+1, ...], 2) * mask_per_img).sum() / norms)
                mah_dis = Mahalanobis(feat_s[-1][b:b+1, ...], feat_t[-1][b:b+1, ...], mask_per_img)
                dis_iter = proposal_dis(mah_dis, i)
                delta_list.append(dis_iter)
            # average = sum(delta_list) / max(1, len(delta_list))
            # index_ = [i for i in range(len(delta_list)) if torch.gt(delta_list[i], average * rate)]
            B = sorted(delta_list, key=lambda x:x.dis)
            index_ = [B[i].index for i in range(int(len(B)*(1 - rate)), len(B))]
        return index_

    def get_fpn_entropy_mask(self,
                             img,
                             img_metas,
                             gt_bboxes,
                             gt_labels,
                             feat_t,
                             proposal_list_t,
                             feat_adapt_s,
                             proposal_list_s):
        with torch.no_grad():
            batch_size = len(proposal_list_s)
            rois_target_list = []
            rois_index_list = []
            for b in range(batch_size):
                proposals_s = proposal_list_s[b].detach()
                proposals_t = proposal_list_t[b].detach()
                proposals_s = proposals_s[(proposals_s[:, -1] > 0.85)]
                proposals_t = proposals_t[(proposals_t[:, -1] > 0.85)]
                rois_s, ins_s = nms(proposals_s, 0.3)
                rois_t, ins_t = nms(proposals_t, 0.3)
                # rois_num = (rois_s.size(0), rois_t.size(0))
                rois_target = torch.cat((rois_t, rois_s), dim=0)
                rois_target, _ = nms(rois_target, 0.6)
                index_ = self.treat_target_poposal(rois_target, feat_t, feat_adapt_s, b, 0.6)
                rois_target_list.append(rois_target)
                rois_index_list.append(index_)

            img = img._data
            mask_list = []
            for level in range(len(feat_t)):
                # batch_size = feat_adapt_s[level].size(0)
                c = feat_adapt_s[level].size(1)
                h = feat_adapt_s[level].size(2)
                w = feat_adapt_s[level].size(3)
                # mask_per_img = torch.zeros([h, w], dtype=torch.double).cuda()
                mask_batch = []
                for b in range(batch_size):
                    mask_per_img = torch.zeros([h, w], dtype=torch.double).cuda()
                    rois_target = rois_target_list[b]
                    index_ = rois_index_list[b]
                    for i in range(int(rois_target.size(0))):
                        if i not in index_ and len(index_) != 0:
                            continue
                        x1 = rois_target[i][1] / img[0].size(2) * h
                        y1 = rois_target[i][0] / img[0].size(3) * w
                        x2 = rois_target[i][3] / img[0].size(2) * h
                        y2 = rois_target[i][2] / img[0].size(3) * w
                        if int(x2) > int(h) or int(y2) > int(w):
                            import pdb; pdb.set_trace()
                            continue
                        mask_per_img[int(x1):int(x2), int(y1):int(y2)] += 1
                    mask_per_img = (mask_per_img > 0).double()
                    mask_batch.append(mask_per_img.detach())
                mask_list.append(torch.stack(mask_batch, dim=0))
        return mask_list

    def forward(self, model, feat_t, feat_s, kd_cfg, kd_decay=1., kd_warm=dict()):
        EntroBetaLoss = entro_beta_loss(feat_s[0].size(0), feat_s[0].size(1), init_num=1.00001).cuda()
        loss_entro_beta = torch.Tensor([0]).cuda()
        losskd_entro = torch.Tensor([0]).cuda()
        losskd_entro_back = torch.Tensor([0]).cuda()
        for i, _neck_feat in enumerate(feat_s):
            entro_mask = self.neck_entro_mask[i]
            entro_mask = entro_mask.unsqueeze(1).repeat(1, _neck_feat.size(1), 1, 1)
            norms_entro = max(1.0, entro_mask.sum() * 2)
            # if 'neck-adapt' in kd_cfg.type and hasattr(model.module, 'neck_adapt'):
            #     neck_feat_adapt = model.module.neck_adapt[i](_neck_feat)
            # else:
            #     neck_feat_adapt = _neck_feat
            neck_feat_adapt = model.module.neck_adapt[i](_neck_feat)

            if 'L1' in kd_cfg.type:
                diff = torch.abs(neck_feat_adapt - feat_t[i])
                loss = torch.where(diff < 1.0, diff, diff ** 2)
                losskd_entro += (loss * entro_mask).sum() / norms_entro
            elif 'Div' in kd_cfg.type:
                losskd_entro += (torch.pow(1 - neck_feat_adapt / (feat_t[i] + 1e-8), 2)
                                 * entro_mask).sum() / norms_entro
            elif 'neck-decouple' in kd_cfg.type:
                losskd_entro += (torch.pow(neck_feat_adapt - feat_t[i], 2) * entro_mask).sum() / norms_entro
            elif 'entro-back' in kd_cfg.type:
                norms_entro_back = max(1.0, (1 - entro_mask).sum() * 2)
                losskd_entro = losskd_entro + (torch.pow(neck_feat_adapt - feat_t[i], 2) *
                                               entro_mask).sum() / norms_entro
                losskd_entro_back = losskd_entro_back + (torch.pow(neck_feat_adapt - feat_t[i], 2) *
                                                         (1 - entro_mask)).sum() / norms_entro_back
            else:
                losskd_entro = losskd_entro + (torch.pow(neck_feat_adapt - feat_t[i], 2) *
                                               entro_mask).sum() / norms_entro

            if 'entro_beta' in kd_cfg.type:
                loss_entro_beta += EntroBetaLoss(feat_t[i], neck_feat_adapt, entro_mask, norms_entro)
        loss_entro_beta = loss_entro_beta / len(feat_s)
        losskd_entro = losskd_entro / len(feat_s)
        losskd_entro = losskd_entro * kd_cfg.hint_neck_w
        if 'decay' in kd_cfg.type:
            losskd_entro *= kd_decay
        if kd_warm.get('hint', False):
            losskd_entro *= 0.
        losskd_entro_back = losskd_entro_back / len(feat_s)
        losskd_entro_back = losskd_entro_back * kd_cfg.hint_neck_w
        if 'decay' in kd_cfg.type:
            losskd_entro_back *= kd_decay
        if kd_warm.get('hint', False):
            losskd_entro_back *= 0.

        return losskd_entro, losskd_entro_back, loss_entro_beta