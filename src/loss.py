import torch
import torch.nn as nn
import os


def calc_iou(a, b):
    GT = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    PRED = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1)
    iw = torch.min(torch.unsqueeze(
        a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(
        a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    intersection = iw * ih
    ua = PRED + GT - intersection
    ua = torch.clamp(ua, min=1e-8)
    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.2
        gamma = 1.5
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchors = anchors[0]

        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())
                continue

            classification = torch.clamp(classification, 1e-8, 1.0 - 1e-8)

            IoU = calc_iou(anchors, bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            targets = torch.ones(classification.shape).cuda() * -1
            targets[torch.lt(IoU_max, 0.4), :] = 0
            positive_indices = torch.ge(IoU_max, 0.5)
            assigned_annotations = bbox_annotation[IoU_argmax, :]
            targets[positive_indices, :] = 0
            targets[positive_indices,
                    assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape).cuda() * alpha

            alpha_factor = torch.where(
                torch.eq(
                    targets,
                    1.),
                alpha_factor,
                1. - alpha_factor)
            focal_weight = torch.where(
                torch.eq(
                    targets,
                    1.),
                1. - classification,
                classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) +
                    (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce
            zeros = torch.zeros(cls_loss.shape).cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            num_positive_anchors = positive_indices.sum()
            classification_losses.append(
                cls_loss.sum() /
                torch.clamp(
                    num_positive_anchors.float(),
                    min=1.0))

            if num_positive_anchors > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:,
                                                 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:,
                                                  3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack(
                    (targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                norm = torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

                targets = targets / norm

                regression_diff = torch.abs(
                    targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean().cuda())
            else:
                regression_losses.append(torch.tensor(0).float().cuda())

        return (torch.stack(classification_losses).mean(dim=0, keepdim=True),
                torch.stack(regression_losses).mean(dim=0, keepdim=True))
