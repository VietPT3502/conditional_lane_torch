import sys
sys.path.append("model")
from lane import Lane
from resnet import *
import torch
from transformer import TransConvEncoderModule
from fpn import FPN
from condlane_head import CondLaneHead
from post_processor import CondLanePostProcessor
import numpy as np
class CondLane(nn.Module):
    def __init__(self, batch_size):
        super(CondLane, self).__init__()
        num_lane_classes = 1
        self.backbone = resnet101(pretrained=True, progress=True, in_channels=[64, 128, 256, 512])
        self.aggregator = TransConvEncoderModule(in_dim=2048, attn_in_dims=[2048, 256], attn_out_dims=[256, 256], strides=[1, 1], ratios=[4, 4], pos_shape=(1, 10, 25))
        self.neck = FPN(in_channels=[256, 512, 1024, 256],out_channels=64,num_outs=4)
        self.head = CondLaneHead(heads=dict(hm=num_lane_classes), in_channels=(64, ), mask_size = (1, 80, 200),
                                 num_classes=num_lane_classes, head_channels=64,
                                 head_layers=1, disable_coords=False,
                                 branch_in_channels=64, branch_channels=64,
                                 branch_out_channels=64, reg_branch_channels=64,
                                 branch_num_conv=1, hm_idx=2, mask_idx=0,
                                 compute_locations_pre=True, location_configs=dict(size=(batch_size, 1, 80, 200), device='cuda:0'))
        self.post_process = CondLanePostProcessor(mask_size=(1, 80, 200), hm_thr=0.5, use_offset=True, nms_thr=4)
    def forward(self, batch):
        output = {}
        fea = self.backbone(batch['img'])
        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])
        if self.neck:
            fea = self.neck(fea)
        if self.training:
            out = self.head(fea, batch=batch)
            output.update(self.head.loss(out, batch))
        else:
            output = self.head(fea)

        return output


    def get_lanes(self, output):
        out_seeds, out_hm = output['seeds'], output['hm']
        ret = []
        for seeds, hm in zip(out_seeds, out_hm):
            lanes, seed = self.post_process(seeds, 4)
            result = adjust_result(
                    lanes=lanes,
                    crop_bbox=[0, 270, 1640, 590],
                    img_shape=(320, 800),
                    tgt_shape=(590, 1640),
                    )
            lanes = []
            for lane in result:
                coord = []
                for x, y in lane:
                    coord.append([x, y])
                coord = np.array(coord)
                coord[:, 0] /= 1640
                coord[:, 1] /= 590
                lanes.append(Lane(coord))
            ret.append(lanes)

        return ret
    
def adjust_result(lanes, crop_bbox, img_shape, tgt_shape=(590, 1640)):

    def in_range(pt, img_shape):
        if pt[0] >= 0 and pt[0] < img_shape[1] and pt[1] >= 0 and pt[
                1] <= img_shape[0]:
            return True
        else:
            return False

    left, top, right, bot = crop_bbox
    h_img, w_img = img_shape[:2]
    crop_width = right - left
    crop_height = bot - top
    ratio_x = crop_width / w_img
    ratio_y = crop_height / h_img
    offset_x = (tgt_shape[1] - crop_width) / 2
    offset_y = top

    results = []
    if lanes is not None:
        for key in range(len(lanes)):
            pts = []
            for pt in lanes[key]['points']:
                pt[0] = float(pt[0] * ratio_x + offset_x)
                pt[1] = float(pt[1] * ratio_y + offset_y)
                pts.append(pt)
            if len(pts) > 1:
                results.append(pts)
    return results
