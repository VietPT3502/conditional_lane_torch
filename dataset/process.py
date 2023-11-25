import sys
sys.path.append("dataset")

from alaug import Alaug
from collect_lane import CollectLane
from transforms import *
img_norm = dict(
    mean=[75.3, 76.6, 77.6],
    std=[50.5, 53.8, 54.3]
)
crop_bbox = [0, 270, 1640, 590]
img_scale = (800, 320)
mask_down_scale = 4
hm_down_scale = 16
line_width = 3
radius = 6
train_processes = [Alaug(transforms=[
                    dict(type='Compose', params=dict(bboxes=False, keypoints=True, masks=False)),
                    dict(type='Crop', x_min=crop_bbox[0], x_max=crop_bbox[2], y_min=crop_bbox[1], y_max=crop_bbox[3], p=1),
                    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
                    dict(type='OneOf',
                        transforms=[
                            dict(type='RGBShift', r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
                            dict(type='HueSaturationValue', hue_shift_limit=(-10, 10), sat_shift_limit=(-15, 15), val_shift_limit=(-10, 10), p=1.0),
                        ],
                        p=0.7),
                    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
                    dict(type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(type='MedianBlur', blur_limit=3, p=1.0)
                        ],
                        p=0.2),
                    dict(type='RandomBrightness', limit=0.2, p=0.6),
                    dict(type='ShiftScaleRotate', shift_limit=0.1, scale_limit=(-0.2, 0.2), rotate_limit=10, border_mode=0, p=0.6),
                    dict(type='RandomResizedCrop', height=img_scale[1], width=img_scale[0], scale=(0.8, 1.2), ratio=(1.7, 2.7), p=0.6),
                    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
                ]),
                CollectLane(down_scale=mask_down_scale, hm_down_scale=hm_down_scale, max_mask_sample=10,
                                        line_width=line_width, radius= radius, keys= ["img", "gt_hm"],
                                        meta_keys= ["gt_masks", "mask_shape", "hm_shape", "down_scale", "hm_down_scale", "gt_points"], 
                                        img_height= 320, img_width= 800),

                Normalize(img_norm=img_norm),
                ToTensor(keys=["img", "gt_hm"], collect_keys=["img_metas"])]

val_processes = [Alaug(transforms=[
                    dict(type='Compose', params=dict(bboxes=False, keypoints=True, masks=False)),
                    dict(type='Crop', x_min=crop_bbox[0], x_max=crop_bbox[2], y_min=crop_bbox[1], y_max=crop_bbox[3], p=1),
                    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
                ]),
                Normalize(img_norm=img_norm),
                ToTensor(keys=["img", "gt_hm"], collect_keys=["img_metas"])]


class Process(object):
    def __init__(self, train=True):
        self.processes = []
        if train:
            for process in train_processes:
                self.processes.append(process)
        else:
            for process in val_processes:
                self.processes.append(process)
    def __call__(self, data):

        for t in self.processes:
            data = t(data)
            if data is None:
                return None
        return data
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.processes:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
