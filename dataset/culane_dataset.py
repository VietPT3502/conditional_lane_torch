import sys
sys.path.append("dataset")
import os
import os.path as osp
import random
import numpy as np
from torch.utils.data import Dataset
import torch
from culane_metric import eval_predictions
import cv2
from tqdm import tqdm
import logging
from process import Process
from functools import partial

from mmcv.parallel import DataContainer as DC
from mmcv.parallel import collate
LIST_FILE = {
    'train': 'list/train_gt.txt',
    'val': 'list/test.txt',
    'test': 'list/test.txt',
} 
ORI_IMAGE_H = 590   
ORI_IMAGE_W = 1640
CUT_HEIGHT = 0
SAMPLE_Y = range(590, 270, -8)
def imshow_lanes(img, lanes, show=False, out_file=None):
    for lane in lanes:
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 4, (255, 0, 0), 2)

    if show:
        cv2.imshow('view', img)
        cv2.waitKey(0)

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)


class CULane(Dataset):
    def __init__(self, data_root, split, work_dir, processes):
        super(CULane, self).__init__()
        self.data_root = data_root 
        self.list_path = osp.join(data_root, LIST_FILE[split])
        self.load_annotations()
        self.processes = processes
        self.cut_height = CUT_HEIGHT
        self.training = 'train' in split 
        self.work_dir = work_dir
        self.ori_img_h = ORI_IMAGE_H
        self.ori_img_w = ORI_IMAGE_W
        self.sample_y = SAMPLE_Y
    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        if not osp.isfile(data_info['img_path']):
            raise FileNotFoundError('cannot find file: {}'.format(data_info['img_path']))

        img = cv2.imread(data_info['img_path'])

        img = img[self.cut_height:, :, :]
        sample = data_info.copy()
        sample.update({'img': img})

        if self.training:
            label = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[self.cut_height:, :]
            sample.update({'mask': label})

        sample = self.processes(sample)
        meta = {'full_img_path': data_info['img_path'],
                'img_name': data_info['img_name']}
        meta = DC(meta, cpu_only=True)
        sample.update({'meta': meta})

        return sample 
    
    def view(self, predictions, img_metas):
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        for lanes, img_meta in zip(predictions, img_metas):
            img_name = img_meta['img_name']
            img = cv2.imread(osp.join(self.data_root, img_name))
            out_file = osp.join(self.work_dir, 'visualization',
                                img_name.replace('/', '_'))
            lanes = [lane.to_array(self.cfg) for lane in lanes]
            imshow_lanes(img, lanes, out_file=out_file)

    def load_annotations(self):
        print('Loading CULane annotations...')
        self.data_infos = []
        with open(self.list_path) as list_file:
            for line in list_file:
                infos = self.load_annotation(line.split())
                self.data_infos.append(infos)

    def load_annotation(self, line):
        infos = {}
        img_line = line[0]
        img_line = img_line[1 if img_line[0] == '/' else 0::]
        img_path = os.path.join(self.data_root, img_line)
        infos['img_name'] = img_line 
        infos['img_path'] = img_path
        if len(line) > 1:
            mask_line = line[1]
            mask_line = mask_line[1 if mask_line[0] == '/' else 0::]
            mask_path = os.path.join(self.data_root, mask_line)
            infos['mask_path'] = mask_path

        if len(line) > 2:
            exist_list = [int(l) for l in line[2:]]
            infos['lane_exist'] = np.array(exist_list)

        anno_path = img_path[:-3] + 'lines.txt'  # remove sufix jpg and add lines.txt
        with open(anno_path, 'r') as anno_file:
            data = [list(map(float, line.split())) for line in anno_file.readlines()]
        lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2) if lane[i] >= 0 and lane[i + 1] >= 0]
                 for lane in data]
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [lane for lane in lanes if len(lane) > 3]  # remove lanes with less than 2 points

        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y
        infos['lanes'] = lanes

        return infos

    def get_prediction_string(self, pred):
        ys = np.array(list(self.sample_y))[::-1] / self.ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join(['{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def evaluate(self, predictions, output_basedir):
        print('Generating prediction output...')
        for idx, pred in enumerate(tqdm(predictions)):
            output_dir = os.path.join(output_basedir, os.path.dirname(self.data_infos[idx]['img_name']))
            output_filename = os.path.basename(self.data_infos[idx]['img_name'])[:-3] + 'lines.txt'
            os.makedirs(output_dir, exist_ok=True)
            output = self.get_prediction_string(pred)
            with open(os.path.join(output_dir, output_filename), 'w') as out_file:
                out_file.write(output)
        result = eval_predictions(output_basedir, self.data_root, self.list_path, official=True)

        return result['F1']
def worker_init_fn(worker_id, seed):
    worker_seed = worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloader(data_root, split, work_dir, processes, batch_size, num_workers, seed):
    if 'train' in split:
        shuffle = True
    else:
        shuffle = False

    dataset = CULane(data_root=data_root, split=split, work_dir=work_dir, processes=processes)


    init_fn = partial(
            worker_init_fn, seed=seed)
    samples_per_gpu = batch_size // 1
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size = batch_size, shuffle = shuffle,
        num_workers = num_workers, pin_memory = False, drop_last = False,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        worker_init_fn=init_fn)

    return data_loader
if __name__ == "__main__":
    # Create an instance of the CULane dataset for visualization
    train_process = Process(train=True)
    dataloader = build_dataloader(data_root="/home/vietpt/vietpt/code/conditional_lane_torch/culane_data", 
                               split="train", work_dir="work_dir", processes=train_process,
                               batch_size=1, num_workers=12, seed=42)


    for sample in dataloader:
        # Get the image and lanes from the sample
        image = sample['img'].squeeze(0).detach().cpu().numpy().copy()
        gt_hm = sample['gt_hm'].squeeze(0).squeeze(0).detach().cpu().numpy().copy()
        print(gt_hm.shape)
        print(image.shape)
        print(sample.keys())
        print(len(sample["img_metas"]["gt_masks"]))
        print(sample["img_metas"]["gt_masks"][0].keys())
        cv2.namedWindow("hm", cv2.WINDOW_NORMAL)
        cv2.imshow("hm", gt_hm)
        cv2.waitKey(0)
        # Visualize the lanes on the image
        # imshow_lanes(image, lanes, show=True)