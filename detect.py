import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse
from dataset.process import Process
from model.model import CondLane
from pathlib import Path
from tqdm import tqdm

from runner import load_network

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

class Detect(object):
    def __init__(self):
        self.processes = Process(train=False)
        self.net = CondLane(batch_size=1).to("cuda")
        self.net.eval()
        load_network(self.net, "best.pth",
                finetune_from=None, verbose=True)

    def preprocess(self, img_path):
        ori_img = cv2.imread(img_path)
        img = ori_img[0:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'img_path':img_path, 'ori_img':ori_img})
        return data
    
    def to_cuda(self, data):
        if isinstance(data, torch.Tensor):
            return data.cuda()

        if isinstance(data, dict):
            return {key: self.to_cuda(value) for key, value in data.items()}

        if isinstance(data, list):
            return [self.to_cuda(item) for item in data]
        return data
    
    def inference(self, data):
        with torch.no_grad():
            data = self.to_cuda(data)
            data = self.net(data)
            # import pdb;pdb.set_trace()
            data = self.net.get_lanes(data)
        return data

    def show(self, data):
        out_file = "vis" 
        if out_file:
            out_file = osp.join(out_file, osp.basename(data['img_path']))
        lanes = [lane.to_array() for lane in data['lanes']]
        imshow_lanes(data['ori_img'], lanes, show=True, out_file=out_file)

    def run(self, data):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]
        if True:
            self.show(data)
        return data

def get_img_paths(path):
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return paths 

def process(args):
    detect = Detect()
    paths = get_img_paths(args.img)
    for p in tqdm(paths):
        # import pdb; pdb.set_trace()
        detect.run(p)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default="./images", help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true', 
            help='Whether to show the image')
    parser.add_argument('--savedir', default="./vis", type=str, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='resa_r34_culane.pth', help='The path of model')
    args = parser.parse_args()
    process(args)
