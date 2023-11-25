import time
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import cv2
from model.model import CondLane
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from dataset.culane_dataset import build_dataloader
from dataset.process import Process
import os
def load_network_specified(net, model_dir, verbose=False):
    pretrained_net = torch.load(model_dir)['net']
    net_state = net.state_dict()
    state = {}
    for k, v in pretrained_net.items():
        if k not in net_state.keys() or v.size() != net_state[k].size():
            if verbose:
                print('skip weights: ' + k)
        state[k] = v
    net.load_state_dict(state, strict=False)
def load_network(net, model_dir, finetune_from=None, verbose=True):
    if finetune_from:
        if verbose:
            print('Finetune model from: ' + finetune_from)
        load_network_specified(net, finetune_from, False)
        return
    pretrained_model = torch.load(model_dir)
    net.load_state_dict(pretrained_model['net'], strict=True)

class Runner(object):
    def __init__(self, seed, batch_size, log_interval=1000, data_root="culane_data", work_dir="work_dir",
                 epochs=16, save_ep=1, eval_ep=1, view=True, load_from=None, finetune_from=None, device="cuda"):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device = device
        self.load_from = load_from
        self.finetune_from = finetune_from
        self.seed = seed
        self.view = view
        self.save_ep=save_ep
        self.eval_ep = eval_ep
        self.epochs = epochs
        self.batch_size = batch_size
        self.net = CondLane(batch_size).to(device)
        self.resume()
        self.log_interval = log_interval
        self.data_root = data_root
        self.work_dir = work_dir
        self.train_process = Process(train=True)
        self.val_process = Process(train=False)
        # self.net.to(torch.device('cuda'))
        # self.net = torch.nn.parallel.DataParallel(
        #         self.net, device_ids = range(self.cfg.gpus)).cuda()

        self.optimizer = torch.optim.AdamW([dict(params=self.net.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)])
        self.scheduler = MultiStepLR(self.optimizer, milestones=[8, 14], gamma=0.1)
        self.lr_update_by_epoch = True
        self.metric = 0.
        self.val_loader = None

    def resume(self):
        if not self.load_from and not self.finetune_from:
            print("dont resume")
            return
        print("resume check point")
        load_network(self.net, self.load_from,
                finetune_from=self.finetune_from, verbose=True)

    def to_cuda(self, data):
        if isinstance(data, torch.Tensor):
            return data.cuda()

        if isinstance(data, dict):
            return {key: self.to_cuda(value) for key, value in data.items()}

        if isinstance(data, list):
            return [self.to_cuda(item) for item in data]

        return data
    
    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        with tqdm(total=len(train_loader)) as pbar:
            for i, data in enumerate(train_loader):
                date_time = time.time() - end
                data = self.to_cuda(data)
                output = self.net(data)
                self.optimizer.zero_grad()
                loss = output['loss']
                loss.backward()
                self.optimizer.step()
                if not self.lr_update_by_epoch:
                    self.scheduler.step()
                batch_time = time.time() - end
                end = time.time()

                if i % self.log_interval == 0 or i == max_iter - 1:
                    lr = self.optimizer.param_groups[0]['lr']
                    print("iteration:", i)
                    print("current lr", lr)
                    hm_loss = output["loss_stats"]['hm_loss'].item()
                    kps_loss = output["loss_stats"]['kps_loss'].item()
                    row_loss = output["loss_stats"]['row_loss'].item()
                    range_loss = output["loss_stats"]['range_loss'].item()
                    total_loss = output["loss_stats"]['loss'].item()
                    print(f"hm_loss: {hm_loss}, kps_loss: {kps_loss}, row_loss: {row_loss}, range_loss: {range_loss}, total_loss: {total_loss}")

                    print("batch_time:", batch_time)
                    print("date_time:", date_time)
                pbar.update(1)

    def train(self):
        print('Build train loader...')
        train_loader = build_dataloader(data_root=self.data_root, 
                               split="train", work_dir="work_dir", processes=self.train_process,
                               batch_size=self.batch_size, num_workers=12, seed=self.seed)

        print('Start training...')
        for epoch in range(self.epochs):
            print("epoch: ", epoch)
            self.train_epoch(epoch, train_loader)
            if (epoch + 1) % self.save_ep == 0 or epoch == self.epochs - 1:
                self.save_ckpt(epoch)
            if (epoch + 1) % self.eval_ep == 0 or epoch == self.epochs - 1:
                self.validate(epoch)
            if self.lr_update_by_epoch:
                self.scheduler.step()

    def validate(self, epoch):
        if not self.val_loader:
            self.val_loader = build_dataloader(data_root=self.data_root, 
                               split="val", work_dir="work_dir", processes=self.val_process,
                               batch_size=1, num_workers=12, seed=self.seed)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.get_lanes(output)
                predictions.extend(output)
            if self.view:
                self.val_loader.dataset.view(output, data['meta'])

        out = self.val_loader.dataset.evaluate(predictions, self.work_dir)
        print(out)
        metric = out
        if metric > self.metric:
            self.metric = metric
            self.save_ckpt(epoch, is_best=True)
        print('Best metric: ' + str(self.metric))

    def save_ckpt(self, epoch, is_best=False):
        save_model(self.net, epoch, self.optimizer, self.scheduler,
                self.work_dir, is_best)


def save_model(net, epoch, optim, scheduler, work_dir, is_best=False):
    model_dir = os.path.join(work_dir, 'ckpt')
    os.system('mkdir -p {}'.format(model_dir))
    ckpt_name = 'best' if is_best else epoch
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(ckpt_name)))