import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import random

from runner import Runner

def main():
    args = parse_args()

    work_dirs = args.work_dirs
    load_from = args.load_from
    finetune_from = args.finetune_from
    view = args.view
    seed = args.seed
    batch_size = args.batch_size
    log_interval = args.log_interval
    data_root = args.data_root
    epochs = args.epochs
    save_ep = args.save_ep
    eval_ep = args.eval_ep
    cudnn.benchmark = True
    # cudnn.fastest = True

    runner = Runner(seed=seed, batch_size=batch_size, log_interval=log_interval, data_root=data_root,
                    work_dir=work_dirs, epochs=epochs, save_ep=save_ep, eval_ep=eval_ep, view=view, load_from=load_from, finetune_from=finetune_from)

    if args.validate:
        runner.validate(epoch=epochs)
    else:
        runner.train()

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument(
        '--data_root', type=str, default='culane_data',
        help='data root')
    parser.add_argument(
        '--work_dirs', type=str, default='work_dir',
        help='work dirs')
    parser.add_argument(
        '--load_from', default=None,
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--finetune_from', default=None,
        help='whether to finetune from the checkpoint')
    parser.add_argument(
        '--view', action='store_true', 
        help='whether to view')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument('--seed', type=int,
                        default=0, help='random seed')

    parser.add_argument('--save_ep', type=int,
                        default=1, help=' save_ep')    
    parser.add_argument('--batch_size', type=int,
                        default=2, help='batch_size')
    parser.add_argument('--epochs', type=int,
                        default=15, help='epochs')
    parser.add_argument('--log_interval', type=int,
                        default=2000, help='log_interval')
    parser.add_argument('--eval_ep', type=int,
                        default=1, help='log_interval')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
