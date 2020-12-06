from __future__ import division
import argparse
import os

import torch
from mmcv import Config

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # 参数读取：读取configs/*.py文件，并检测mmcv.config对象用于后续模型训练参数设置
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    # 在图片输入尺度固定时开启，可以加速，一般都是关的，只有在固定尺度的网络如SSD512中才开启
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        # print(cfg.dist_params)
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('MMDetection Version: {}'.format(__version__))
    logger.info('Config: {}'.format(cfg.text))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    # 构建模型：调用mmdet.models.build_detector,从config中读取必要的模型参数构建模型，包括模型参数+训练参数+测试参数
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    # 数据集读取：调用mmdet.datasets.build_dataset，根据从配置文件中读取训练测试数据集的信息，建立数据集对象
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val)) # 可以支持train+val训练
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # 开始训练
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7, 8, 9'
    main()
