from __future__ import division

import argparse
import os.path as osp
import shutil
import tempfile
import os
import mmcv
from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval, \
    HBBSeg2Comp4, OBBDet2Comp4, \
    HBBOBB2Comp4, HBBDet2Comp4

import argparse
from mmdet.datasets import build_dataset, build_dataloader
from mmcv import Config
from DOTA_devkit.ResultMerge_multi_process import *
import DOTA_devkit.utils as util
# import pdb; pdb.set_trace()
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default='configs/DOTA/faster_rcnn_r101_fpn_1x_dota2_v3_RoITrans_v5.py')
    parser.add_argument('--outdir')
    parser.add_argument('--pkl_file')
    parser.add_argument('--type', default=r'OBB',
                        help='parse type of detector')
    args = parser.parse_args()

    return args

def parse_results(config_file, resultfile, dstpath, type):
    cfg = Config.fromfile(config_file)
    cfg.data.test.test_mode = True
    data_test = cfg.data.test
    dataset = build_dataset(data_test)
    outputs = mmcv.load(resultfile)
    if type == 'OBB':
        #  dota1 has tested
        obb_results_dict = OBBDet2Comp4(dataset, outputs)


    if 'obb_results_dict' in vars():
        if not os.path.exists(os.path.join(dstpath, 'L1_results')):
            os.makedirs(os.path.join(dstpath, 'L1_results'))

        for cls in obb_results_dict:
            with open(os.path.join(dstpath, 'L1_results', cls + '.txt'), 'w') as obb_f_out:
                for index, outline in enumerate(obb_results_dict[cls]):
                    if index != (len(obb_results_dict[cls]) - 1):
                        obb_f_out.write(outline + '\n')
                    else:
                        obb_f_out.write(outline)



if __name__ == '__main__':
    args = parse_args()
    config_file = args.config
    config_name = os.path.splitext(os.path.basename(config_file))[0]
    pkl_file = os.path.join(args.outdir, args.pkl_file)
    output_path = args.outdir
    type = args.type
    parse_results(config_file, pkl_file, output_path, type)

