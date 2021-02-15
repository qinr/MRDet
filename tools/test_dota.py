import argparse
import os
import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model, get_classes, tensor2imgs
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from DOTA_devkit.ResultMerge_multi_process import mergebypoly_multiprocess
import numpy as np
import os
import cv2
from DOTA_devkit.dota_utils import GetFileFromThisRootDir, custombasename

def draw_bbox(img, bboxes, labels, path, class_names):
    img_show = mmcv.image.imread(img)
    # bgr
    bbox_color = [(0, 255, 0),  # green
                  (255, 0, 0), #深蓝
                  (255, 255, 0), # 浅蓝，亮
                  (0, 0, 255), #红
                  (255, 0, 255), # purple
                  (255, 128, 0), #天蓝（比浅蓝深一点）
                  (0, 255, 255), #黄
                  (207, 203, 211), #white
                  (128, 255, 0), # 青色
                  (128, 0, 255), #玫红
                  (255, 0, 128), # 紫
                  (0, 128, 255), # 橘色
                  (0, 255, 128), #草绿
                  (0, 0, 128), #深红
                  (128, 0, 0)] #藏蓝
    text_color = (255, 0, 0)  # green

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        pts = np.array([[bbox_int[0], bbox_int[1]],
                        [bbox_int[2], bbox_int[3]],
                        [bbox_int[4], bbox_int[5]],
                        [bbox_int[6], bbox_int[7]]], dtype=np.int32)
        cv2.polylines(img_show, [pts], True, bbox_color[label], thickness=2)
        # cv2.polylines(img_show, [pts], True, text_color, thickness=2)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        font = 0.5
        # cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
        #             cv2.FONT_HERSHEY_COMPLEX, font, text_color)
        cv2.imwrite(path, img)

def draw_result(data, result, outdir, class_names, score_thr=0.001):
    bbox_result = result
    img_metas = data['img_meta'][0].data[0]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for img_meta in img_metas:
        h, w, _ = img_meta['ori_shape']
        filename = img_meta['filename']
        img = mmcv.imread(filename)
        img_show = img[:h, :w, :]

        path = os.path.basename(os.path.splitext(img_meta['filename'])[0])
        path = os.path.join(outdir, path + '.jpg')

        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        if score_thr > 0:
            assert bboxes.shape[1] == 9
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        draw_bbox(img_show, bboxes, labels, path, class_names)


def single_gpu_test(model, data_loader, outdir, show=False):
    model.eval()
    # model.eval()，让model变成测试模式，对dropout和batch normalization的操作在训练和测试的时候是不一样的
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            draw_result(data, result, osp.join(outdir, 'images'), dataset.CLASSES, score_thr=0.001)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--outdir', help='output dir')
    parser.add_argument('--out', help='output result file') # .pkl文件
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def write_dota_results(path, boxes, dataset, threshold=0.001):
    '''
    :param path: output dir path
    :param boxes: list(list(ndarray))
    :param threshold: 置信度下限，小于此置信度的bbox不输出
    :return: 
    '''
    classes = dataset.CLASSES
    img_infos = dataset.img_infos
    assert len(boxes) == len(img_infos)
    print("write no merge results\n")
    for i, img_info in enumerate(img_infos):
        # print("img {}: {}".format(i, img_info['id']))
        img_id = img_info['id']
        for j, cls in enumerate(classes):
            txt_path = osp.join(path, 'Task1_' + cls + '.txt')
            with open(txt_path, 'a') as f:
                box = boxes[i][j] # (n, 9)
                inds = box[:, 8] > threshold
                box = box[inds]
                for k in range(box.shape[0]):
                    # print(cls)
                    # print('{} {} {} {} {} {} {} {} {} {}\n'.format(
                    #     img_id, box[k, 8],
                    #     int(box[k, 0]), int(box[k, 1]),
                    #     int(box[k, 2]), int(box[k, 3]),
                    #     int(box[k, 4]), int(box[k, 5]),
                    #     int(box[k, 6]), int(box[k, 7])))
                    f.write('{} {} {} {} {} {} {} {} {} {}\n'.format(
                        img_id, box[k, 8],
                        int(box[k, 0]), int(box[k, 1]),
                        int(box[k, 2]), int(box[k, 3]),
                        int(box[k, 4]), int(box[k, 5]),
                        int(box[k, 6]), int(box[k, 7])))

def main():
    args = parse_args()

    assert args.out , \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.outdir, args.show)
        # outputs:list(list(ndarray)),外层list:图片，内层list:类别
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    # 将结果保存到.pkl文件中
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(outputs, osp.join(args.outdir, args.out))





if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()

