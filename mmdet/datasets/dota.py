from .registry import DATASETS
from .xml_style import XMLDataset
from .coco import CocoDataset
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from .custom import CustomDataset


@DATASETS.register_module
class DOTADatasetCoco(CocoDataset):
    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship',
               'tennis-court', 'basketball-court',
               'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor',
               'swimming-pool', 'helicopter')

    def __init__(self, **kwargs):
        super(DOTADatasetCoco, self).__init__(**kwargs)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            # bbox = [x1, y1 , x1 + w - 1, y1 + h -1]
            bbox = [x1 - 1, y1 - 1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                # gt_masks_ann.append(ann['segmentation'])
                x1, y1, x2, y2, x3, y3, x4, y4 = ann['segmentation'][0]
                gt_masks_ann.append([[x1 - 1, y1 - 1,
                                     x2 - 1, y2 - 1,
                                     x3 - 1, y3 - 1,
                                     x4 - 1, y4 - 1]])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

