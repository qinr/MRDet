from .registry import DATASETS
from .xml_style import XMLDataset
from .coco import CocoDataset
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np


@DATASETS.register_module
class HRSC2016DatasetCoco(CocoDataset):
    CLASSES = ('ship',)

    def __init__(self, **kwargs):
        super(HRSC2016DatasetCoco, self).__init__(**kwargs)

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

        seg_map = img_info['filename'].replace('jpg', 'bmp')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

@DATASETS.register_module
class HRSC2016DatasetVOCH(XMLDataset):
    CLASSES = ('ship',)

    def __init__(self, **kwargs):
        super(HRSC2016DatasetVOCH, self).__init__(**kwargs)
        self.year = 2007 # HRSC2016采用2007的mAP计算方法

    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = 'AllImages/{}.bmp'.format(img_id)
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            # size = root.find('size')
            width = int(root.find('Img_SizeWidth').text)
            height = int(root.find('Img_SizeHeight').text)
            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations',
                            '{}.xml'.format(img_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        hrsc_objects = root.find('HRSC_Objects')
        for obj in hrsc_objects.findall('HRSC_Object'):
            object_id = obj.find('Object_ID').text
            name = 'ship'
            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)
            bbox = [
                int(obj.find('box_xmin').text),
                int(obj.find('box_ymin').text),
                int(obj.find('box_xmax').text),
                int(obj.find('box_ymax').text)
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

