'''
将HRSC2016中(x, y, w, h, theta)的格式转换为(x1, y1, x2, y2, x3, y3, x4, y4)的格式
'''
import os
import xml.etree.ElementTree as ET
import math
from lxml.etree import Element, SubElement, tostring
import pprint
from xml.dom.minidom import parseString
import glob

origin_ann_dir = r'/home/qinran_2020/data/HRSC2016/FullDataSet/Annotations'
new_ann_dir = r'/home/qinran_2020/data/HRSC2016/FullDataSet/Annotations_dota'
pi = 3.1415926

# 解析文件名出来

xml_Lists = glob.glob(origin_ann_dir + '/*.xml')
len(xml_Lists)

xml_basenames = []  # e.g. 100.jpg
for item in xml_Lists:
    xml_basenames.append(os.path.basename(item))

xml_names = []  # e.g. 100
for item in xml_basenames:
    temp1, temp2 = os.path.splitext(item)
    xml_names.append(temp1)

for it in xml_names:
    tree = ET.parse(os.path.join(origin_ann_dir, str(it) + '.xml'))
    root = tree.getroot()

    # HRSC_Objects=root.findall('HRSC_Objects')
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'train_images'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = str(it) + '.bmp'  # str(1) + '.jpg'
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = root.find('Img_SizeWidth').text
    node_height = SubElement(node_size, 'height')
    node_height.text = root.find('Img_SizeHeight').text
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = root.find('Img_SizeDepth').text

    dota_label = os.path.join(new_ann_dir, it + '.txt')
    with open(dota_label, 'w') as f_out:
        f_out.write("imagesource\n")
        f_out.write("gsd\n")
        for index, Object in enumerate(root.findall('./HRSC_Objects/HRSC_Object')):
            mbox_cx = float(Object.find('mbox_cx').text)
            mbox_cy = float(Object.find('mbox_cy').text)
            mbox_w = float(Object.find('mbox_w').text)
            mbox_h = float(Object.find('mbox_h').text)
            mbox_ang = float(Object.find('mbox_ang').text)
            difficult = int(Object.find('difficult').text)
            if difficult != 0:
                print(difficult)

            # 计算舰首 与舰尾点坐标

            bow_x = mbox_cx + mbox_w / 2 * math.cos(mbox_ang)
            bow_y = mbox_cy + mbox_w / 2 * math.sin(mbox_ang)

            tail_x = mbox_cx - mbox_w / 2 * math.cos(mbox_ang)
            tail_y = mbox_cy - mbox_w / 2 * math.sin(mbox_ang)


            bowA_x = round(bow_x + mbox_h / 2 * math.sin(mbox_ang))
            bowA_y = round(bow_y - mbox_h / 2 * math.cos(mbox_ang))

            bowB_x = round(bow_x - mbox_h / 2 * math.sin(mbox_ang))
            bowB_y = round(bow_y + mbox_h / 2 * math.cos(mbox_ang))

            tailA_x = round(tail_x + mbox_h / 2 * math.sin(mbox_ang))
            tailA_y = round(tail_y - mbox_h / 2 * math.cos(mbox_ang))

            tailB_x = round(tail_x - mbox_h / 2 * math.sin(mbox_ang))
            tailB_y = round(tail_y + mbox_h / 2 * math.cos(mbox_ang))

            outline = str(bowA_x) + ' ' + str(bowA_y) + ' ' + str(bowB_x) + ' ' +\
                      str(bowB_y) + ' ' + str(tailB_x) + ' ' + str(tailB_y) + ' ' +\
                      str(tailA_x) + ' ' + str(tailA_y) + ' ' + 'ship' + ' ' + str(difficult)

            if index == (len(root.findall('./HRSC_Objects/HRSC_Object')) - 1):
                f_out.write(outline)
            else:
                f_out.write(outline + '\n')
