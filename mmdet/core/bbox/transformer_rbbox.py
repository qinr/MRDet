import mmcv
import numpy as np
import math
import math
import torch
import copy
import cv2

def ndarray2tensor(arrays, device_id):
    '''
    :param arrays:list(ndarray) 
    :return: 
    '''
    tensors = []
    for i in range(len(arrays)):
        tensor = torch.from_numpy(arrays[i]).to(device_id)
        tensors.append(tensor)
    return tensors

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_list(rbboxes_list):
    rbboxes_new = map(get_best_begin_point, rbboxes_list)
    return list(rbboxes_new)


def get_best_begin_point(rbboxes):
    '''
    :param rbboxes: (x1, y1, x2, y2, x3, y3, x4, y4)
    :return: 调整顺序成(x1', y1', x2', y2', x3', y3', x4', y4')
            使得与(xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax）的空间位置一致
   '''
    rbboxes_best = rbboxes.new_zeros(rbboxes.size())
    for i in range(len(rbboxes)):
        rbbox = rbboxes[i]
        x1 = rbbox[0]
        y1 = rbbox[1]
        x2 = rbbox[2]
        y2 = rbbox[3]
        x3 = rbbox[4]
        y3 = rbbox[5]
        x4 = rbbox[6]
        y4 = rbbox[7]

        xmin = min(x1, x2, x3, x4)
        ymin = min(y1, y2, y3, y4)
        xmax = max(x1, x2, x3, x4)
        ymax = max(y1, y2, y3, y4)
        combinate = rbbox.new_tensor([[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                     [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]])
        dst_coordinate = torch.tensor([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        force = 100000000.0
        force_flag = 0
        for j in range(4):
            temp_force = cal_line_length(combinate[j][0], dst_coordinate[0]) + \
                         cal_line_length(combinate[j][1], dst_coordinate[1]) + \
                         cal_line_length(combinate[j][2], dst_coordinate[2]) + \
                         cal_line_length(combinate[j][3], dst_coordinate[3])
            if temp_force < force:
                force = temp_force
                force_flag = j
        temp = combinate[force_flag].view(1, 8)
        rbboxes_best[i] = temp

    return rbboxes_best

def mask2poly_single(binary_mask):
    """

    :param binary_mask:
    :return:
    """
    try:
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contour_lens = np.array(list(map(len, contours)))
        # max_id = contour_lens.argmax()
        # max_contour = contours[max_id]
        max_contour = max(contours, key=len)
        rect = cv2.minAreaRect(max_contour)
        poly = cv2.boxPoints(rect)
        # 点排列为顺时针顺序，第一个点即为矩形旋转angle的基准点；
        # angle为顺时针旋转角度，即第一点和第四点连成的线段与x轴正方向之间的夹角的相反数。
        # poly = TuplePoly2Poly(poly)
    except:
        import pdb
        pdb.set_trace()
    return poly

def mask2poly(masks):
    '''
    :param masks: list(ndarray): 一张图的mask
    :return: polys: list(ndarray)
    '''
    polys = map(mask2poly_single, masks)
    return list(polys)

def mask_2_rbbox(masks):
    '''
    :param masks: ndarray: 一张图的mask
    :return: ndarray
    '''
    polys = mask2poly(masks)
    polys_new = np.stack(polys).reshape(-1, 8)
    return polys_new

def mask_2_rbbox_list(mask_list):
    '''
    :param mask_list: list(ndarray),第一层list:图片数目，第二层：图片中bbox数目 
    :return: rbbox_list: list(ndarray),第一层list:图片数目，第二层:图片中bbox数目
    '''
    rbbox_list = map(mask_2_rbbox, mask_list)
    return list(rbbox_list)


def choose_best_match(rrois, gt_rrois):
    """
    choose best match representation of gt_rrois for a rrois
    将gt_rrois的角度调整到最好回归的角度，即差距最小的角度
    :param rrois: (x_ctr, y_ctr, w, h, angle)
            shape: (n, 5)
    :param gt_rrois: (x_ctr, y_ctr, w, h, angle)
            shape: (n, 5)
    :return: gt_rroi_news: gt_roi with new representation
            shape: (n, 5)
    """
    # TODO: check the dimensions
    rroi_angles = rrois[:, 4].unsqueeze(1)

    gt_xs, gt_ys, gt_ws, gt_hs, gt_angles = copy.deepcopy(gt_rrois[:, 0]), copy.deepcopy(gt_rrois[:, 1]), \
                                            copy.deepcopy(gt_rrois[:, 2]), copy.deepcopy(gt_rrois[:, 3]), \
                                            copy.deepcopy(gt_rrois[:, 4])

    gt_angle_extent = torch.cat(((gt_angles - math.pi/2.)[:, np.newaxis], (gt_angles)[:, np.newaxis],
                                 (gt_angles + math.pi/2.)[:, np.newaxis], (gt_angles + math.pi)[:, np.newaxis]), 1)
    dist = abs(rroi_angles - gt_angle_extent)
    min_index = torch.argmin(dist, 1)

    gt_rrois_extent0 = torch.cat((gt_xs.unsqueeze(1), gt_ys.unsqueeze(1), \
                                 gt_hs.unsqueeze(1), gt_ws.unsqueeze(1), gt_angles.unsqueeze(1) - np.pi/2.), 1)
    gt_rrois_extent1 = copy.deepcopy(gt_rrois)
    gt_rrois_extent2 = torch.cat((gt_xs.unsqueeze(1), gt_ys.unsqueeze(1), \
                                 gt_hs.unsqueeze(1), gt_ws.unsqueeze(1), gt_angles.unsqueeze(1) + np.pi/2.), 1)
    gt_rrois_extent3 = torch.cat((gt_xs.unsqueeze(1), gt_ys.unsqueeze(1), \
                                 gt_ws.unsqueeze(1), gt_hs.unsqueeze(1), gt_angles.unsqueeze(1) + np.pi), 1)
    gt_rrois_extent = torch.cat((gt_rrois_extent0.unsqueeze(1),
                                     gt_rrois_extent1.unsqueeze(1),
                                     gt_rrois_extent2.unsqueeze(1),
                                     gt_rrois_extent3.unsqueeze(1)), 1)

    gt_rrois_new = torch.zeros_like(gt_rrois)
    # TODO: add pool.map here
    for curiter, index in enumerate(min_index):
        gt_rrois_new[curiter, :] = gt_rrois_extent[curiter, index, :]


    return gt_rrois_new

def rbbox2hbbox(rbboxes):
    '''
    将rbbox(x1,y1,x2,y2,x3,y3,x4,y4)转换成hbbox(xmin,ymin,xmax,ymax)
    :param：
        rbbox(list[Tensor]): Tensor(n,8),每一项为(x1,y1,x2,y2,x3,y3,x4,y4)
    :return: 
        hbbox(list[Tensor])：Tensor(n,4),每一项为(xmin,ymin,xmax,ymax)

    '''
    if rbboxes is None:
        return None
    hbboxes = []
    for i in range(len(rbboxes)):
        rbbox = rbboxes[i]
        xs = rbbox.view(-1, 4, 2)[:, :, 0]
        ys = rbbox.view(-1, 4, 2)[:, :, 1]

        xmin = torch.min(xs, dim=1)[0]
        ymin = torch.min(ys, dim=1)[0]
        xmax = torch.max(xs, dim=1)[0]
        ymax = torch.max(ys, dim=1)[0]

        hbboxes.append(torch.stack([xmin, ymin, xmax, ymax], dim=1))

    return hbboxes


def hbbox2rbbox(hbboxes):
    '''
    :param hbboxes(list[Tensor]):(xmin,ymin,xmax,ymax,p) 
    :return: rbboxes(list[Tensor]):(x1,y1,x2,y2,x3,y3,x4,y4) 
    '''

    rbboxes = []
    for i in range(len(hbboxes)):
        hbbox = hbboxes[i]
        xmin = hbbox[:, 0]
        ymin = hbbox[:, 1]
        xmax = hbbox[:, 2]
        ymax = hbbox[:, 3]

        hbbox = torch.stack([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax], dim=-1)
        rbboxes.append(hbbox)

    return rbboxes

def hbbox2rbboxRec(hbboxes):
    '''
    :param hbboxes: Tensor, (xmin, ymin, xmax, ymax)
    :return: rbboxes:Tensor, (x, y, w, h, theta)
    '''
    if hbboxes is None:
        return None
    num_hbboxes = hbboxes.size(0)
    h = hbboxes[..., 2] - hbboxes[..., 0] + 1.0
    w = hbboxes[..., 3] - hbboxes[..., 1] + 1.0
    x_center = hbboxes[..., 0] + 0.5 * (h - 1.0)
    y_center = hbboxes[..., 1] + 0.5 * (w - 1.0)
    hbboxes_rec = torch.cat(
        (x_center.unsqueeze(1), y_center.unsqueeze(1), w.unsqueeze(1), h.unsqueeze(1)), 1)
    initial_angles = hbboxes_rec.new_ones((num_hbboxes, 1)) * math.pi / 2
    # initial_angles = -torch.ones((num_boxes, 1)) * np.pi/2
    hbboxes_rec = torch.cat((hbboxes_rec, initial_angles), 1)
    return hbboxes_rec


def hbboxList2rbboxRec(hbboxes):
    '''
    :param hbboxes: list(Tensor), (xmin, ymin, xmax, ymax)
    :return: rbboxes: list(Tensor), (x, y, w, h, theta)
    '''
    recs = []
    for i in range(len(hbboxes)):
        rec = hbbox2rbboxRec(hbboxes[i])
        recs.append(rec)
    return recs



def rec2delta(roi_recs, gt_recs, means=[0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
    '''
    :param roi_recs(list): (xr,yr,wr,hr,thetar)
    :param gt_recs(list): (x*,y*,w*,h*,theta*)
    :return: 
        deltas(list):(tx,ty,tw,th,ttheta)
    '''

    x = roi_recs[:, 0]
    y = roi_recs[:, 1]
    w = roi_recs[:, 2]
    h = roi_recs[:, 3]
    theta = roi_recs[:, 4]

    x_gt = gt_recs[:, 0]
    y_gt = gt_recs[:, 1]
    w_gt = gt_recs[:, 2]
    h_gt = gt_recs[:, 3]
    theta_gt = gt_recs[:, 4]

    delta_x = (torch.cos(theta) * (x_gt - x) + torch.sin(theta) * (y_gt - y)) / w
    delta_y = (torch.cos(theta) * (y_gt - y) - torch.sin(theta) * (x_gt - x)) / h
    delta_w = torch.log(w_gt / w)
    delta_h = torch.log(h_gt / h)
    delta_theta = (theta_gt - theta)

    deltas = torch.stack([delta_x, delta_y, delta_w, delta_h, delta_theta], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas



def delta2rec(deltas,
              roi_recs,
              means=[0, 0, 0, 0, 0],
              stds=[1, 1, 1, 1, 1],
              max_shape=None,
              wh_ratio_clip=16 / 1000):
    '''
    :param deltas: (tx,ty,tw,th,ttheta)
    :param roi_recs: (x,y,w,h,theta)
    :return: 
        roi_preds:(x_pred,y_pred,w_pred,h_pred,theta_pred)
    '''
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    deform_deltas = deltas * stds + means

    delta_x = deform_deltas[:, 0::5]
    delta_y = deform_deltas[:, 1::5]
    delta_w = deform_deltas[:, 2::5]
    delta_h = deform_deltas[:, 3::5]
    delta_theta = deform_deltas[:, 4::5]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    delta_w = delta_w.clamp(min=-max_ratio, max=max_ratio)
    delta_h = delta_h.clamp(min=-max_ratio, max=max_ratio)
    x =(roi_recs[:, 0]).unsqueeze(1).expand_as(delta_x)
    y = (roi_recs[:, 1]).unsqueeze(1).expand_as(delta_y)
    w = (roi_recs[:, 2]).unsqueeze(1).expand_as(delta_w)
    h = (roi_recs[:, 3]).unsqueeze(1).expand_as(delta_h)
    theta = (roi_recs[:, 4]).unsqueeze(1).expand_as(delta_theta)

    # x_pred = w.mul(torch.cos(theta)).mul(delta_x) - h.mul(torch.sin(theta)).mul(delta_y) + x
    # y_pred = w.mul(torch.sin(theta)).mul(delta_x) + h.mul(torch.cos(theta)).mul(delta_y) + y
    # w_pred = w.mul(torch.exp(delta_w))
    # h_pred = h.mul(torch.exp(delta_h))
    # theta_pred = (delta_theta * 2 * math.pi + theta) % (2 * math.pi)

    x_pred = delta_x * w * torch.cos(theta) - delta_y * h * torch.sin(theta) + x
    y_pred = delta_x * w * torch.sin(theta) + delta_y * h * torch.cos(theta) + y
    w_pred = w * delta_w.exp()
    h_pred = h * delta_h.exp()
    theta_pred = theta + delta_theta
    return torch.stack([x_pred, y_pred, w_pred, h_pred, theta_pred], dim=-1).view_as(deltas)




def rbbox2rroi(rbbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 6), [batch_ind, x, y, w, h, theta]
    """
    rrois_list = []
    for img_id, bboxes in enumerate(rbbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rrois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rrois = bboxes.new_zeros((0, 6))
        rrois_list.append(rrois)
    rrois = torch.cat(rrois_list, 0)
    return rrois


def rbbox2result(rbboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 9)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if rbboxes.shape[0] == 0:
        return [
            np.zeros((0, 9), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        rbboxes = rbboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [rbboxes[labels == i, :] for i in range(num_classes - 1)]


