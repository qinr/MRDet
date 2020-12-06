import torch
import numpy as np
from functools import partial

def hbbox2rbboxRec_v2(hbboxes):
    '''
    :param hbboxes: Tensor, (xmin, ymin, xmax, ymax)
    :return: rbboxes:Tensor, (x, y, w, h, theta)
    '''
    if hbboxes is None:
        return None
    num_hbboxes = hbboxes.size(0)
    w = hbboxes[..., 2] - hbboxes[..., 0] + 1.0
    h = hbboxes[..., 3] - hbboxes[..., 1] + 1.0
    x_center = hbboxes[..., 0] + 0.5 * (w - 1.0)
    y_center = hbboxes[..., 1] + 0.5 * (h - 1.0)
    hbboxes_rec = torch.cat(
        (x_center.unsqueeze(1), y_center.unsqueeze(1), w.unsqueeze(1), h.unsqueeze(1)), 1)
    initial_angles = hbboxes_rec.new_zeros((num_hbboxes, 1))
    # initial_angles = -torch.ones((num_boxes, 1)) * np.pi/2
    hbboxes_rec = torch.cat((hbboxes_rec, initial_angles), 1)

    # inds = w < h
    # hbboxes_rec[inds, 2] = h[inds]
    # hbboxes_rec[inds, 3] = w[inds]
    # hbboxes_rec[inds, 4] = hbboxes_rec[inds, 4] - np.pi / 2.

    return hbboxes_rec

def rbboxPoly2RectangleList_v2(rbbox_list):
    rec = map(rbboxPoly2Rectangle_v2, rbbox_list)
    return list(rec)

def rbboxPoly2Rectangle_v2(rbbox):
    '''
    :param rbboxes(Tensor):(x1,y2,x2,y2,x3,y3,x4,y4) 
    :return: recs(Tensor):(x_center, y_center, w, h, theta)
    theta范围(-pi, pi)
    '''
    if rbbox is None:
        return None
    if rbbox.size(0) == 0:
        rec = rbbox.new_zeros(0, 5)
        return rec
    rbbox = rbbox.view(-1, 4, 2)
    rbbox = rbbox.permute(0, 2, 1)
    angle = torch.atan2((rbbox[:, 1, 1] - rbbox[:, 1, 0]), (rbbox[:, 0, 1] - rbbox[:, 0, 0]))
    # arctan((y2 - y1) / (x2 - x1))
    # angle = torch.atan2((rbbox[:, 1, 3] - rbbox[:, 1, 0]), (rbbox[:, 0, 3] - rbbox[:, 0, 0]))
    center = rbbox.new_zeros((rbbox.shape[0], 2, 1))
    for i in range(4):
        center[:, 0, 0] += rbbox[:, 0, i]
        center[:, 1, 0] += rbbox[:, 1, i]
    center = center / 4.0

    R = rbbox.new_tensor([[[torch.cos(_angle), -torch.sin(_angle)],
                           [torch.sin(_angle), torch.cos(_angle)]] for _angle in angle])
    RT = R.permute(0, 2, 1)
    normalized = [torch.mm(RT[i], rbbox[i] - center[i]) for i in range(rbbox.shape[0])]  # list([Tensor])
    normalized = torch.stack(normalized, dim=0)

    xmin = torch.min(normalized[:, 0, :], dim=1)[0]
    ymin = torch.min(normalized[:, 1, :], dim=1)[0]
    xmax = torch.max(normalized[:, 0, :], dim=1)[0]
    ymax = torch.max(normalized[:, 1, :], dim=1)[0]

    w = xmax - xmin + 1
    h = ymax - ymin + 1
    angle = angle
    x_center = center[:, 0, 0]
    y_center = center[:, 1, 0]

    rec = torch.stack([x_center, y_center, w, h, angle], dim=-1)

    # inds = w < h
    # rec[inds, 2] = h[inds]
    # rec[inds, 3] = w[inds]
    # rec[inds, 4] = rec[inds, 4] - np.pi / 2

    return rec

def rbboxRec2Poly_v2(rbboxes, max_shape=None):
    x_center = rbboxes[:, 0::5]
    y_center = rbboxes[:, 1::5]
    w = rbboxes[:, 2::5] - 1
    h = rbboxes[:, 3::5] - 1
    theta = rbboxes[:, 4::5]

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    x1 = costheta * (- w / 2.0) - sintheta * (- h / 2.0) + x_center
    y1 = sintheta * (-w / 2.0) + costheta * (- h /2.0) + y_center
    x2 = costheta * (w / 2.0) - sintheta * (-h / 2.0) + x_center
    y2 = sintheta * (w / 2.0) + costheta * (-h / 2.0) + y_center
    x3 = costheta * (w / 2.0) - sintheta * (h / 2.0) + x_center
    y3 = sintheta * (w / 2.0) + costheta * (h / 2.0) + y_center
    x4 = costheta * (- w / 2.0) - sintheta * (h / 2.0) + x_center
    y4 = sintheta * (- w / 2.0) + costheta * (h / 2.0) + y_center

    if max_shape != None:
        poly = torch.cat([x1.clamp(min=0, max=max_shape[1] - 1),
                          y1.clamp(min=0, max=max_shape[0] - 1),
                          x2.clamp(min=0, max=max_shape[1] - 1),
                          y2.clamp(min=0, max=max_shape[0] - 1),
                          x3.clamp(min=0, max=max_shape[1] - 1),
                          y3.clamp(min=0, max=max_shape[0] - 1),
                          x4.clamp(min=0, max=max_shape[1] - 1),
                          y4.clamp(min=0, max=max_shape[0] - 1)], dim=-1)
    else:
        poly = torch.cat([x1, y1, x2, y2, x3, y3, x4, y4], dim=-1)
    if rbboxes.size(1) != 5:
        poly = poly.view(poly.size(0), 8, -1)
        poly = poly.permute(0, 2, 1)
        poly = poly.contiguous().view(poly.size(0), -1)

    return poly


def get_new_begin_point_v1(rbboxes):
    rbboxes_new = rbboxes.new_zeros(rbboxes.size())
    x = rbboxes[:, 0::2]
    xmin_ind = torch.argmin(x, dim=1)
    for i in range(len(rbboxes)):
        for j in range(4):
            rbboxes_new[i, 2 * j] = rbboxes[i, (2 * xmin_ind[i] + 2 * j) % 8]
            rbboxes_new[i, 2 * j + 1] = rbboxes[i, (2 * xmin_ind[i] + 2 * j + 1) % 8]
    return rbboxes_new

def rec2target_v1(hbbox_rec, gt_rbbox_rec, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    hbbox_w = hbbox_rec[:, 2]
    hbbox_h = hbbox_rec[:, 3]
    hbbox_theta = hbbox_rec[:, 4]

    gt_rbbox_w = gt_rbbox_rec[:, 2]
    gt_rbbox_h = gt_rbbox_rec[:, 3]
    gt_rbbox_theta = gt_rbbox_rec[:, 4]

    delta_theta = gt_rbbox_theta - hbbox_theta

    delta_w = gt_rbbox_w / hbbox_w
    delta_h = gt_rbbox_h / hbbox_h

    t11 = torch.cos(delta_theta) * delta_w
    t12 = -torch.sin(delta_theta) * delta_h
    t21 = torch.sin(delta_theta) * delta_w
    t22 = torch.cos(delta_theta) * delta_h

    t = torch.stack([t11, t12, t21, t22], dim=-1)

    means = t.new_tensor(means).unsqueeze(0)
    stds = t.new_tensor(stds).unsqueeze(0)
    t = t.sub_(means).div_(stds)

    return t

def rec2target_v2(hbbox_rec, gt_rbbox_rec, means=[0, 0, 0], stds=[1, 1, 1]):
    hbbox_w = hbbox_rec[:, 2]
    hbbox_h = hbbox_rec[:, 3]
    hbbox_theta = hbbox_rec[:, 4]

    gt_rbbox_w = gt_rbbox_rec[:, 2]
    gt_rbbox_h = gt_rbbox_rec[:, 3]
    gt_rbbox_theta = gt_rbbox_rec[:, 4]

    delta_theta = gt_rbbox_theta - hbbox_theta

    delta_w = torch.log(gt_rbbox_w / hbbox_w)
    delta_h = torch.log(gt_rbbox_h / hbbox_h)

    t1 = torch.sin(delta_theta)

    t = torch.stack([t1, delta_w, delta_h], dim=-1)

    means = t.new_tensor(means).unsqueeze(0)
    stds = t.new_tensor(stds).unsqueeze(0)
    t = t.sub_(means).div_(stds)

    return t

def rec2target_v3(hbbox_rec, gt_rbbox_rec, means=[0, 0, 0], stds=[1, 1, 1]):
    hbbox_w = hbbox_rec[:, 2]
    hbbox_h = hbbox_rec[:, 3]
    hbbox_theta = hbbox_rec[:, 4]

    gt_rbbox_w = gt_rbbox_rec[:, 2]
    gt_rbbox_h = gt_rbbox_rec[:, 3]
    gt_rbbox_theta = gt_rbbox_rec[:, 4]

    delta_theta = gt_rbbox_theta - hbbox_theta

    delta_w = torch.log(gt_rbbox_w / hbbox_w)
    delta_h = torch.log(gt_rbbox_h / hbbox_h)

    t = torch.stack([delta_theta, delta_w, delta_h], dim=-1)

    means = t.new_tensor(means).unsqueeze(0)
    stds = t.new_tensor(stds).unsqueeze(0)
    t = t.sub_(means).div_(stds)

    return t

def rec2target_v4(rbbox_rec, gt_rbbox_rec, means=[0, 0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1, 1]):
    rbbox_x = rbbox_rec[:, 0]
    rbbox_y = rbbox_rec[:, 1]
    rbbox_w = rbbox_rec[:, 2]
    rbbox_h = rbbox_rec[:, 3]
    rbbox_theta = rbbox_rec[:, 4]

    gt_rbbox_x = gt_rbbox_rec[:, 0]
    gt_rbbox_y = gt_rbbox_rec[:, 1]
    gt_rbbox_w = gt_rbbox_rec[:, 2]
    gt_rbbox_h = gt_rbbox_rec[:, 3]
    gt_rbbox_theta = gt_rbbox_rec[:, 4]

    delta_x = (gt_rbbox_x - rbbox_x) / rbbox_w
    delta_y = (gt_rbbox_y - rbbox_y) / rbbox_h

    delta_theta = gt_rbbox_theta - rbbox_theta

    delta_w = gt_rbbox_w / rbbox_w
    delta_h = gt_rbbox_h / rbbox_h

    t11 = torch.cos(delta_theta) * delta_w
    t12 = -torch.sin(delta_theta) * delta_h
    t21 = torch.sin(delta_theta) * delta_w
    t22 = torch.cos(delta_theta) * delta_h

    t = torch.stack([delta_x, delta_y, t11, t12, t21, t22], dim=-1)

    means = t.new_tensor(means).unsqueeze(0)
    stds = t.new_tensor(stds).unsqueeze(0)
    t = t.sub_(means).div_(stds)

    return t

def delta2hbboxrec(rois,
                   deltas,
                   means=[0, 0, 0, 0],
                   stds=[1, 1, 1, 1],
                   wh_ratio_clip=16/1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means  # 在bbox2delta中进行了标准化，这里要做逆变换
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    rec = torch.stack([gx, gy, gw, gh], dim=-1).view_as(deltas)
    return rec

def delta2hbboxrec5(rois,
                   deltas,
                   means=[0, 0, 0, 0],
                   stds=[1, 1, 1, 1],
                   wh_ratio_clip=16/1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means  # 在bbox2delta中进行了标准化，这里要做逆变换
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    gtheta = gw.new_zeros((gw.size(0), gw.size(1)))
    rec = torch.stack([gx, gy, gw, gh, gtheta], dim=-1).view(deltas.size(0), -1)
    return rec

def target2poly_v1(hbboxes,
                 obb_pred,
                 max_shape,
                 means=[0, 0, 0, 0],
                 stds=[1, 1, 1, 1]):
    means = obb_pred.new_tensor(means).repeat(1, obb_pred.size(1) // 4)
    stds = obb_pred.new_tensor(stds).repeat(1, obb_pred.size(1) // 4)
    deform_obb_pred = obb_pred * stds + means

    t11 = deform_obb_pred[:, 0::4]
    t12 = deform_obb_pred[:, 1::4]
    t21 = deform_obb_pred[:, 2::4]
    t22 = deform_obb_pred[:, 3::4]

    x_center = hbboxes[:, 0::4]
    y_center = hbboxes[:, 1::4]
    w = hbboxes[:, 2::4] - 1
    h = hbboxes[:, 3::4] - 1

    x1 = (-w / 2.0) * t11 + (-h / 2.0) * t12 + x_center
    y1 = (-w / 2.0) * t21 + (-h / 2.0) * t22 + y_center
    x2 = (w / 2.0) * t11 + (-h / 2.0) * t12 + x_center
    y2 = (w / 2.0) * t21 + (-h / 2.0) * t22 + y_center
    x3 = (w / 2.0) * t11 + (h / 2.0) * t12 + x_center
    y3 = (w / 2.0) * t21 + (h / 2.0) * t22 + y_center
    x4 = (-w / 2.0) * t11 + (h / 2.0) * t12 + x_center
    y4 = (-w / 2.0) * t21 + (h / 2.0) * t22 + y_center

    poly = torch.cat([x1.clamp(min=0, max=max_shape[1] - 1),
                      y1.clamp(min=0, max=max_shape[0] - 1),
                      x2.clamp(min=0, max=max_shape[1] - 1),
                      y2.clamp(min=0, max=max_shape[0] - 1),
                      x3.clamp(min=0, max=max_shape[1] - 1),
                      y3.clamp(min=0, max=max_shape[0] - 1),
                      x4.clamp(min=0, max=max_shape[1] - 1),
                      y4.clamp(min=0, max=max_shape[0] - 1)], dim=-1)
    if obb_pred.size(1) != 4:
        poly = poly.view(poly.size(0), 8, -1)
        poly = poly.permute(0, 2, 1)
        poly = poly.contiguous().view(poly.size(0), -1)

    return poly

def target2poly_v1_circle(hbboxes,
                 obb_pred,
                 max_shape,
                 means=[0, 0, 0, 0],
                 stds=[1, 1, 1, 1]):
    means = obb_pred.new_tensor(means).repeat(1, obb_pred.size(1) // 4)
    stds = obb_pred.new_tensor(stds).repeat(1, obb_pred.size(1) // 4)
    deform_obb_pred = obb_pred * stds + means

    t11 = deform_obb_pred[:, 0::4]
    t12 = deform_obb_pred[:, 1::4]
    t21 = deform_obb_pred[:, 2::4]
    t22 = deform_obb_pred[:, 3::4]

    # 强制storage-tank(10)和roundant(12)水平
    t11[:, 10] = 1
    t11[:, 12] = 1
    t12[:, 10] = 0
    t12[:, 12] = 0
    t21[:, 10] = 0
    t21[:, 12] = 0
    t22[:, 10] = 1
    t22[:, 12] = 1

    x_center = hbboxes[:, 0::4]
    y_center = hbboxes[:, 1::4]
    w = hbboxes[:, 2::4] - 1
    h = hbboxes[:, 3::4] - 1

    x1 = (-w / 2.0) * t11 + (-h / 2.0) * t12 + x_center
    y1 = (-w / 2.0) * t21 + (-h / 2.0) * t22 + y_center
    x2 = (w / 2.0) * t11 + (-h / 2.0) * t12 + x_center
    y2 = (w / 2.0) * t21 + (-h / 2.0) * t22 + y_center
    x3 = (w / 2.0) * t11 + (h / 2.0) * t12 + x_center
    y3 = (w / 2.0) * t21 + (h / 2.0) * t22 + y_center
    x4 = (-w / 2.0) * t11 + (h / 2.0) * t12 + x_center
    y4 = (-w / 2.0) * t21 + (h / 2.0) * t22 + y_center

    poly = torch.cat([x1.clamp(min=0, max=max_shape[1] - 1),
                      y1.clamp(min=0, max=max_shape[0] - 1),
                      x2.clamp(min=0, max=max_shape[1] - 1),
                      y2.clamp(min=0, max=max_shape[0] - 1),
                      x3.clamp(min=0, max=max_shape[1] - 1),
                      y3.clamp(min=0, max=max_shape[0] - 1),
                      x4.clamp(min=0, max=max_shape[1] - 1),
                      y4.clamp(min=0, max=max_shape[0] - 1)], dim=-1)
    if obb_pred.size(1) != 4:
        poly = poly.view(poly.size(0), 8, -1)
        poly = poly.permute(0, 2, 1)
        poly = poly.contiguous().view(poly.size(0), -1)

    return poly

def target2poly_v2(hbboxes,
                 obb_pred,
                 max_shape,
                 means=[0, 0, 0],
                 stds=[1, 1, 1],
                 wh_ratio_clip=16 / 1000):
    means = obb_pred.new_tensor(means).repeat(1, obb_pred.size(1) // 3)
    stds = obb_pred.new_tensor(stds).repeat(1, obb_pred.size(1) // 3)
    deform_obb_pred = obb_pred * stds + means

    delta_sin = deform_obb_pred[:, 0::3]
    delta_w = deform_obb_pred[:, 1::3]
    delta_h = deform_obb_pred[:, 2::3]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    delta_w = delta_w.clamp(min=-max_ratio, max=max_ratio)
    delta_h = delta_h.clamp(min=-max_ratio, max=max_ratio)
    delta_w = delta_w.exp()
    delta_h = delta_h.exp()
    delta_cos = torch.sqrt(1 - delta_sin * delta_sin)

    x_center = hbboxes[:, 0::4]
    y_center = hbboxes[:, 1::4]
    w = hbboxes[:, 2::4]
    h = hbboxes[:, 3::4]

    t11 = delta_cos * delta_w
    t12 = -delta_sin * delta_h
    t21 = delta_sin * delta_w
    t22 = delta_cos * delta_h

    x1 = (-w / 2.0) * t11 + (-h / 2.0) * t12 + x_center
    y1 = (-w / 2.0) * t21 + (-h / 2.0) * t22 + y_center
    x2 = (w / 2.0) * t11 + (-h / 2.0) * t12 + x_center
    y2 = (w / 2.0) * t21 + (-h / 2.0) * t22 + y_center
    x3 = (w / 2.0) * t11 + (h / 2.0) * t12 + x_center
    y3 = (w / 2.0) * t21 + (h / 2.0) * t22 + y_center
    x4 = (-w / 2.0) * t11 + (h / 2.0) * t12 + x_center
    y4 = (-w / 2.0) * t21 + (h / 2.0) * t22 + y_center

    poly = torch.cat([x1.clamp(min=0, max=max_shape[1] - 1),
                      y1.clamp(min=0, max=max_shape[0] - 1),
                      x2.clamp(min=0, max=max_shape[1] - 1),
                      y2.clamp(min=0, max=max_shape[0] - 1),
                      x3.clamp(min=0, max=max_shape[1] - 1),
                      y3.clamp(min=0, max=max_shape[0] - 1),
                      x4.clamp(min=0, max=max_shape[1] - 1),
                      y4.clamp(min=0, max=max_shape[0] - 1)], dim=-1)
    poly = poly.view(poly.size(0), 8, -1)
    poly = poly.permute(0, 2, 1)
    poly = poly.contiguous().view(poly.size(0), -1)

    return poly

def target2poly_v3(rbboxes,
                 obb_pred,
                 max_shape,
                 means=[0, 0, 0, 0, 0, 0],
                 stds=[1, 1, 1, 1, 1, 1]):
    means = obb_pred.new_tensor(means).repeat(1, obb_pred.size(1) // 6)
    stds = obb_pred.new_tensor(stds).repeat(1, obb_pred.size(1) // 6)
    deform_obb_pred = obb_pred * stds + means


    t11 = deform_obb_pred[:, 2::6]
    t12 = deform_obb_pred[:, 3::6]
    t21 = deform_obb_pred[:, 4::6]
    t22 = deform_obb_pred[:, 5::6]

    w = rbboxes[:, 2::5] - 1
    h = rbboxes[:, 3::5] - 1
    x_center = rbboxes[:, 0::5] + deform_obb_pred[:, 0::6] * w
    y_center = rbboxes[:, 1::5] + deform_obb_pred[:, 1::6] * h
    theta = rbboxes[:, 4::5]
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    x1_local = (-w / 2.0) * t11 + (-h / 2.0) * t12
    y1_local = (-w / 2.0) * t21 + (-h / 2.0) * t22
    x2_local = (w / 2.0) * t11 + (-h / 2.0) * t12
    y2_local = (w / 2.0) * t21 + (-h / 2.0) * t22
    x3_local = (w / 2.0) * t11 + (h / 2.0) * t12
    y3_local = (w / 2.0) * t21 + (h / 2.0) * t22
    x4_local = (-w / 2.0) * t11 + (h / 2.0) * t12
    y4_local = (-w / 2.0) * t21 + (h / 2.0) * t22

    x1 = costheta * x1_local - sintheta * y1_local + x_center
    y1 = sintheta * x1_local + costheta * y1_local + y_center
    x2 = costheta * x2_local - sintheta * y2_local + x_center
    y2 = sintheta * x2_local + costheta * y2_local + y_center
    x3 = costheta * x3_local - sintheta * y3_local + x_center
    y3 = sintheta * x3_local + costheta * y3_local + y_center
    x4 = costheta * x4_local - sintheta * y4_local + x_center
    y4 = sintheta * x4_local + costheta * y4_local + y_center

    poly = torch.cat([x1.clamp(min=0, max=max_shape[1] - 1),
                      y1.clamp(min=0, max=max_shape[0] - 1),
                      x2.clamp(min=0, max=max_shape[1] - 1),
                      y2.clamp(min=0, max=max_shape[0] - 1),
                      x3.clamp(min=0, max=max_shape[1] - 1),
                      y3.clamp(min=0, max=max_shape[0] - 1),
                      x4.clamp(min=0, max=max_shape[1] - 1),
                      y4.clamp(min=0, max=max_shape[0] - 1)], dim=-1)
    if obb_pred.size(1) != 4:
        poly = poly.view(poly.size(0), 8, -1)
        poly = poly.permute(0, 2, 1)
        poly = poly.contiguous().view(poly.size(0), -1)

    return poly


def enlarge_bridge(gt_rbboxes_poly_list, gt_labels_list, **kwargs):
    func = partial(enlarge_bridge_single, **kwargs) if kwargs else enlarge_bridge_single
    gt_rbboxes_rec_adjust = map(func, gt_rbboxes_poly_list, gt_labels_list)
    return list(gt_rbboxes_rec_adjust)


def enlarge_bridge_single(gt_rbboxes_poly,
                        gt_labels,
                        w_enlarge=1.2,
                        h_enlarge=1.4,
                        max_shape=None):
    # 对于bridge类别，调整gt_rbbox，长边 * 1.2， 短边 * 1.4
    gt_rbboxes_rec = rbboxPoly2Rectangle_v2(gt_rbboxes_poly)
    inds_bridge = (gt_labels == 3)
    inds1 = (gt_rbboxes_rec[:, 2] > gt_rbboxes_rec[:, 3])
    inds1 = (inds_bridge + inds1) == 2
    gt_rbboxes_rec[inds1, 2] = gt_rbboxes_rec[inds1, 2] * w_enlarge
    gt_rbboxes_rec[inds1, 3] = gt_rbboxes_rec[inds1, 3] * h_enlarge
    inds2 = (gt_rbboxes_rec[:, 2] <= gt_rbboxes_rec[:, 3])
    inds2 = (inds_bridge + inds2) == 2
    gt_rbboxes_rec[inds2, 2] = gt_rbboxes_rec[inds2, 2] * h_enlarge
    gt_rbboxes_rec[inds2, 3] = gt_rbboxes_rec[inds2, 3] * w_enlarge
    gt_rbboxes_poly_new = rbboxRec2Poly_v2(gt_rbboxes_rec, max_shape)
    return gt_rbboxes_poly_new

def shrink_bridge_single(det_bboxes,
                         det_labels,
                         max_shape=None,
                         w_enlarge=1.2,
                         h_enlarge=1.4):
    if len(det_bboxes) == 0:
        return det_bboxes
    det_bboxes_rec = rbboxPoly2Rectangle_v2(det_bboxes[:, :8])
    inds_bridge = (det_labels == 2)
    inds1 = (det_bboxes_rec[:, 2] > det_bboxes_rec[:, 3])
    inds1 = (inds_bridge + inds1) == 2
    det_bboxes_rec[inds1, 2] = det_bboxes_rec[inds1, 2] / w_enlarge
    det_bboxes_rec[inds1, 3] = det_bboxes_rec[inds1, 3] / h_enlarge
    inds2 = (det_bboxes_rec[:, 2] <= det_bboxes_rec[:, 3])
    inds2 = (inds_bridge + inds2) == 2
    det_bboxes_rec[inds2, 2] = det_bboxes_rec[inds2, 2] / h_enlarge
    det_bboxes_rec[inds2, 3] = det_bboxes_rec[inds2, 3] / w_enlarge
    det_bboxes_poly_new = rbboxRec2Poly_v2(det_bboxes_rec, max_shape)
    return torch.cat([det_bboxes_poly_new, det_bboxes[:, 8, None]], dim=-1)


def poly2bbox(polys):
    """
    without label
    :param polys: (x1, y1, ..., x4, y4) (n, 8)
    :return: boxes: (xmin, ymin, xmax, ymax) (n, 4)
    """
    n = polys.shape[0]
    xs = np.reshape(polys, (n, 4, 2))[:, :, 0]
    ys = np.reshape(polys, (n, 4, 2))[:, :, 1]

    xmin = np.min(xs, axis=1)
    ymin = np.min(ys, axis=1)
    xmax = np.max(xs, axis=1)
    ymax = np.max(ys, axis=1)

    xmin = xmin[:, np.newaxis]
    ymin = ymin[:, np.newaxis]
    xmax = xmax[:, np.newaxis]
    ymax = ymax[:, np.newaxis]

    return np.concatenate((xmin, ymin, xmax, ymax), 1)

def rbboxPoly2rroiRec(rbbox_list):
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
            bboxes = rbboxPoly2Rectangle_v2(bboxes[:, :8])
            rrois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rrois = bboxes.new_zeros((0, 6))
        rrois_list.append(rrois)
    rrois = torch.cat(rrois_list, 0)
    return rrois

def hbbox2rec(hbboxes):
    if hbboxes is None:
        return None
    num_hbboxes = hbboxes.size(0)
    w = hbboxes[..., 2] - hbboxes[..., 0] + 1.0
    h = hbboxes[..., 3] - hbboxes[..., 1] + 1.0
    x_center = hbboxes[..., 0] + 0.5 * (w - 1.0)
    y_center = hbboxes[..., 1] + 0.5 * (h - 1.0)
    hbboxes_rec = torch.cat(
        (x_center.unsqueeze(1), y_center.unsqueeze(1), w.unsqueeze(1), h.unsqueeze(1)), 1)
    return hbboxes_rec


if __name__ == '__main__':
    rbbox = torch.Tensor([[1,1,2,1,2,8,1,8],
                          [1,2,2,1,10,5,9,6]])
    label=torch.Tensor([3,0])
    print(enlarge_bridge_single(rbbox, label))
