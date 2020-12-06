import torch

num_classes = 16
anchor_num_cls = 10
anchor_num_reg = 20
model_coco = torch.load("../checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth")
# model_coco = torch.load("../work_dirs/al_detection/faster_rcnn_r50_fpn_hrsc2016/100/latest.pth")
for key, value in model_coco["state_dict"].items():
    print(key)

# # weight
model_coco["state_dict"]["bbox_head.fc_cls.weight"] = model_coco["state_dict"][
                                                                     "roi_head.bbox_head.fc_cls.weight"][:num_classes,
                                                                 :]
model_coco["state_dict"]["bbox_head.fc_reg.weight"] = model_coco["state_dict"][
                                                                     "roi_head.bbox_head.fc_reg.weight"][:num_classes*4, :]
model_coco["state_dict"]["bbox_head.shared_fcs.0.weight"] = model_coco["state_dict"][
                                                                     "roi_head.bbox_head.shared_fcs.0.weight"]
model_coco["state_dict"]["bbox_head.shared_fcs.1.weight"] = model_coco["state_dict"][
                                                                     "roi_head.bbox_head.shared_fcs.1.weight"]

# bias
model_coco["state_dict"]["bbox_head.fc_cls.bias"] = model_coco["state_dict"][
                                                                   "roi_head.bbox_head.fc_cls.bias"][:num_classes]
model_coco["state_dict"]["bbox_head.fc_reg.bias"] = model_coco["state_dict"][
                                                                     "roi_head.bbox_head.fc_reg.bias"][:num_classes*4]
model_coco["state_dict"]["bbox_head.shared_fcs.0.bias"] = model_coco["state_dict"][
                                                                     "roi_head.bbox_head.shared_fcs.0.bias"]
model_coco["state_dict"]["bbox_head.shared_fcs.1.bias"] = model_coco["state_dict"][
                                                                     "roi_head.bbox_head.shared_fcs.1.bias"]


# save new model
torch.save(model_coco, "../checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15_classes_%d.pth" % num_classes)