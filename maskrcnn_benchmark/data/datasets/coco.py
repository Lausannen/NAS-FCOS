# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import cv2

import os
import torch
import torchvision

from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, mode, resolution, backbone, post_branch, 
        remove_images_without_annotations, transforms=None,
        special_deal=False
    ):
        super(COCODataset, self).__init__(root, ann_file)
        self.mode = mode
        self.rank = get_rank()
        self.backbone = backbone
        self.transforms = transforms
        self.resolution = resolution
        self.post_branch = post_branch
        self.special_deal = special_deal
        self._img_search_set = os.path.join(ann_file, "%s.txt")

        self._split_feature = os.path.join(ann_file, "features_%s_%s", "%s.pth")
        self._split_target  = os.path.join(ann_file, self.post_branch, "targets_%s_%s", "%s.pth")
        self._split_label   = os.path.join(ann_file, self.post_branch, "labels_%s_%s", "%s.pth")
        self._split_reg     = os.path.join(ann_file, self.post_branch, "regs_%s_%s", "%s.pth")

        if self.mode == 0:
            # sort indices for reproducible results
            self.ids = sorted(self.ids)

            # filter images without detection annotations
            if remove_images_without_annotations:
                self.ids = [
                    img_id
                    for img_id in self.ids
                    if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
                ]

            self.json_category_id_to_contiguous_id = {
                v: i + 1 for i, v in enumerate(self.coco.getCatIds())
            }
            self.contiguous_category_id_to_json_id = {
                v: k for k, v in self.json_category_id_to_contiguous_id.items()
            }
            self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        elif self.mode == 1:
            try:
                image_set_name = "coco_%s_%s_%s_search" % (
                    self.backbone, self.resolution, self.post_branch
                )
                with open(self._img_search_set % image_set_name) as f:
                    self.ids = f.readlines()
                self.ids = [x.strip("\n") for x in enumerate(self.ids)]
            except FileNotFoundError:
                raise ValueError("Can not open file {}.txt".format(image_set_name))

        else:
            raise ValueError("Mode {} is not available for COCO Dataset".format(self.mode))

    def __getitem__(self, idx):
        if self.mode == 0:
            img, anno = super(COCODataset, self).__getitem__(idx)
            # filter crowd annotations
            # TODO might be better to add an extra field
            anno = [obj for obj in anno if obj["iscrowd"] == 0]

            boxes = [obj["bbox"] for obj in anno]
            boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
            target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

            classes = [obj["category_id"] for obj in anno]
            classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
            classes = torch.tensor(classes)
            target.add_field("labels", classes)

            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size)
            target.add_field("masks", masks)

            target = target.clip_to_image(remove_empty=True)

            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target, idx

        elif self.mode == 1:
            img_id = self.ids[index]
            feature_list = torch.load(self._split_feature % (self.backbone, self.resolution, img_id)
                            , map_location=torch.device("cpu"))
            if self.special_deal:
                label = torch.load(self._split_label % (self.backbone, self.resolution, img_id)
                                , map_location=torch.device("cpu"))
                reg = torch.load(self._split_reg % (self.backbone, self.resolution, img_id)
                                , map_location=torch.device("cpu"))
                return feature_list, label, reg, index
            else:
                target = torch.load(self._split_target % (self.backbone, self.resolution, img_id)
                                , map_location=torch.device("cpu"))
                return feature_list, target, index
        else:
            raise ValueError("Mode {} do not support now".format(self.mode))

    def compute_target_maps(self, targets_map, target_id, box):
        '''
        :param label:
        :param polygons:
        :param box:
        :param im_sizes: (h, w)
        :return:
        '''
        x0, y0, x1, y1 = box.view(-1).to(torch.int32).numpy()
        targets_map = cv2.rectangle(
            targets_map, (x0, y0), (x1, y1),
            color=target_id, thickness=cv2.FILLED
        )
        return targets_map

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
