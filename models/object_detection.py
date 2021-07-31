# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
object_detection.py
Created on May 03 2020 19:38
object detection using detectron
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools
import torch
from torch.nn import functional as F
import torchvision
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model, postprocessing
from detectron2.structures import Boxes, Instances
from detectron2.layers import batched_nms


def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Override default method at detectron2/modeling/roi_heads/fast_rcnn.py

    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).
    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.
    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    thresholds = np.array([1, 0.5, 0.25]) * score_thresh
    for score_thresh in thresholds:
        # Filter results based on detection scores
        filter_mask = scores > score_thresh  # R x K
        # R' x 2. First column contains indices of the R predictions;
        # Second column contains indices of classes.
        filter_inds = filter_mask.nonzero()
        if num_bbox_reg_classes == 1:
            boxes_filtered = boxes[filter_inds[:, 0], 0]
        else:
            boxes_filtered = boxes[filter_mask]
        scores_filtered = scores[filter_mask]

        # Apply per-class NMS
        keep = batched_nms(boxes_filtered, scores_filtered, filter_inds[:, 1], nms_thresh)
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes_filtered, scores_filtered, filter_inds = boxes_filtered[keep], scores_filtered[keep], filter_inds[keep]
        if len(scores_filtered) > 0:
            break

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes_filtered)
    result.scores = scores_filtered
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


def default_cfg(model_name='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', threshold=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    return cfg


def get_maskrcnn_annotations(model_outputs, max_obj_num=0, shuffle=False):
    """convert detectron output of a maskrcnn model to dictionary:
    field include 'score', 'class', 'mask', 'box', 'count'
    :param max_obj_num  if positive, constraint the output to maximum number of objects per image
    """
    out = []
    for model_out in model_outputs:
        instances = model_out['instances']
        ins_classes_ = instances.pred_classes.to('cpu').numpy()
        ins_scores_ = instances.scores.to('cpu').numpy()
        ins_boxes_ = instances.pred_boxes.to('cpu').tensor.numpy().squeeze() +\
                      np.zeros((1, 1))
        ins_masks_ = instances.pred_masks.to('cpu').numpy()
        # remove too small boxes
        sizes_ = ins_boxes_[:, 2:] - ins_boxes_[:, :2]
        valid_ids = np.where(np.prod(sizes_ >= 2, axis=1))[0]
        if max_obj_num and len(valid_ids) > max_obj_num:
            sel_id = np.argsort(-ins_scores_[valid_ids])[:max_obj_num]
            count = max_obj_num
        else:
            count = len(valid_ids)
            sel_id = np.arange(count)
        ins_output = {'score': ins_scores_[valid_ids][sel_id],  # [N,]
                      'class': ins_classes_[valid_ids][sel_id],  # [N,]
                      'box': ins_boxes_[valid_ids][sel_id],  # [Nx4]
                      'mask': ins_masks_[valid_ids][sel_id],  # [NxHxW]
                      'count': count}  # [scalar]
        if 'box_features' in model_out:
            ins_output['box_features'] = model_out['box_features'].to('cpu').numpy()[valid_ids][sel_id]
        if shuffle and count > 1:  # random shuffle object order
            rnd_id = np.random.permutation(count)
            ins_output = {key: val[rnd_id] for key, val in ins_output.items() if key != 'count'}
            ins_output['count'] = count
        out.append(ins_output)
    return out


def get_geo_relation(annotations, img_sizes):
    """
    extract geo and inter-relation features from detected bounding boxes
    ref: Dipu etal. "Learning Structural Similarity of User Interface
    Layouts using Graph Networks". ECCV 2020.  
    :param  annotations  list of object detection results, one for each image (output of get_maskrcnn_annotations)
    :param  img_sizes    list of (h, w) of input images
    output: [(geo, rela, edge),...] list of tuple, one for each image. geo has size Nx5, 
                rela is N**2x7, edge has size N**2x2
    """
    out = []
    for annotation, img_size in zip(annotations, img_sizes):
        boxes = annotation['box']
        h0, w0 = img_size[0], img_size[1]
        # geo
        cx, cy = (boxes[:, 2] + boxes[:, 0])/2, (boxes[:, 3] + boxes[:, 1])/2
        w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]  # N,
        geo = [cx/w0, cy/h0, w/w0, h/h0, w*h/(w0*h0)]
        geo = np.array(geo).T  # Nx5

        # relation
        dx = cx[None, :] - cx[:, None]  # delta x ij, NxN
        dy = cy[None, :] - cy[:, None]  # delta y ij, NxN

        r1 = np.arctan2(dy, dx)  # rela1, angle between ij, NxN
        ai = np.sqrt(w*h)[:, None]  # Ai, Nx1
        r2 = dx / ai  # rela2, ratio of delta x_ij / area i, NxN
        r3 = dy / ai  # rela3, ratio of delta y_ij / area_i, NxN
        r4 = w[None, :] / w[:, None]  # rela4, ratio of width j/i, NxN
        r5 = h[None, :] / h[:, None]  # rela5, ratio of height j/i, NxN
        r6 = np.sqrt(dx*dx + dy*dy) / np.sqrt(h0*h0 + w0*w0)  # rela6, distance normalised against diag, NxN
        r7 = ai.T/ai  # rela6, ratio of area, NxN
        ox = np.minimum(boxes[:, 2][:, None], boxes[:, 2][None, :]) - \
            np.maximum(boxes[:, 0][:, None], boxes[:, 0][None, :])
        ox = np.where(ox > 0, ox, 0)  # overlapping along x, NxN
        oy = np.minimum(boxes[:, 3][:, None], boxes[:, 3][None, :]) - \
            np.maximum(boxes[:, 1][:, None], boxes[:, 1][None, :])
        oy = np.where(oy > 0, oy, 0)  # overlapping along y, NxN
        intersection = ox*oy  # intersection ij, NxN
        union = (w*h)[:, None] + (w*h)[None, :] - intersection  # union ij, NxN
        r0 = intersection / union  # IOU, NxN
        rela = np.array([r0, r1, r2, r3, r4, r5, r6, r7]).transpose(1,2,0).reshape((-1, 8))

        # edges is simply catersian product
        inds = np.arange(len(boxes))
        edge = np.array(list(itertools.product(inds, inds)))  # N*2, 2
        out.append((geo, rela, edge))
    return out


def join_dict(dict_list):
    """
    convert list of dict into dict of list
    assume all dicts have same keys and value types
    """
    # out = {key: np.array([adict[key] for adict in dict_list]) for key in dict_list[0]}
    out = {key: np.empty(len(dict_list), dtype=object) for key in dict_list[0]}
    for key in dict_list[0]:
        out[key][:] = [adict[key] for adict in dict_list]
    out['count'] = out['count'].astype(np.int64)
    return out


def add_background(prediction, background_id):
    """add background as an extra object"""
    out = {}
    out['count'] = prediction['count'] + 1  # background is one object
    out['score'] = np.append(prediction['score'], 1.0)
    out['class'] = np.append(prediction['class'], background_id)
    h, w = prediction['mask'].shape[-2:]
    out['box'] = np.r_[prediction['box'], np.array([0., 0., w, h], dtype=np.float32)[None, ...]]
    background_mask = ~np.sum(prediction['mask'], axis=0, keepdims=True).astype(np.bool)
    out['mask'] = np.concatenate([prediction['mask'], background_mask])
    return out


class BatchPredictor(object):
    """
    object detection with detectron

    Example Usage:
    cfg_file = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
	cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(hps.model_name)
    predictor = BatchPredictor(cfg)
    meta = predictor.metadata
	
    img = cv2.imread('example.jpg', cv2.IMREAD_COLOR)
    res = predictor([img,])[0]

    """
    def __init__(self, cfg, batch_size=8, feat_level=2, include_big_box=1):
        """
        :param cfg               detectron2 config object (from detectron2.config.get_cfg)
        :param batch_size        input images to be split to minibatchs internally
        :param feat_level        0 if not return features, 1 if pull features after box_pooler, 2 if after box_heads()
        :param include_big_box   1: add whole image as an extra object, 2: add background, 0: not adding extra object
        """
        detectron2.modeling.roi_heads.fast_rcnn.fast_rcnn_inference_single_image = fast_rcnn_inference_single_image
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.batch_size = batch_size
        self.device = torch.device('cuda:0')  # detectron2 always use gpu0
        self.include_big_box = include_big_box
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.ncats = len(self.metadata.thing_classes)
        if self.include_big_box:
            self.ncats += 1
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
        assert feat_level in [0, 1, 2], "[Object detection] feat level %d not supported." % feat_level
        self.feat_level = feat_level
        # memory holder
        # dummy = [np.ones((cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, 3), dtype=np.uint8)] * batch_size
        # _ = self.__call__(dummy)

    def __call__(self, original_images):
        """
        Args:
            original_images (list of np.ndarray): list of images of shape (Hi, Wi, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_images = [org_img[:, :, ::-1] for org_img in original_images]
        inputs = []
        for original_image in original_images:
            height, width = original_image.shape[:2]
            image = self.transform_gen.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs.append({"image": image, "height": height, "width": width})
        predictions = []
        for i in range(0, len(original_images), self.batch_size):
            input_batch = inputs[i: min(i+self.batch_size, len(original_images))]
            predictions.extend(self.inference(input_batch))  # self.model(input_batch)
        return predictions
    
    def inference(self, batch_ims):
        bsz = len(batch_ims)
        with torch.no_grad():
            images = self.model.preprocess_image(batch_ims)
            features = self.model.backbone(images.tensor)  # set of cnn features
            proposals, _ = self.model.proposal_generator(images, features, None)  # RPN
            # ROI pooling
            if self.feat_level == 0:  # no feature extraction, use built-in roi_heads
                pred_instances, _ = self.model.roi_heads(images, features, proposals, None)
                if self.include_big_box:
                    pred_instances = self.add_big_box(pred_instances)
                results = self.model._postprocess(pred_instances, batch_ims, images.image_sizes)  # scale box to orig size
            else:
                pred_instances, box_features = self.roi_heads(features, proposals)
                results = self.model._postprocess(pred_instances, batch_ims, images.image_sizes)  # scale box to orig size
                # attach box features
                for i in range(bsz):
                    results[i].update(box_features=box_features[i])
        if self.include_big_box == 2:  # replace whole image mask with background mask
            for res in results:
                if len(res['instances'].pred_classes) > 1:  # at least 1 object other than whole image
                    res['instances'].pred_masks[-1] = res['instances'].pred_masks[:-1].sum(dim=0) == 0
        return results

    # def inference_panoptic(self, batch_ims):
    #     """
    #     not working for now
    #     """
    #     bsz = len(batch_ims)
    #     with torch.no_grad():
    #         images = self.model.preprocess_image(batch_ims)  # exactly same preprocess as in instance seg
    #         features = self.model.backbone(images.tensor)  # set of cnn features
    #         gt_sem_seg = None
    #         sem_seg_results, sem_seg_losses = self.model.sem_seg_head(features, gt_sem_seg)
    #         gt_instances = None
    #         proposals, proposal_losses = self.model.proposal_generator(images, features, gt_instances)
    #         # detector_results, detector_losses = self.model.roi_heads(images, features, proposals, gt_instances)
    #         detector_results, box_features = self.roi_heads(features, proposals)
    #         processed_results = []
    #         for sem_seg_result, detector_result, input_per_image, image_size, box_feature in zip(
    #             sem_seg_results, detector_results, batched_inputs, images.image_sizes, box_features
    #         ):
    #             height = input_per_image.get("height", image_size[0])
    #             width = input_per_image.get("width", image_size[1])
    #             sem_seg_r = postprocessing.sem_seg_postprocess(sem_seg_result, image_size, height, width)
    #             detector_r = postprocessing.detector_postprocess(detector_result, height, width)

    #             processed_results.append({"sem_seg": sem_seg_r, "instances": detector_r})

    #             if self.model.combine_on:
    #                 panoptic_r = combine_semantic_and_instance_outputs(
    #                     detector_r,
    #                     sem_seg_r.argmax(dim=0),
    #                     self.model.combine_overlap_threshold,
    #                     self.model.combine_stuff_area_limit,
    #                     self.model.combine_instances_confidence_threshold,
    #                     box_feature
    #                 )
    #                 processed_results[-1]["panoptic_seg"] = panoptic_r
    #         return processed_results


    def roi_heads(self, features, proposals):
        """
        reimplement the inference logic of model roi_heads(), also return features
            of the detected boxes

        original roi_heads returns a list of Instances objects during inference:
        results, _ = self.model.roi_heads(images, features, proposals, None)

        this roi_heads will return results as above and a list of box features
        """
        try:
            bsz = len(proposals)
            features_ = [features[f] for f in self.model.roi_heads.in_features]
            # get proposal boxes
            if self.include_big_box:
                boxes = []
                for proposal in proposals:
                    y, x = proposal.image_size
                    big_box = torch.tensor([0., 0., x, y]).unsqueeze(0).to(self.device)
                    box = Boxes(torch.cat([proposal.proposal_boxes.tensor, big_box]))
                    boxes.append(box)  # append whole image box at the end
            else:
                boxes = [x.proposal_boxes for x in proposals]
            box_num = [box.tensor.shape[0] for box in boxes]

            # import pdb; pdb.set_trace()
            box_features = self.model.roi_heads.box_pooler(features_, boxes)
            if self.feat_level == 1:  # preserve box features lv1 from here
                box_features_out = F.adaptive_avg_pool2d(box_features, (1,1)).squeeze()
                box_features_out = list(torch.split(box_features_out, box_num, dim=0))
                
            box_features = self.model.roi_heads.box_head(box_features)
            if self.feat_level == 2:  # preserve box features lv2 from here
                box_features_out = list(torch.split(box_features, box_num, dim=0))
            if self.include_big_box:  # remove big box before feeding to predictor
                box_features = torch.split(box_features, box_num, dim=0)
                box_features = torch.cat([box_feature[:-1] for box_feature in box_features])  # back to normal

            predictions = self.model.roi_heads.box_predictor(box_features)
            pred_instances, pred_inds = self.model.roi_heads.box_predictor.inference(predictions, proposals)
            pred_instances = self.model.roi_heads.forward_with_given_boxes(features, pred_instances)
            if self.include_big_box:
                # get corresponding features
                for i in range(bsz):
                    box_feats = box_features_out[i][pred_inds[i]]
                    big_box_feat = box_features_out[i][-1].unsqueeze(0)
                    box_features_out[i] = torch.cat([box_feats, big_box_feat])
                # update pred_instances with big box
                pred_instances = self.add_big_box(pred_instances)
            else:
                box_features_out = [box_features_out[i][pred_inds[i]] for i in range(bsz)]
        except RuntimeError as e:
            print(e)
            # import pdb; pdb.set_trace()
        return pred_instances, box_features_out

    def add_big_box(self, instances):
        """
        add a box covering the whole image
        """
        out = []
        for i in range(len(instances)):
            tfields = instances[i].get_fields()
            y, x = instances[i].image_size
            big_box = torch.tensor([0., 0., x, y]).unsqueeze(0).to(self.device)
            tfields['pred_boxes'] = Boxes(torch.cat([tfields['pred_boxes'].tensor, big_box]))
            big_score = torch.tensor([1.0], dtype=torch.float32).to(self.device)
            tfields['scores'] = torch.cat([tfields['scores'], big_score])
            big_class = torch.tensor([self.ncats-1], dtype=torch.int64).to(self.device)
            tfields['pred_classes'] = torch.cat([tfields['pred_classes'], big_class])
            mask_size = tfields['pred_masks'].shape
            big_mask = torch.ones((1, mask_size[1], mask_size[2], mask_size[3]), dtype=torch.float32).to(self.device)
            tfields['pred_masks'] = torch.cat([tfields['pred_masks'], big_mask])
            out.append(Instances((y, x), **tfields))
        return out


def combine_semantic_and_instance_outputs(
    instance_results,
    semantic_results,
    overlap_threshold,
    stuff_area_limit,
    instances_confidence_threshold,
    box_features
):
    """
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/meta_arch/panoptic_fpn.py

    Implement a simple combining logic following
    "combine_semantic_and_instance_predictions.py" in panopticapi
    to produce panoptic segmentation outputs.
    Args:
        instance_results: output of :func:`detector_postprocess`.
        semantic_results: an (H, W) tensor, each is the contiguous semantic
            category id
    Returns:
        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
            Each dict contains keys "id", "category_id", "isthing".
    """
    panoptic_seg = torch.zeros_like(semantic_results, dtype=torch.int32)

    # sort instance outputs by scores
    sorted_inds = torch.argsort(-instance_results.scores)

    current_segment_id = 0
    segments_info = []

    instance_masks = instance_results.pred_masks.to(dtype=torch.bool, device=panoptic_seg.device)

    # Add instances one-by-one, check for overlaps with existing ones
    for inst_id in sorted_inds:
        score = instance_results.scores[inst_id].item()
        if score < instances_confidence_threshold:
            break
        mask = instance_masks[inst_id]  # H,W
        mask_area = mask.sum().item()

        if mask_area == 0:
            continue

        intersect = (mask > 0) & (panoptic_seg > 0)
        intersect_area = intersect.sum().item()

        if intersect_area * 1.0 / mask_area > overlap_threshold:
            continue

        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": True,
                "score": score,
                "category_id": instance_results.pred_classes[inst_id].item(),
                "instance_id": inst_id.item(),
                "box_features": box_features[inst_id],
                "mask": mask,
                "box": get_box_coordinate(mask)
            }
        )

    # Add semantic results to remaining empty areas
    semantic_labels = torch.unique(semantic_results).cpu().tolist()
    for semantic_label in semantic_labels:
        if semantic_label == 0:  # 0 is a special "thing" class
            continue
        mask = (semantic_results == semantic_label) & (panoptic_seg == 0)
        mask_area = mask.sum().item()
        if mask_area < stuff_area_limit:
            continue

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": False,
                "category_id": semantic_label,
                "area": mask_area
            }
        )

    return panoptic_seg, segments_info


def get_box_coordinate(mask):
    y, x = np.where(mask.cpu().numpy())
    return torch.tensor([x.min(), y.min(), x.max(), y.max()], dtype=torch.int32)
