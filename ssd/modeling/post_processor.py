import torch

from ssd.utils.nms import boxes_nms
from ssd.utils.nms import quad_boxes_nms
import cv2
import numpy as np
class PostProcessor:
    def __init__(self,
                 iou_threshold,
                 score_threshold,
                 image_size,
                 max_per_class=200,
                 max_per_image=-1):
        self.confidence_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.width = image_size
        self.height = image_size
        self.max_per_class = max_per_class
        self.max_per_image = max_per_image

    def __call__(self, confidences, locations, width=None, height=None, batch_ids=None):
        """filter result using nms
        Args:
            confidences: (batch_size, num_priors, num_classes)
            locations: (batch_size, num_priors, 4)
            width(int): un-normalized using width
            height(int): un-normalized using height
            batch_ids: which batch to filter ?
        Returns:
            List[(boxes, labels, scores)],
            boxes: (n, 4)
            labels: (n, )
            scores: (n, )
        """
        # print('width:',width)
        # print('height:',height)
        if width is None:
            width = self.width
        if height is None:
            height = self.height

        batch_size = confidences.size(0)
        if batch_ids is None:
            batch_ids = torch.arange(batch_size, device=confidences.device)
        else:
            batch_ids = torch.tensor(batch_ids, device=confidences.device)

        locations = locations[batch_ids]

        confidences = confidences[batch_ids]
        # print('confidences.size():',confidences.size())
        # print('locations.size():',locations.size())


        results = []
        for decoded_boxes, scores in zip(locations, confidences):
            # per batch
            filtered_boxes = []
            filtered_labels = []
            filtered_probs = []
            for class_index in range(1, scores.size(1)):
                probs = scores[:, class_index]
                # print('self.confidence_threshold:',self.confidence_threshold)
                mask = probs > self.confidence_threshold
                # print('mask:',mask)
                # print('probs before mask:',probs[0:20])
                probs = probs[mask]

                # print('probs:',probs)
                if probs.size(0) == 0:
                    continue
                boxes = decoded_boxes[mask, :]
                # print('width:',width)
                boxes[:, 0] *= width
                boxes[:, 2] *= width
                boxes[:, 4] *= width
                boxes[:, 6] *= width

                boxes[:, 1] *= height
                boxes[:, 3] *= height
                boxes[:, 5] *= height
                boxes[:, 7] *= height

                keep = quad_boxes_nms(boxes, probs, self.iou_threshold, self.max_per_class)
                boxes = boxes[keep, :]
                probs = probs[keep]

                filtered_boxes.append(boxes)
                filtered_probs.append(probs)

            # no object detected
            if len(filtered_boxes) == 0:
                filtered_boxes = torch.empty(0, 4)
                filtered_probs = torch.empty(0)
            else:  # cat all result
                filtered_boxes = torch.cat(filtered_boxes, 0)
                filtered_probs = torch.cat(filtered_probs, 0)
            if 0 < self.max_per_image < filtered_probs.size(0):
                keep = torch.argsort(filtered_probs, dim=0, descending=True)[:self.max_per_image]
                filtered_boxes = filtered_boxes[keep, :]
                filtered_probs = filtered_probs[keep]
            results.append((filtered_boxes, filtered_probs))
        return results
