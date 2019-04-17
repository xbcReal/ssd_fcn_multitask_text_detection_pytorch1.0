import torch

from ssd.modeling.post_processor import PostProcessor
from .data_preprocessing import PredictionTransform


class Predictor:
    def __init__(self, cfg, model, iou_threshold, score_threshold, device):
        self.cfg = cfg
        self.model = model
        self.transform = PredictionTransform(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.PIXEL_MEAN)
        self.post_processor = PostProcessor(iou_threshold=iou_threshold,
                                            score_threshold=score_threshold,
                                            image_size=cfg.INPUT.IMAGE_SIZE,
                                            max_per_class=cfg.TEST.MAX_PER_CLASS,
                                            max_per_image=cfg.TEST.MAX_PER_IMAGE)
        self.device = device
        self.model.eval()

    def predict(self, image):
        #height,width是原始图像的宽和高
        height, width, _ = image.shape
        # print('height:',height)
        # print('width:',width)
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            scores, quad ,seg_map= self.model(images,None,None)
            quad[:,:,0] = 1 - quad[:,:,0]
            quad[:,:,2] = 1 - quad[:,:,2]
            quad[:,:,4] = 1 - quad[:,:,4]
            quad[:,:,6] = 1 - quad[:,:,6]

        results = self.post_processor(scores, quad,width=width, height=height)
        boxes, scores = results[0]
        return boxes, scores,seg_map
