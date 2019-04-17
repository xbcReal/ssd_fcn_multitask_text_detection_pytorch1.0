from ..transforms.transforms import *


class TrainAugmentation:
    def __init__(self, size, mean=0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size

        self.augment=Compose([
            ConvertFromInts(),
            # PhotometricDistort 在这里不能用于数据增强，会极大损害图像质量和文本区的图像质量
            Expand(self.mean),
            Rotate_small_angle(10),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            RandomRotate90_180_270(),
            #SubtractMeans(self.mean),
            # SubtractMeansAndDiv255(self.mean),
            ToTensor(),
        ])

    def __call__(self, img, quad,score_map):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
        """
        return self.augment(img, quad,score_map)


class TestTransform:
    def __init__(self, size, mean=0.0):
        self.transform = Compose([
            ToPercentCoords(),
            SubtractMeans(mean),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0):
        self.transform = Compose([
            Resize(size),
            #SubtractMeans(mean),
            # SubtractMeansAndDiv255(mean),
            ToTensor()
        ])

    def __call__(self, image):
        image, _,_= self.transform(image,None,None,False)
        return image
