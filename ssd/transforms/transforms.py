# from https://github.com/amdegroot/ssd.pytorch


import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
import PIL.Image as Image
import math
def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
def jaccard_quad_box(quads, box):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    bouding_box_for_quads = None
    # inter = intersect(box_a, box_b)
    # area_a = ((box_a[:, 2] - box_a[:, 0]) *
    #           (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    # area_b = ((box_b[2] - box_b[0]) *
    #           (box_b[3] - box_b[1]))  # [A,B]
    # union = area_a + area_b - inter
    return inter / union  # [A,B]



class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, quad,score_map,train=True):
        for t in self.transforms:
            img, quad ,score_map= t(img, quad,score_map,train)
        return img, quad,score_map


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, quad=None):
        return self.lambd(img, quad)


class ConvertFromInts(object):
    def __call__(self, image, quad,score_map,train=True):
        return image.astype(np.float32), quad,score_map

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, quad=None):
        image = image.astype(np.float32)
        image -= self.mean
        # print(image)
        return image.astype(np.float32), quad


class SubtractMeansAndDiv255(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, quad=None,score_map=None,train=None):
        image = image.astype(np.float32)
        image -= self.mean
        image = image.astype(np.float32) / 255.0
        # print(image)

        return image , quad,score_map


class ToAbsoluteCoords(object):
    def __call__(self, image, quad=None):
        height, width, channels = image.shape
        quad[:, 0] *= width
        quad[:, 2] *= width
        quad[:, 1] *= height
        quad[:, 3] *= height

        return image, quad


class ToPercentCoords(object):
    def __call__(self, image, quad,score_map=None,train=True):
        height, width, channels = image.shape
        quad[:, [0,2,4,6]] /= width
        quad[:, [1,3,5,7]] /= height
        return image, quad,score_map


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, quad,score_map,train=True):
        #这里只resize图片而没有resize quad的原因是quad已经被转化为百分比的长度了,resize图片后quad占的百分比还是不变,所以这里只需要resize图片即可
        image = cv2.resize(image, (self.size,self.size))
        if train and score_map is not None:
            score_map = cv2.resize(score_map, (self.size,self.size))
        return image, quad,score_map


class RandomSaturation(object):
    def __init__(self, lower=0.9, upper=1.1):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, quad=None,score_map=None,train=True):
        if random.randint(2):
            x = random.uniform(self.lower, self.upper)
            image[:, :, 1] *= x

        return image, quad,score_map


class RandomHue(object):
    def __init__(self, delta=3.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, quad=None,score_map=None,train=True):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            #image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            #image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, quad,score_map


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, quad=None,score_map=None,train=True):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, quad,score_map


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, quad=None,score_map=None,train=True):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, quad,score_map


class RandomContrast(object):
    def __init__(self, lower=0.9, upper=1.1):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, quad=None,score_map=None,train=True):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, quad,score_map


class RandomBrightness(object):
    def __init__(self, delta=3):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, quad=None,score_map=None,train=True):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            # print('delta:',delta)
            image += delta
        return image, quad,score_map

#随机逆时针旋转90度
class RandomRotate90_180_270(object):
    def __init__(self):
        pass
    def __call__(self, image, quad,score_map,train=True):
        quad=np.array(quad)
        random_number=random.randint(4)
        if random_number==0:
            # image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            # for i in range(np.shape(quad)[0]):
            #     cv2.circle(image, (int(quad[i][0]*512), int(quad[i][1]*512)), 5, (0, 255, 0), 5)
            #     cv2.circle(image, (int(quad[i][2]*512), int(quad[i][3]*512)), 5, (255, 255, 255), 5)
            #     cv2.circle(image, (int(quad[i][4]*512), int(quad[i][5]*512)), 5, (255, 0, 0), 5)
            #     cv2.circle(image, (int(quad[i][6]*512), int(quad[i][7]*512)), 5, (0, 0, 255), 5)
            # cv2.imshow('img', image)
            # cv2.waitKey(0)
            pass
        elif random_number==1:
            #逆时针旋转90度
            image = np.rot90(image)
            if score_map is None:
                pass
            else:
                score_map = np.rot90(score_map)
            #image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            #因为逆时针旋转了90度之后需要交换四个点各自xy的坐标
            quad=np.concatenate((np.expand_dims(quad[:,1],1),np.expand_dims(1-quad[:,0],1),
                                 np.expand_dims(quad[:,3],1),np.expand_dims(1-quad[:,2],1),
                                 np.expand_dims(quad[:,5],1),np.expand_dims(1-quad[:,4],1),
                                 np.expand_dims(quad[:,7],1),np.expand_dims(1-quad[:,6],1)),axis=1)
            #因为逆时针旋转了90度之后，原来是第一个坐标的要变为最后一个坐标，原来是第234个坐标的点要变为第123个坐标
            quad=np.concatenate((quad[:,2:],quad[:,0:2]),axis=1)
            #print(np.shape(quad))
            #print(quad*512)
            # for i in range(np.shape(quad)[0]):
            #     cv2.circle(image, (int(quad[i][0]*512), int(quad[i][1]*512)), 5, (0, 255, 0), 5)
            #     cv2.circle(image, (int(quad[i][2]*512), int(quad[i][3]*512)), 5, (255, 255, 255), 5)
            #     cv2.circle(image, (int(quad[i][4]*512), int(quad[i][5]*512)), 5, (255, 0, 0), 5)
            #     cv2.circle(image, (int(quad[i][6]*512), int(quad[i][7]*512)), 5, (0, 0, 255), 5)
            # cv2.imshow('img', image)
            # cv2.waitKey(0)
        elif random_number==2:
            pass
            # 逆时针旋转180度
            # image = np.rot90(image)
            # image = np.rot90(image)
            # score_map = np.rot90(score_map)
            # score_map = np.rot90(score_map)
            # # image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            # # 因为逆时针旋转了180度之后需要交换四个点各自xy的坐标
            # quad = 1 - quad
            # # 因为逆时针旋转了180度之后，原来ABCD坐标的要变为CDAB
            # quad = np.concatenate((quad[:, 4:], quad[:, 0:4]), axis=1)
            # print(np.shape(quad))
            # print(quad*512)
            # for i in range(np.shape(quad)[0]):
            #     cv2.circle(image, (int(quad[i][0]*512), int(quad[i][1]*512)), 5, (0, 255, 0), 5)
            #     cv2.circle(image, (int(quad[i][2]*512), int(quad[i][3]*512)), 5, (255, 255, 255), 5)
            #     cv2.circle(image, (int(quad[i][4]*512), int(quad[i][5]*512)), 5, (255, 0, 0), 5)
            #     cv2.circle(image, (int(quad[i][6]*512), int(quad[i][7]*512)), 5, (0, 0, 255), 5)
            # cv2.imshow('img', image)
            # cv2.waitKey(0)
        elif random_number==3:
            # 逆时针旋转270度
            image = np.rot90(image)
            image = np.rot90(image)
            image = np.rot90(image)
            if score_map is None:
                pass
            else:
                score_map = np.rot90(score_map)
                score_map = np.rot90(score_map)
                score_map = np.rot90(score_map)
            # image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            # 因为逆时针旋转了270度之后需要交换四个点各自xy的坐标
            quad = np.concatenate((np.expand_dims(1 - quad[:, 1], 1), np.expand_dims(quad[:, 0], 1),
                                   np.expand_dims(1 - quad[:, 3], 1), np.expand_dims(quad[:, 2], 1),
                                   np.expand_dims(1 - quad[:, 5], 1), np.expand_dims(quad[:, 4], 1),
                                   np.expand_dims(1 - quad[:, 7], 1), np.expand_dims(quad[:, 6], 1)), axis=1)
            # 因为逆时针旋转了270度之后，原来ABCD坐标要变为DABC
            quad = np.concatenate((quad[:, 6:], quad[:, 0:6]), axis=1)
            # print(np.shape(quad))
            # print(quad*512)
            # for i in range(np.shape(quad)[0]):
            #     cv2.circle(image, (int(quad[i][0]*512), int(quad[i][1]*512)), 5, (0, 255, 0), 5)
            #     cv2.circle(image, (int(quad[i][2]*512), int(quad[i][3]*512)), 5, (255, 255, 255), 5)
            #     cv2.circle(image, (int(quad[i][4]*512), int(quad[i][5]*512)), 5, (255, 0, 0), 5)
            #     cv2.circle(image, (int(quad[i][6]*512), int(quad[i][7]*512)), 5, (0, 0, 255), 5)
            # cv2.imshow('img', image)
            # cv2.waitKey(0)
        return image,quad,score_map



class ToCV2Image(object):
    def __call__(self, tensor, quad=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), quad


class ToTensor(object):
    def __call__(self, cvimage, quad, score_map,train=True):
        if train:
            if score_map is None:
                return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), quad,score_map
            else:
                return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), quad,torch.from_numpy(score_map.astype(np.float32))
        else:
            return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), quad,score_map

class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, quad,score_map,train=True):
        # guard against no boxes
        if quad is not None and quad.shape[0] == 0:
            return image, quad
        image_copy=image.copy()
        quad_copy=quad.copy()
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, quad, score_map

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image
                if score_map is None:
                    current_score_map = None
                    pass
                else:
                    current_score_map = score_map
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                #overlap = jaccard_quad_box(quad, rect)
                #
                # # is min and max overlap constraint satisfied? if not try again
                # if overlap.min() < min_iou and max_iou < overlap.max():
                #     continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],:]
                if score_map is None:
                    pass
                else:
                    current_score_map = current_score_map[rect[1]:rect[3], rect[0]:rect[2]]
                # keep overlap with gt box IF center in sampled patch

                if len(np.shape(quad))==2:
                    centers_x = (quad[:, 0] + quad[:, 2] + quad[:, 4] + quad[:, 6]) / 4.0
                    centers_x=centers_x[np.newaxis,:]
                    centers_x=np.transpose(centers_x)
                    centers_y = (quad[:, 1] + quad[:, 3] + quad[:, 5] + quad[:,7]) / 4.0
                    centers_y=centers_y[np.newaxis,:]
                    centers_y=np.transpose(centers_y)
                    centers = np.concatenate([centers_x,centers_y],axis=1)
                else:
                    assert len(np.shape(quad))==1
                    centers_x = (quad[0] + quad[2] + quad[4] + quad[6]) / 4.0
                    centers_x = centers_x[np.newaxis, :]
                    centers_x = np.transpose(centers_x)
                    centers_y = (quad[1] + quad[3] + quad[5] + quad[7]) / 4.0
                    centers_y = centers_y[np.newaxis, :]
                    centers_y = np.transpose(centers_y)
                    centers = np.concatenate([centers_x, centers_y], axis=1)

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_quad = quad[mask, :].copy()

                # should we use the box left and top corner or the crop's
                current_quad[:,[0,2,4,6]] = np.maximum(current_quad[:, [0,2,4,6]],rect[0])
                current_quad[:,[0,2,4,6]] = np.minimum(current_quad[:, [0,2,4,6]],rect[2])
                current_quad[:,[1, 3, 5, 7]] = np.maximum(current_quad[:, [1, 3, 5, 7]], rect[1])
                current_quad[:,[1, 3, 5, 7]] = np.minimum(current_quad[:, [1, 3, 5, 7]], rect[3])
                # adjust to crop (by substracting crop's left,top)
                current_quad[:,[0,2,4,6]] -= rect[0]
                current_quad[:,[1, 3, 5, 7]] -= rect[1]
                # print(current_quad)
                # print('current_image.size:',np.shape(current_image))
                height, width, _ = current_image.shape
                # for i in range(np.shape(current_quad)[0]):
                #     print('i:',i)
                #     cv2.circle(current_image, (int(current_quad[i][0]), int(current_quad[i][1])), 5, (0, 255, 0), 5)
                #     cv2.circle(current_image, (int(current_quad[i][2]), int(current_quad[i][3])), 5, (255, 255, 255), 5)
                #     cv2.circle(current_image, (int(current_quad[i][4]), int(current_quad[i][5])), 5, (255, 0, 0), 5)
                #     cv2.circle(current_image, (int(current_quad[i][6]), int(current_quad[i][7])), 5, (0, 0, 255), 5)
                # cv2.imshow('img',current_image.astype(np.uint8))
                # cv2.imshow('score_map',current_score_map)
                # cv2.waitKey()
                assert len(np.shape(current_quad)) == 2
                return current_image, current_quad,current_score_map
        return image_copy,quad_copy,score_map

def get_rotate_mat(theta):
	return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(quad, theta, anchor=None):
    v = quad.reshape((4,2)).T
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)



class Rotate_small_angle(object):
    def __init__(self,angle_range):
        self.angle_range = angle_range

    def rotate(self,image, angle, center=None, scale=1.0):
        # 获取图像尺寸
        if len(image.shape) == 3:
            h, w,_ = image.shape
        elif len(image.shape) == 2:
            h,w = image.shape

        # 若未指定旋转中心，则将图像中心设为旋转中心
        if center is None:
            center = ((w-1) / 2, (h-1) / 2)

        # 执行旋转
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        # 返回旋转后的图像
        return rotated
    def __call__(self, image, quad,score_map,train=True):
        # if random.randint(2):
        #     return image, quad, score_map
        h, w, _ = np.shape(image)
        center_x = (w - 1) / 2
        center_y = (h - 1) / 2
        angle = self.angle_range * (np.random.rand() * 2 - 1)
        # print('angle:',angle)
        image = self.rotate(image,angle)
        if score_map is None:
            pass
        else:
            score_map = self.rotate(score_map, angle)
        new_quad = np.zeros(quad.shape)
        for i, vertice in enumerate(quad):
            new_quad[i, :] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x], [center_y]]))

        return image, new_quad, score_map

class Expand(object):
    #扩大图像，用均值进行四周区域的填充
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, quad,score_map,train=True):
        if random.randint(2):
            return image,quad,score_map

        height, width, depth = image.shape
        ratio = random.uniform(1,1.4)
        #确定左上角点
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)
        # print('ratio:',ratio)


        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)

        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),int(left):int(left + width)] = image
        if score_map is None:
            pass
        else:
            expand_score_image = np.zeros(
                (int(height * ratio), int(width * ratio)),
                dtype=image.dtype)
            expand_score_image[:, :] = 0
            expand_score_image[int(top):int(top+height),int(left):int(left+width)] = score_map
            score_map = expand_score_image
        image = expand_image
        #print('quad in expand:',quad)
        quad = quad.copy()
        x_axis_index=[0,2,4,6]
        quad[:,x_axis_index]+=int(left)
        y_axis_index=[1,3,5,7]
        quad[:,y_axis_index]+=int(top)
        #print(np.shape(image))

        sw = random.uniform(3 / 4., 4 / 3.)
        sh = random.uniform(3 / 4., 4 / 3.)
        w = int(width * ratio * sw)
        h = int(height * ratio * sh)
        # print(np.shape(image))
        image = cv2.resize(image,(w, h))
        # print(np.shape(image))
        if score_map is None:
            pass
        else:
            score_map = cv2.resize(score_map,(w,h))
        quad[:, x_axis_index] *= sw
        quad[:,y_axis_index] *= sh
        return image,quad, score_map


class RandomMirror(object):
    def __call__(self, image, quad,score_map,train=True):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            if score_map is None:
                pass
            else:
                score_map = score_map[:,::-1]
            quad = quad.copy()
            quad[:, [0,2,4,6]] = width - quad[:, [0,2,4,6]]
            quad = np.concatenate((quad[:, 2:4], quad[:, 0:2], quad[:, 6:], quad[:, 4:6]), axis=1)
        return image, quad , score_map


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", transform='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', transform='RGB'),  # RGB
            RandomContrast()  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, quad,score_map,train=True):
        im = image.copy()
        im, quad,_score_map= self.rand_brightness(im, quad,score_map)

        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, quad ,score_map= distort(im, quad,score_map)
        return im,quad,score_map
        #return self.rand_light_noise(im, quad,score_map)