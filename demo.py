import glob
import os

import torch
from PIL import Image
from tqdm import tqdm
from ssd.config import cfg
from ssd.modeling.predictor import Predictor
from ssd.modeling.vgg_ssd import build_ssd_model
import argparse
import numpy as np
import cv2
from ssd.utils.viz import draw_bounding_boxes
from shapely.geometry import Polygon,MultiPoint  #多边形


from scipy.spatial import distance as dist
import numpy as np
import math
def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a, b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down

def order_points_quadrangle(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left and bottom-left coordinate, use it as an
    # base vector to calculate the angles between the other two vectors

    vector_0 = np.array(bl - tl)
    vector_1 = np.array(rightMost[0] - tl)
    vector_2 = np.array(rightMost[1] - tl)

    angle = [np.arccos(cos_dist(vector_0, vector_1)), np.arccos(cos_dist(vector_0, vector_2))]
    (br, tr) = rightMost[np.argsort(angle), :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")
def validate_clockwise_points(points):
    """
    Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
    """

    if len(points) != 8:
        raise Exception("Points list not valid." + str(len(points)))

    point = [
                [int(points[0]) , int(points[1])],
                [int(points[2]) , int(points[3])],
                [int(points[4]) , int(points[5])],
                [int(points[6]) , int(points[7])]
            ]
    edge = [
                ( point[1][0] - point[0][0])*( point[1][1] + point[0][1]),
                ( point[2][0] - point[1][0])*( point[2][1] + point[1][1]),
                ( point[3][0] - point[2][0])*( point[3][1] + point[2][1]),
                ( point[0][0] - point[3][0])*( point[0][1] + point[3][1])
    ]

    summatory = edge[0] + edge[1] + edge[2] + edge[3];
    if summatory>0:
        return False
    else:
        return True
def run_demo(cfg, checkpoint_file, iou_threshold, score_threshold, images_dir, output_dir):
    device = torch.device(cfg.MODEL.DEVICE)
    model = build_ssd_model(cfg)
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded weights from {}.'.format(checkpoint_file))
    model = model.to(device)
    model.eval()
    predictor = Predictor(cfg=cfg,
                          model=model,
                          iou_threshold=iou_threshold,
                          score_threshold=score_threshold,
                          device=device)
    cpu_device = torch.device("cpu")

    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    add_count = 0
    for image_path in tqdm(image_paths):
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        # image_mirror = image[:, ::-1]
        output = predictor.predict(image)
        boxes, scores,seg_map = [o.to(cpu_device).numpy() for o in output]
        seg_map = cv2.resize(seg_map, (512, 512)) * 255
        seg_map = seg_map.astype(np.uint8)
        # seg_map = cv2.applyColorMap(seg_map, cv2.COLORMAP_JET)
        seg_map = cv2.resize(seg_map,(1280,720),interpolation=cv2.INTER_CUBIC)
        drawn_image = draw_bounding_boxes(image, boxes).astype(np.uint8)
        image_name = os.path.basename(image_path)
        txt_path = os.path.join(output_dir,'txtes')
        if not os.path.exists(txt_path):
            os.makedirs(txt_path)
        txt_path = os.path.join(txt_path,'res_'+image_name.replace('jpg','txt'))
        #multi-output merge
        merge_output = False
        if merge_output:
            ret, binary = cv2.threshold(seg_map, 75, 255, cv2.THRESH_BINARY)
            # cv2.imshow('binary:',binary)
            # cv2.waitKey()

            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            w, h = np.shape(binary)
            for contour in contours:
                # 获取最小包围矩形
                rect = cv2.minAreaRect(contour)

                # 中心坐标
                x, y = rect[0]
                # cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), 5)

                # 长宽,总有 width>=height
                width, height = rect[1]
                if width < 10 or height < 10:
                    continue

                # 角度:[-90,0)
                angle = rect[2]
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                box[:, 0] = np.clip(box[:, 0], 0, h)
                box[:, 1] = np.clip(box[:, 1], 0, w)

                poly1 = Polygon(box).convex_hull
                intersect = False
                for item in boxes:
                    print('item:',item)
                    poly2 = Polygon(item.reshape(4,2)).convex_hull
                    if poly1.intersects(poly2):  # 如果两四边形相交
                        intersect = True
                        break
                if not intersect:
                    print('boxes.shape:',np.shape(boxes))
                    box = box.reshape((1,8))
                    print('box.shape:',np.shape(box))
                    num,_ = np.shape(boxes)
                    if num == 0:
                        print('num == 0')
                        boxes = box
                    else:
                        boxes = np.concatenate((boxes,box))
                    print('boxes.shape:',np.shape(boxes))
                    print('add one box')
                    add_count += 1
                    # cv2.line(image, (box[0][0], box[0][1]), (box[0][2], box[0][3]), (0, 0, 255), thickness=4)
                    # cv2.line(image,(box[0][2], box[0][3]), (box[0][4], box[0][5]), (0, 0, 255), thickness=4)
                    # cv2.line(image,(box[0][4], box[0][5]), (box[0][6], box[0][7]), (0, 0, 255), thickness=4)
                    # cv2.line(image, (box[0][6], box[0][7]), (box[0][0], box[0][1]), (0, 0, 255), thickness=4)
                    # cv2.imshow('img',image)
                    # cv2.waitKey()




        # print('txt_path:',txt_path)
        with open(txt_path,'w+') as f:
            for box in boxes:
                box_temp=np.reshape(box,(4,2))
                box=order_points_quadrangle(box_temp)
                box=np.reshape(box,(1,8)).squeeze(0)
                is_valid = validate_clockwise_points(box)
                if not is_valid:
                    continue
                # print('box:',box)
                line=''
                for item in box:
                    if item < 0:
                        item = 0
                    line += str(int(item))+','
                line = line[:-1] + '\n'
                f.write(line)
        path = os.path.join(output_dir, image_name)
        print('path:',path)
        Image.fromarray(drawn_image).save(path)
        path = os.path.join(output_dir, image_name.split('.')[0]+'_segmap.'+image_name.split('.')[1])
        # print(path)
        # 存储score_map
        cv2.imwrite(path,seg_map)
    print('add count:',add_count)


def main():
    parser = argparse.ArgumentParser(description="ssd_fcn_multitask_text_detectior training with pytorch.")
    parser.add_argument(
        "--config-file",
        default="configs/icdar2015_incidental_scene_text.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    #ssd512_vgg_iteration_021125可以到59
    parser.add_argument("--checkpoint_file", default='/home/binchengxiong/ssd_fcn_multitask_text_detection_pytorch1.0/output/ssd512_vgg_iteration_140000.pth',type=str, help="Trained weights.")
    parser.add_argument("--iou_threshold", type=float, default=0.1)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--images_dir", default='/home/binchengxiong/ssd_fcn_multitask_text_detection_pytorch1.0/demo/', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='/home/binchengxiong/ssd_fcn_multitask_text_detection_pytorch1.0/demo/result2/', type=str, help='Specify a image dir to save predicted images.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             checkpoint_file=args.checkpoint_file,
             iou_threshold=args.iou_threshold,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir
             )


if __name__ == '__main__':
    main()
