import numpy as np
from six.moves import range
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

NUM_COLORS = len(STANDARD_COLORS)

try:
    FONT = ImageFont.truetype('arial.ttf', 24)
except IOError:
    FONT = ImageFont.load_default()


def _draw_single_box(image, quad, color='green', display_str=None, font=None, thickness=4):
    draw = ImageDraw.Draw(image)
    #draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)
    draw.line([(quad[0], quad[1]), (quad[6], quad[7]), (quad[4], quad[5]), (quad[2], quad[3]), (quad[0], quad[1])], width=thickness, fill=color)

    return image


def draw_bounding_boxes(image, boxes):
    """Draw bboxes(labels, probs) on image
    Args:
        image: numpy array image, shape should be (height, width, channel)
        boxes: bboxes, shape should be (N, 4), and each row is (xmin, ymin, xmax, ymax)
        labels: labels, shape: (N, )
        probs: label scores, shape: (N, ), can be False/True or None
        class_name_map: list or dict, map class id to class name for visualization.
             can be False/True or None
    Returns:
        An image with information drawn on it.
    """
    num_boxes = boxes.shape[0]
    gt_boxes_new = boxes.copy()
    draw_image = Image.fromarray(np.uint8(image))
    for i in range(num_boxes):
        draw_image = _draw_single_box(image=draw_image,
                                      quad=gt_boxes_new[i,:],
                                      font=FONT)

    image = np.array(draw_image, dtype=np.float32)
    return image
