"""
A bunch of small helper functions. All of the function are taken from my own github:
https://github.com/Jako-K/utils/blob/main/__code/_helpers.py
"""
import math
import random
import time
import os
from datetime import timedelta
import matplotlib.pyplot as plt
from PIL import Image, ImageColor
import numpy as np
import cv2
import pathlib
import json
import requests

def expand_jupyter_screen(percentage:int = 75):
    assert percentage in [i for i in range(50,101)], "Bad argument" # Below 50 just seems odd, assumed to be a mistake
    from IPython.core.display import display, HTML
    argument = "<style>.container { width:" + str(percentage) + "% !important; }</style>"
    display(HTML(argument))


def read_json(path):
    assert os.path.exists(path), "Bad path"
    assert (path[-5:] == ".json"), "Bad extension, expected .json"

    f = open(path)
    data = json.load(f)
    f.close()

    return data


class _ColorRGB:
    blue = (31, 119, 180)
    orange = (255, 127, 14)
    green = (44, 160, 44)
    red = (214, 39, 40)
    purple = (148, 103, 189)
    brown = (140, 86, 75)
    pink = (227, 119, 194)
    grey = (127, 127, 127)
    white = (225, 255, 255)
    all_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'white']

    def random_color(self, only_predefined_colors=True):
        if only_predefined_colors:
            return getattr(self, random.choice(self.all_colors))
        else:
            return [random.randint(0, 255) for i in range(3)]

    def random_not_to_bright_color(self):
        return [random.randint(0, 150) for i in range(3)]


    def is_legal(self, color):
        for color_channel in color:
            if not(0 <= color_channel <= 255):
                return False
        return True
colors_rgb = _ColorRGB()


def hex_color_to_rgb(hex_color):
    return ImageColor.getcolor(hex_color, "RGB")


def cv2_draw_bounding_boxes(image, p1, p2, label=None, conf=None, color="random", line_thickness=2,
                            text_color=(200, 200, 200)):


    if color == "random":
        color = colors_rgb.random_not_to_bright_color()
    elif color[0] == "#":
        color = hex_color_to_rgb(color)

    cv2.rectangle(image, p1, p2, color, line_thickness)

    text = ""
    if label:
        text += label
    if conf:
        if label:
            text += ": "
        text += str(round(conf * 100, 3)) + "%"

    if label or conf:
        new_p2 = (p1[0] + 10 * len(text), p1[1] - 15)
        cv2.rectangle(image, p1, new_p2, color=color, thickness=-1)
        cv2.putText(image, text, (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1, cv2.LINE_AA)

def cv2_show_image(image, resize_factor=1.0, name=""):
    assert resize_factor > 0, "resize_factor must have a value greater than 0"

    if in_jupyter():
        img = cv2_image_to_pillow(image)
        img = pillow_resize_image(img, resize_factor)
        display(img)
    else:
        img = cv2_resize_image(image, resize_factor)
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def in_jupyter():
    # Not the cleanest, but gets the job done
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def cv2_resize_image(image, scale_factor):
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    return cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)


def pillow_resize_image(image, scale_factor):
    width = int(image.size[0] * scale_factor)
    height = int(image.size[1] * scale_factor)
    return image.resize((width, height), resample=0, box=None)


def cv2_image_to_pillow(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def save_plt_plot(save_path, fig=None, dpi=300):
    assert extract_file_extension(save_path) in [".png", ".jpg", ".pdf"]
    if fig is None:
        plt.savefig(save_path, dpi = dpi, bbox_inches = 'tight')
    else:
        fig.savefig(save_path, dpi = dpi, bbox_inches = 'tight')


def extract_file_extension(file_name:str):
	"""
	>>> extract_file_extensions("some_path/works_with_backslashes\\and_2x_extensions.tar.gz")
	'.tar.gz'
	"""
	assert file_name.find(".") != -1, "No ´.´ found"
	suffixes = pathlib.Path(file_name).suffixes
	return ''.join(pathlib.Path(file_name).suffixes)


def write_to_file(file_path: str, write_string: str, only_txt: bool = True):
    """ Appends a string to the end of a file"""
    if only_txt:
        assert extract_file_extension(file_path) == ".txt", "´only_txt´ = true, but file type is not .txt"

    file = open(file_path, mode="a")
    print(write_string, file=file, end="")
    file.close()


def get_image_size(path, WxH=True):
    assert os.path.exists(path), "Bad path"
    height, width = cv2.imread(path).shape[:2]
    return (width, height) if WxH else (height, width)


def normal_bb_coordinates_to_yolo_format(bb, img_width, img_height, label, xywh=False):
    if not xywh:
        x1, y1, x2, y2 = bb
        bb_width, bb_height = (x2 - x1), (y2 - y1)
    else:
        x1, y1, bb_width, bb_height = bb

    # Width and height
    bb_width_norm = bb_width / img_width
    bb_height_norm = bb_height / img_height

    # Center
    bb_center_x_norm = (x1 + bb_width / 2) / img_width
    bb_center_y_norm = (y1 + bb_height / 2) / img_height

    # Yolo format --> |class_name center_x center_y width height|.txt  -  NOT included the two '|'
    string = str(label)
    for s in [bb_center_x_norm, bb_center_y_norm, bb_width_norm, bb_height_norm]:
        string += " " + str(s)

    return string


def get_image_from_url(url:str, return_type="cv2"):
    assert return_type in ["pillow", "cv2"], "`return_type` not in ['pillow', 'cv2']"
    if return_type == "cv2":
        return np.asarray(Image.open(requests.get(url, stream=True).raw))
    elif return_type == "pillow":
        return Image.open(requests.get(url, stream=True).raw)


def extract_file_name(path):
    return path.split("\\")[-1].split(".")[0]


def cv2_frame_center(frame, WxH=True):
    h, w = frame.shape[:2]
    return  (w//2, h//2) if WxH else (h//2, w//2)


class Timer:
    """
    EXAMPLE:

    timer = Timer()

    timer.start()
    time.sleep(2)
    timer.stop()

    print(timer.get_elapsed_time())

    """

    def __init__(self, time_unit="seconds", start_on_create=False, precision_decimals=3):
        self._start_time = None
        self._elapsed_time = None
        self._is_running = False
        self._unit = None; self.set_unit(time_unit)
        self.precision_decimals = precision_decimals
        if start_on_create:
            self.start()

    def start(self):
        if self._start_time is not None:
            self.reset()

        self._start_time = time.time()
        self._is_running = True

    def _calculate_elapsed_time(self):
        if self._start_time is None:
            return None
        else:
            return round(time.time() - self._start_time, self.precision_decimals)

    def stop(self):
        assert self._start_time is not None, "Call `start()` before `stop()`"
        self._elapsed_time = self._calculate_elapsed_time()
        self._is_running = False

    def get_elapsed_time(self):
        current_time = self._calculate_elapsed_time() if self._is_running else self._elapsed_time

        if current_time is None:
            return 0
        elif self._unit == "seconds":
            return current_time
        elif self._unit == "minutes":
            return current_time / 60.0
        elif self._unit == "hours":
            return current_time / 3600.0
        elif self._unit == "hour/min/sec":
            return str(timedelta(seconds=current_time)).split(".")[0] # the ugly bit is just to remove ms
        else:
            raise RuntimeError("Should not have gotten this far")

    def set_unit(self, time_unit:str = "hour/min/sec"):
        assert time_unit in ("hour/min/sec", "seconds", "minutes", "hours")
        self._unit = time_unit

    def reset(self):
        self._start_time = None
        self._elapsed_time = None
        self._is_running = False


class FPS_Timer:
    """
    EXAMPLE:

    fps_timer = FPS_Timer()
    fps_timer.start()

    for _ in range(10):
        time.sleep(0.2)
        fps_timer.increment()
        print(fps_timer.get_fps())

    """

    def __init__(self, precision_decimals=3):
        self._start_time = None
        self._elapsed_time = None
        self.fpss = []
        self.precision_decimals = precision_decimals


    def start(self):
        assert self._start_time is None, "Call `reset()` before you call `start()` again"
        self._start_time = time.time()


    def _get_elapsed_time(self):
        return round(time.time() - self._start_time, self.precision_decimals)


    def increment(self):
        self.fpss.append(self._get_elapsed_time())

    def get_frame_count(self):
        return len(self.fpss)

    def get_fps(self, rounded=3):
        assert self._start_time is not None, "Call `start()` before you call `get_fps()`"
        if len(self.fpss) < 2:
            fps = 0
        else:
            fps = 1 / (self.fpss[-1] - self.fpss[-2])
        return round(fps, 3)


    def reset(self):
        self._elapsed_time = None
        self.fpss = []


def int_sign(x:int):
    return math.copysign(1, x)