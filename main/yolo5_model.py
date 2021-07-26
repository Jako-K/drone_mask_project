"""
Modified YOLO5 model for detection of: [faces_with_mask, faces_without_mask, valid_mask, cell phones, person]

Intended use:
1.) model = Yolo5()
2.) info_about_predictions = model.get_predictions(cv2_image).
"""


import cv2
import torch
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from project_utils.yolo5_helpers import detect
from project_utils import helpers as H


class DetectionHandler:
    """Just a convenience class which helps with the processing and accessing of detections """
    def __init__(self, info):
        self.info = info
        self.target_detected = None
        self.target = None
        self.phone_cutout = None
        self.face_cutout = None
        self._preprocess()


    def _preprocess(self, target:str="face"):
        df = None
        if target == "face":
            mask_index = list(self.info[self.info["label_str"] == "with_mask"].index)
            no_mask_index = list(self.info[self.info["label_str"] == "without_mask"].index)
            all_index = list(set(mask_index + no_mask_index))
            df = self.info.iloc[all_index]
        elif target == "person":
            df = self.info[self.info["label_str"] == "person"]

        self.target_detected = len(df) > 0
        if self.target_detected:
            best_prediction = df.iloc[df.conf.argmax()]
            self.target = best_prediction


    def draw_arrow(self, frame):
        assert self.target is not None
        p1 = H.cv2_frame_center(frame)
        p2 = self.target.center
        cv2.arrowedLine(frame, p1, p2, (40, 39, 214), 2)


    def do_phone_cutout(self, frame_raw):
        """Cut out labels with phone from frame if one is present"""
        df_phone = self.info[self.info["label_str"] == "cell_phone"]

        if len(df_phone):
            best_prediction = df_phone.iloc[df_phone.conf.argmax()]
            x1, y1, x2, y2 = [int(n) for n in list(best_prediction["x1":"y2"])]
            self.phone_cutout = frame_raw[y1:y2, x1:x2]

            # The place the cutout is shown has larger height then width, so longest side needs to be the vertical one
            if self.phone_cutout.shape[1] > self.phone_cutout.shape[0]:
                self.phone_cutout = cv2.rotate(self.phone_cutout, cv2.ROTATE_90_COUNTERCLOCKWISE)

            self.phone_cutout  = cv2.resize(self.phone_cutout , (170, 300), interpolation=cv2.INTER_AREA)


    def do_face_cutout(self, frame_raw):
        """Cut out labels with 'face' from frame if one is present"""
        mask_index = list(self.info[self.info["label_str"] == "with_mask"].index)
        no_mask_index = list(self.info[self.info["label_str"] == "without_mask"].index)
        all_index = list(set(mask_index + no_mask_index))
        df_all_faces = self.info.iloc[all_index]

        if len(df_all_faces):
            best_prediction = df_all_faces.iloc[df_all_faces.conf.argmax()]
            x1, y1, x2, y2 = [int(n) for n in list(best_prediction["x1":"y2"])]
            self.face_cutout = frame_raw[y1:y2, x1:x2]
            self.face_cutout = cv2.resize(self.face_cutout, (170, 300), interpolation=cv2.INTER_AREA)


    def draw_all_bbs(self, frame):
        """Draw bounding boxes"""
        for i, pred in self.info.iterrows():
            x1, y1, x2, y2 = [int(n) for n in list(pred["x1":"y2"])]
            H.cv2_draw_bounding_boxes(frame, (x1, y1), (x2, y2), label=pred.label_str, conf=pred.conf, color=pred.color)


class Yolo5:
    """ Simple wrapper for two of Ultralytic's YOLO5 models. One custom and one with pretrained weights """
    def __init__(self, mask_model_path, size="s", device="cuda", min_conf=0.5):
        super().__init__()
        assert size in ["s", "m", "l"], "the size can either be: s,m or l"
        assert device in ["cuda" or "cpu"], "device can either be: cuda or cpu"
        assert 0<min_conf<=1, "minimum confidence most be in percentage i.e (0,1]"

        # Config
        self.min_conf = min_conf
        self.device = device

        # Loading models
        self.yolo5_normal = torch.hub.load(r'ultralytics/yolov5', f'yolov5{size}', pretrained=True).to(device).eval()
        self.yolo_mask = torch.load(mask_model_path)["model"].float().fuse().eval().to(device)

        # Mappings
        self.mask_model_int_to_label = {0:"valid_mask", 1:"without_mask", 2:"with_mask"}
        self.coco_model_int_to_label = {0: "person", 67: "cell_phone"}
        self.mask_model_int_to_color = {0:H.colors_rgb.red, 1:H.colors_rgb.blue, 2:H.colors_rgb.orange}
        self.coco_model_int_to_color = {0:H.colors_rgb.green, 67:H.colors_rgb.purple}


    def _get_label_name(self, label_int, is_mask_model):
        if is_mask_model:
            return self.mask_model_int_to_label[label_int]
        else:
            return self.coco_model_int_to_label[label_int]


    def _get_color_name(self, label_int, is_mask_model):
        if is_mask_model:
            return self.mask_model_int_to_color[label_int]
        else:
            return self.coco_model_int_to_color[label_int]


    def get_predictions(self, img):
        """Try and detect cell phones, humans, faces with a mask, faces without a mask and masks themselves. """

        with torch.no_grad():
            predictions_normal = detect(model=self.yolo5_normal, img_org=img, classes=[67, 0], # class 67 is `person` in coco
                                        device=self.device, confidence_threshold=self.min_conf)
            predictions_mask = detect(model=self.yolo_mask, img_org=img, device=self.device, confidence_threshold=self.min_conf)

            info = []

            for is_mask_model, predictions in enumerate([predictions_normal, predictions_mask]):

                if not len(predictions):
                    continue

                for prediction in predictions:
                    x1, y1, x2, y2, confidence, class_label = prediction.tolist()

                    center = [int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)]
                    temp = {"x1":x1, "y1":y1, "x2":x2, "y2":y2,
                            "width":abs(x2-x1), "height":abs(y2-y1),
                            "center": tuple(center),
                            "conf": confidence,
                            "difference_vector": H.cv2_frame_center(img) - np.array(center),
                            "label":int(class_label),
                            "label_str":self._get_label_name(class_label, is_mask_model),
                            "color":self._get_color_name(class_label, is_mask_model)
                            }
                    info.append(temp)

        return DetectionHandler( pd.DataFrame(info) )if info else None


# Just for testing
if __name__ == "__main__":
    sample_image = cv2.imread("../data/test.jpg")
    yolo_model = Yolo5(mask_model_path="./yolo_mask_model.pt", device="cuda")
    detections = yolo_model.get_predictions(sample_image)
    detections.draw_all_bbs(sample_image)
    print(detections.info)
    H.cv2_show_image(sample_image)
    #cv2.imwrite("final_model_pred.png", sample_image)