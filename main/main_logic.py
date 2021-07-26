"""
Here the whole project comes together.

Intended use:
    1.) drone = TelloDrone()
    2.) drone.initialize(take_off=True)

Testing on images without the drone:
    1.) drone = TelloDrone()
    2.) drone.inference_single_image(image_path)

"""

from yolo5_model import Yolo5, DetectionHandler
from djitellopy import Tello
import numpy as np
import cv2
import sys

sys.path.append('../')
from project_utils import helpers as H

class DroneController:
    """ Basically a middleman between the `TelloDrone` class and the actual Tello drone. """
    def __init__(self, min_wait_between_commands=0.1):
        self.drone = None
        self.take_off = None
        self.min_wait = min_wait_between_commands
        self.timer = H.Timer(time_unit="seconds")
        self.last_command = None


    def start_drone(self, take_off:bool):
        self.drone = Tello()
        self.drone.connect()
        self.drone.streamoff()
        self.drone.streamon()
        if take_off:
            self.drone.takeoff()
        self.timer.start()


    def _handle_received_command(self):
        """ The Tello drone needs a breathing room between commands, this function ensures it gets just that. """
        if self.timer.get_elapsed_time() > self.min_wait:
            self.timer.start()
            return True
        else:
            return False


    def rotate(self, angle:int, wait_for_return:bool=True):
        """
        Handle djitellopy rotation.
        Angle is in degrees.
        Can send commands without violating the minimum wait time required by the drone to operate correctly
        """

        if abs(angle) > 360:
            raise ValueError("Angle must be in [-360, 360]")

        if (wait_for_return is False) and (self._handle_received_command() is False):
            return False
        else:
            self.last_command = f"Rotate {'right' if angle<0 else 'left'} by {abs(angle)} degrees"

        clockwise = angle < 0
        angle = abs(angle)

        if wait_for_return:
            if clockwise:
                self.drone.send_command_with_return(f"cw {angle}")
            else: # Counter clockwise
                self.drone.send_command_with_return(f"ccw {angle}")

        elif not wait_for_return:
            if clockwise:
                self.drone.send_command_without_return(f"cw {angle}")
            else: # Counter clockwise
                self.drone.send_command_without_return(f"ccw {angle}")

        return True


    def move_up_down(self, amount:int, wait_for_return:bool=True):
        """
        Handle djitellopy movement in the up and down direction.
        amount is in centimeters.
        Can send commands without violating the minimum wait time required by the drone to operate correctly
        """

        if not (20 <= abs(amount) <= 500):
            raise ValueError("Movement must be in [-500, -20] or [20, 500]")

        if (wait_for_return is False) and (self._handle_received_command() is False):
            return False
        else:
            self.last_command = f"{'up' if amount > 0 else 'down'} by {abs(amount)} cm"

        move_up = amount > 0
        amount = abs(amount)
        if wait_for_return:
            if move_up:
                self.drone.send_command_with_return(f"up {amount}")
            else: # move down
                self.drone.send_command_with_return(f"down {amount}")


        elif not wait_for_return:
            if move_up:
                self.drone.send_command_without_return(f"up {amount}")
            else: # move down
                self.drone.send_command_without_return(f"down {amount}")

        return True


    def get_status(self, image_inference=False) -> dict:
        """ Return some key statistics from the drone and rename them to something more readable. """
        to_return = ['bat', 'temph', 'yaw', 'h']
        rename_to_return = ["Battery", "Max temp", "Yaw rotation", "Height"]
        if image_inference:
            tello_status_dict = {'bat':"", 'temph':"", 'yaw':"", 'h':""}
        else:
            tello_status_dict = self.drone.get_current_state()
        return {str(rename_to_return[i]): str(tello_status_dict[to_return[i]]) for i in range(4)}


    def turn_off(self):
        self.drone.end()


    def get_frame(self, resize_factor=1.0):
        """ Grab and return the current frame from the Tello drone and make resizing of it possible """
        frame = self.drone.get_frame_read().frame
        return H.cv2_resize_image(frame, resize_factor)


class TelloDrone:
    """ This is the main interface for the entire project and the only class which is intended to be used directly """
    def __init__(self):
        self.yolo_model = Yolo5(mask_model_path="./yolo_mask_model.pt", device="cuda")
        self.fps_timer = H.FPS_Timer()
        self.controller = DroneController()
        self.is_live = False
        self.image_inference_is_live = False
        self.max_height = 175 # cm
        self.min_height = 5 # cm
        self.searching = False


    def initialize(self, take_off:bool):
        self.controller.start_drone(take_off)
        self.is_live = True

        # If the program crash for whatever reason, unhandled cv2 windows are a pain, hence the try and except.
        try:
            self._main_loop()
        except Exception as error:
            print(error)
            cv2.destroyAllWindows()
            self.controller.turn_off() # Just to be sure the drone doesn't go mayhem during some error.


    def _handle_minor_stuff_loop(self):
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press Q to exit and land if airborne
            cv2.destroyAllWindows()
            self.controller.turn_off()
            self.is_live = False

        self.fps_timer.increment()


    def _handle_drone_movement(self, frame, detection_handler:DetectionHandler):
        """ Essentially the drones brain. Processes the information provided by the detection_handler and act
        appropriately by use of rotation and up/down movement"""


        # If the drone hasn't detected any target(s) it starts searching for one by simple rotating around
        if (detection_handler is not None) and detection_handler.target_detected:
            self.searching = False
            x_component, y_component = detection_handler.target.difference_vector
            detection_handler.draw_arrow(frame)
        else:
            self.searching = True
            self.controller.rotate(-5, wait_for_return=False)
            return


        """ Rotation """
        # Seems like a waste of time and energy to chase pixel perfect precision. Within 15% of the frame width seems ok
        frame_width = frame.shape[1]
        if abs(x_component) > 0.15*frame_width:
            # Rotation calculation.
            #   -   width/100 * scaling = 1 --> scaling = 100/width
            #   -   scaling*x_component than gives the theoretical rotation necessary to point the drone towards the human.
            #   -   This did however become hyper-responsive and looked and felt way too aggressive, hence the rotate_damper.
            rotation_scaling_factor = (100/frame_width) * 0.35  # damped by 0.35, this ought to be normalized to be resolution agnostic.
            rotation = int(x_component * rotation_scaling_factor)
            self.controller.rotate(rotation, wait_for_return=False)


        """ Movement up/down """
        # Acceptable height difference is set to 2.5% of the frame height. The height check is an extra safety measure.
        # The minimum up/down movement command the Tello drone accepts is 20 cm and the max 500 cm. 500 cm is way to
        # much for this use case and the limit is set to 50 cm instead. A damping factor is added like with the rotation
        frame_height = frame.shape[0]
        if self.image_inference_is_live:
            height_check = True
        else:
            height_check = self.min_height < self.controller.drone.get_height() < self.max_height
        if (abs(y_component) > int(0.025*frame_height)) and height_check:
            movement_up_down = int(y_component * 0.40) # damped by 0.40, this ought to be normalized to be resolution agnostic.
            low_limit_ok, high_limit_ok = 20 <= abs(movement_up_down), abs(movement_up_down) <= 50

            if low_limit_ok and high_limit_ok:
                self.controller.move_up_down(movement_up_down, wait_for_return=False)
            elif not low_limit_ok:
                self.controller.move_up_down(H.int_sign(movement_up_down) * 20, wait_for_return=False)
            elif not high_limit_ok:
                self.controller.move_up_down(H.int_sign(movement_up_down) * 50, wait_for_return=False)
            else:
                ValueError("Shouldn't have gotten this far")


    def _handle_display(self, frame, detection_handler):
        """ The entire  """
        h, w, c = frame.shape
        canvas = np.zeros( (h,w+400,c) , dtype=frame.dtype) + 128 # Make a grey image a bit bigger than the frame
        canvas[:, :w, :] = frame # Add the frame to the gray image
        canvas[:, w:w+5, :] = 60 # Grey separating line (vertical)

        # FPS
        font = cv2.FONT_HERSHEY_DUPLEX
        font_size = 0.5
        font_thickness = 1
        font_color = (200,200,200)
        if not self.image_inference_is_live:
            cv2.putText(canvas, f"FPS: {self.fps_timer.get_fps()}", (10, 20), font,
                        font_size, (50,50,50), font_thickness, cv2.LINE_AA)


        """ Drone control """
        cv2.putText(canvas, "Drone Control", (w+10, 30), font, 1, font_color, 2, cv2.LINE_AA)
        canvas[35:37, w + 10:w + 400 - 10, :] = 150  # whitish separating line (horizontal)
        drone_print = [f"Searching: {self.searching}",
                        f"Command: {self.controller.last_command}"]
        for i, status in enumerate(drone_print):
            cv2.putText(canvas, "- " + status, (w + 10, 45+20*(i+1)), font, font_size, font_color, 1, cv2.LINE_AA)


        """ Face detection """
        cv2.putText(canvas, "Face", (w+10, 150), font, 1, font_color, 2, cv2.LINE_AA)
        canvas[155:157, w + 10:w + 180, :] = 150  # whitish separating line (horizontal)
        canvas[165:465, w + 10: w + 180, :] = 118
        if detection_handler is not None and detection_handler.face_cutout is not None:
            canvas[165:465, w+10 : w+180, :] = detection_handler.face_cutout


        """ Cell phone detection """
        cv2.putText(canvas, "Phone", (w+200, 150), font, 1, font_color, 2, cv2.LINE_AA)
        canvas[155:157, w + 200:w + 200+170, :] = 150  # whitish separating line (horizontal)
        canvas[165:465, w + 200: w + 200 + 170, :] = 118
        if detection_handler is not None and detection_handler.phone_cutout is not None:
            canvas[165:465, w + 200: w + 200+170, :] = detection_handler.phone_cutout


        """ Drone status """
        cv2.putText(canvas, "Drone Status", (w+10, 510), font, 1, font_color, 2, cv2.LINE_AA)
        canvas[515:517, w + 10:w + 400 - 10, :] = 150  # whitish separating line (horizontal)


        status = self.controller.get_status(self.image_inference_is_live)
        for i, key in enumerate(status):
            cv2.putText(canvas, "- " + key+": "+status[key], (w + 10, 520+20*(i+1)), font, font_size, font_color, 1, cv2.LINE_AA)


        # finally the drawn upon frame is displayed together with the side bar with cutouts and other information
        cv2.imshow("YOLO", canvas)


    def _main_loop(self):
        self.fps_timer.start()

        while self.is_live:
            frame = self.controller.get_frame(resize_factor=1.0)

            """ YOLO """
            if (detection_handler := self.yolo_model.get_predictions(frame)) is not None:
                detection_handler.do_phone_cutout(frame)
                detection_handler.do_face_cutout(frame)
                detection_handler.draw_all_bbs(frame)

            self._handle_minor_stuff_loop()
            self._handle_drone_movement(frame, detection_handler)
            self._handle_display(frame, detection_handler)


    def inference_single_image(self, image_path):
        self.image_inference_is_live = True
        self.fps_timer.start()
        frame = cv2.imread(image_path)
        frame_raw = frame.copy()

        """ YOLO """
        if (detection_handler := self.yolo_model.get_predictions(frame)) is not None:
            detection_handler.do_phone_cutout(frame)
            detection_handler.do_face_cutout(frame)
            detection_handler.draw_all_bbs(frame)

        self._handle_minor_stuff_loop()
        self._handle_drone_movement(frame, detection_handler)
        self._handle_display(frame, detection_handler)

        H.cv2_show_image(frame_raw)


if __name__ == "__main__":
    drone = TelloDrone()
    drone.initialize(take_off=True)

    #drone.inference_single_image("../data/test1.jpg")
