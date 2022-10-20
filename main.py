import tensorflow as tf
import numpy as np
from playsound import playsound
import threading
from functools import partial
from kivymd.app import MDApp
from kivy.clock import Clock
import time
from kivy.core.image import Texture
from kivy.config import Config
import cv2
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivymd.toast.kivytoast.kivytoast import toast

Config.set('graphics', 'resizable', False)
Window.size = (360,750)
interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_thunder_3.tflite')
interpreter.allocate_tensors()


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)


def calculate_angle(a, b, c, frame):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    y, x, z = frame.shape
    first_point = np.squeeze(np.multiply(a, [y, x, 1]))
    center_point = np.squeeze(np.multiply(b, [y, x, 1]))
    last_point = np.squeeze(np.multiply(c, [y, x, 1]))

    # calculate angle

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    ###################################

    cv2.putText(frame, text=str(int(angle)) + "deg", org=(int(center_point[1]) - 60, int(center_point[0]) + 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 0),
                thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (int(center_point[1]), int(center_point[0])), 4, (255, 0, 0), -2)
    cv2.line(frame, (int(first_point[1]), int(first_point[0])), (int(center_point[1]), int(center_point[0])),
             (0, 0, 255), 2)
    cv2.line(frame, (int(center_point[1]), int(center_point[0])), (int(last_point[1]), int(last_point[0])),
             (0, 0, 255), 2)

    return angle


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape

    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

class LandingPage(Screen):
    pass

class AboutPage(Screen):
    pass

class CapturePage(Screen):
    pass

class ResultPage(Screen):
    pass

class LoadingPage(Screen):
    pass


class MainApp(MDApp):
    name = "main"
    knee_angle_index = 0
    knee_angle = 0
    knee_angles = []

    torso_angles = []
    max_torso_index = 0
    min_torso_index = 0
    torso_angle = 0

    def build(self):
        self.res_images = []
        self.frames = []
        self.screen_manager = ScreenManager()
        self.screen_manager.add_widget(LandingPage(name="LandingPage"))
        self.screen_manager.add_widget(AboutPage(name="AboutPage"))
        self.screen_manager.add_widget(CapturePage(name="CapturePage"))
        self.screen_manager.add_widget(LoadingPage(name="LoadingPage"))
        self.screen_manager.add_widget(ResultPage(name="ResultPage"))

        kv = Builder.load_file("main.kv")
        self.theme_cls.theme_style = "Dark"
        threading.Thread(target=self.doit, daemon=True).start()
        self.new_screen = CapturePage()
        return self.screen_manager

    def doit(self):
        self.do_vid = True
        self.frame = None
        cam = cv2.VideoCapture(0)
        while (self.do_vid):
            ret, self.frame = cam.read()
            self.frame = cv2.flip(self.frame, 1)
            Clock.schedule_once(partial(self.display_frame, self.frame))
            cv2.waitKey(1)
        cam.release()
        cv2.destroyAllWindows()

    def display_frame(self, frame, dt):
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(frame.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')
        texture.flip_vertical()
        self.root.get_screen('CapturePage').ids.vid.texture = texture

    def start_take_pic_thread(self):
        threading.Thread(target=self.delay_take_pic).start()

    def delay_take_pic(self):
        self.root.get_screen('CapturePage').ids.start_btn.disabled = True
        playsound('countdown.wav')
        playsound('start.wav')
        threading.Thread(target=self.take_pic).start()

    def take_pic(self, *args):
        self.i = 0
        self.timer = 0
        self.limit = 300
        start_time = time.time()
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            frame = self.get_running_app().frame
            frame = frame[0:480,80:560]

            self.frames.append(frame)
            Clock.usleep(33333)
            cv2.waitKey(1)
            self.i += 1
            self.timer += 1

            if(self.timer == self.limit):
                break

        self.update_label()
        self.root.get_screen('CapturePage').ids.process_btn.disabled = False
        print("Finished iterating in: " + str(int(elapsed_time)) + " seconds")
        print("Number of frames:" + str(len(self.frames)))

    def start_show_result_thread(self):
        threading.Thread(target=self.process_result).start()
        self.screen_manager.current = "LoadingPage"
        Clock.schedule_once(self.go_to_result_page, 65)

    def process_result(self):
        start_time = time.time()
        counter = 0
        for frame in self.frames:
            # if counter % 3 == 0:
            self.res_images.append(self.load_result_frames(frame, 0))
            counter += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
        print("Result Frames Finished iterating in: " + str(int(elapsed_time)) + " seconds")
        print("Number of result frames:" + str(len(self.res_images)))

    def go_to_result_page(self, dt):

        # getting the index of the max angle of lower hip
        max_knee_angle = max(self.knee_angles)
        self.knee_angle_index = self.knee_angles.index(max_knee_angle)

        # getting the index of the max angle of torso
        max_torso_angle = max(self.torso_angles)
        self.max_torso_index = self.torso_angles.index(max_torso_angle)

        # getting the index of the min angle of torso
        min_torso_angle = min(self.torso_angles)
        self.min_torso_index = self.torso_angles.index(min_torso_angle)
        print(max_torso_angle)
        print(self.min_torso_index)
        print(min_torso_angle)
        print(self.max_torso_index)
        print(max_knee_angle)
        print(self.knee_angle_index)
        print(type(self.res_images[self.max_torso_index]))
        torso_max_frame = self.res_images[self.max_torso_index]
        torso_min_frame = self.res_images[self.min_torso_index]
        knee_frame = self.res_images[self.knee_angle_index]

        # displaying max torso angle frame
        texture = Texture.create(size=(torso_max_frame.shape[1], torso_max_frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(torso_max_frame.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')
        texture.flip_vertical()
        self.root.get_screen('ResultPage').ids.upper_low.texture = texture
        self.root.get_screen('ResultPage').ids.upper_max_angle.text = str(
            "{:.2f}".format(self.torso_angles[self.max_torso_index])) + " deg"
        if self.torso_angles[self.max_torso_index] > 47.95 or self.torso_angles[self.max_torso_index] < 20:
            self.root.get_screen('ResultPage').ids.upper_max_angle.text_color = [0.8, 0, 0, 0.6]
            self.root.get_screen('ResultPage').ids.upper_max_icon.text_color = [0.8, 0, 0, 0.6]
        else:
            self.root.get_screen('ResultPage').ids.upper_max_angle.text_color = [0, 1, 1, 0.6]
            self.root.get_screen('ResultPage').ids.upper_max_icon.text_color = [0, 1, 1, 0.6]

        # displaying min torso angle frame
        texture = Texture.create(size=(torso_min_frame.shape[1], torso_min_frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(torso_min_frame.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')
        texture.flip_vertical()
        self.root.get_screen('ResultPage').ids.upper_high.texture = texture
        self.root.get_screen('ResultPage').ids.upper_min_angle.text = str(
            "{:.2f}".format(self.torso_angles[self.min_torso_index])) + " deg"
        if self.torso_angles[self.min_torso_index] > 47.95 or self.torso_angles[self.min_torso_index] < 20:
            self.root.get_screen('ResultPage').ids.upper_min_angle.text_color = [0.8, 0, 0, 0.6]
            self.root.get_screen('ResultPage').ids.upper_min_icon.text_color = [0.8, 0, 0, 0.6]
        else:
            self.root.get_screen('ResultPage').ids.upper_min_angle.text_color = [0, 1, 1, 0.6]
            self.root.get_screen('ResultPage').ids.upper_min_icon.text_color = [0, 1, 1, 0.6]


        #displaying knee angle frame
        texture = Texture.create(size=(knee_frame.shape[1], knee_frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(knee_frame.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')
        texture.flip_vertical()
        self.root.get_screen('ResultPage').ids.lower.texture = texture
        self.root.get_screen('ResultPage').ids.knee_max_angle.text = str(
            "{:.2f}".format(self.knee_angles[self.knee_angle_index])) + " deg"
        if self.knee_angles[self.knee_angle_index] > 155 or self.knee_angles[self.knee_angle_index] < 145:
            self.root.get_screen('ResultPage').ids.knee_max_angle.text_color = [0.8, 0, 0, 0.6]
            self.root.get_screen('ResultPage').ids.knee_max_icon.text_color = [0.8, 0, 0, 0.6]
        else:
            self.root.get_screen('ResultPage').ids.knee_max_angle.text_color = [0, 1, 1, 0.6]
            self.root.get_screen('ResultPage').ids.knee_max_icon.text_color = [0, 1, 1, 0.6]


        self.screen_manager.current = "ResultPage"

    def load_result_frames(self, frames, *args):
        img = frames.copy()
        frame = frames.copy()

        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
        input_image = tf.cast(img, dtype=tf.float32)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        # HIP AND HORIZONTAL KEYPOINT
        right_hip = keypoints_with_scores[0][0][12]
        horizontal_hip = [right_hip[0], right_hip[1] + 0.12, 0.4]

        # UPPER BODY KEYPOINTS
        right_wrist = keypoints_with_scores[0][0][10]
        right_elbow = keypoints_with_scores[0][0][8]
        right_shoulder = keypoints_with_scores[0][0][6]

        # LOWER BODY KEYPOINTS
        right_knee = keypoints_with_scores[0][0][14]
        right_ankle = keypoints_with_scores[0][0][16]

        # UPPER BODY FEATURES
        #elbow_angle = calculate_angle(right_wrist, right_elbow, right_shoulder, frame)
        #shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip, frame)
        self.hip_angle_upper = calculate_angle(horizontal_hip, right_hip, right_shoulder, frame)

        # LOWER BODY FEATURES
        self.knee_angle = calculate_angle(right_hip, right_knee, right_ankle, frame)
        # hip_angle_lower = calculate_angle(horizontal_hip, right_hip, right_knee, frame)
        draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
        draw_keypoints(frame, keypoints_with_scores, 0.4)

        # storing knee angles in a list
        self.knee_angles.append(self.knee_angle)

        # storing the max and min angle of torso
        self.torso_angles.append(self.hip_angle_upper)

        return frame

    def restart(self):
        self.res_images = []
        self.frames = []
        self.knee_angles = []
        self.torso_angles = []

        self.root.get_screen('CapturePage').ids.start_btn.disabled = False
        self.root.get_screen('CapturePage').ids.process_btn.disabled = True
        self.root.get_screen('CapturePage').ids.process.text = "\u2022For a better angle of the body - Align your bicycle's wheels with the two circles on the screen\n\u2022 Press Start button to start the timer.\n\u2022 A series of beep will play - make sure to be ready on your bicycle before the beeping stops\n\u2022 You must start pedaling after hearing the voice - 'start pedaling'\n\u2022 You can stop pedaling after hearing the voice - 'stop pedaling'"

    def update_label(self):
        self.root.get_screen('CapturePage').ids.process.text = "\u2022 Press Process button to start processing the captured images.\n\u2022 Note: The processing time will take 30 - 40 seconds.\n\u2022 A new page will show after processing with the results from the session.\n"

    def upper_max_icon_toast(self):
        if self.torso_angles[self.max_torso_index] > 47.95 :
            toast("Try to lean a little forward.")
        elif self.torso_angles[self.max_torso_index] < 20:
            toast("Try not to lean too much forward.")
        else:
            if self.torso_angles[self.max_torso_index] > 20 and self.torso_angles[self.max_torso_index] < 40:
                toast("The max angle is in a correct measurement and in an aerodynamic position!")
            else:
                toast("The max angle is in a correct measurement and in a normal position!")
    def upper_min_icon_toast(self):
        if self.torso_angles[self.min_torso_index] > 47.95 :
            toast("Try to lean a little forward.")
        elif self.torso_angles[self.min_torso_index] < 20:
            toast("Try not to lean too much forward.")
        else:
            if self.torso_angles[self.min_torso_index] > 20 and self.torso_angles[self.min_torso_index] < 40:
                toast("The min angle is in a correct measurement and in an aerodynamic position!")
            else:
                toast("The min angle is in a correct measurement and in a normal position!")

    def knee_max_icon_toast(self):
        if self.knee_angles[self.knee_angle_index] > 155:
            toast("The knee angle measurement is wrong. Try to adjust the saddle height lower.")
        elif self.knee_angles[self.knee_angle_index] < 145:
            toast("The knee angle measurement is wrong. Try to adjust the saddle height higher.")
        else:
            toast("The knee angle measurement is correct!")

# run the app
if __name__=='__main__':
    MainApp().run()
