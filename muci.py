#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf2_ros
import subprocess
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs  # <- potreben za sam modul
import tf2_geometry_msgs.tf2_geometry_msgs  # <- NUJNO za aktivacijo podpore PointStamped
import sys
import csv
import os
import datetime
import builtins

# --- DODANO: Preusmeritev izpisov v CSV ---
class TeeToCSV:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.filename = filename
        self.csv_writer = None
        self._initialize_csv()

    def _initialize_csv(self):
        file_exists = os.path.isfile(self.filename)
        self.log = open(self.filename, "a", newline='')
        self.csv_writer = csv.writer(self.log)
        if not file_exists or os.stat(self.filename).st_size == 0:
            self.csv_writer.writerow(["timestamp", "type", "message"])

    def write(self, message):
        if message.strip() != "":
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.csv_writer.writerow([now, "PRINT", message.strip()])
            self.log.flush()
        self.terminal.write(message)
        self.terminal.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        if hasattr(self, 'log'):
            self.log.close()

csv_path = os.path.join(os.path.dirname(__file__), "detekcije.csv")
sys.stdout = TeeToCSV(csv_path)
sys.stderr = TeeToCSV(csv_path)

# --- Prepiši rospy log funkcije ---
old_print = print

def print(*args, **kwargs):
    old_print(*args, **kwargs)
    # print gre že v sys.stdout, ki je preusmerjen

old_info = rospy.loginfo
old_warn = rospy.logwarn
old_err = rospy.logerr

def log_to_csv(level, msg):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_path, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([now, level, msg])


def loginfo(msg):
    old_info(msg)
    log_to_csv("INFO", msg)

def logwarn(msg):
    old_warn(msg)
    log_to_csv("WARN", msg)

def logerr(msg):
    old_err(msg)
    log_to_csv("ERR", msg)

rospy.loginfo = loginfo
rospy.logwarn = logwarn
rospy.logerr = logerr
# --- konec dodatka ---

def append_detection(x, y):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists or os.stat(csv_path).st_size == 0:
            writer.writerow(["timestamp", "x", "y"])
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([now, x, y])

class ColorDetector:
    def __init__(self):
        rospy.init_node('color_detector_headless', anonymous=True)
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.image_sub = rospy.Subscriber('/detection_left/color/image_raw', Image, self.image_callback)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            lower = np.array([140, 100, 100])
            upper = np.array([170, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                return
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            height, width = cv_image.shape[:2]
            real_x, real_y = self.map_to_grid(cx, cy, width, height)
            # Zapiši detekcijo v CSV
            append_detection(real_x, real_y)
            # Terminalski izpis (ne gre v CSV)
            print(f"Zaznana točka: x={real_x:.2f}, y={real_y:.2f}")
            # Pretvori v PointStamped v base_link ali drug lokalni okvir
            pt_local = PointStamped()
            pt_local.header.stamp = rospy.Time.now()
            pt_local.header.frame_id = "base_link"
            pt_local.point.x = real_x
            pt_local.point.y = real_y
            pt_local.point.z = 0.0
            try:
                pt_world = self.tf_buffer.transform(pt_local, "map", rospy.Duration(0.5))
                # Nič več loganja v CSV, samo terminal
                print(f"Točka v MAP: x={pt_world.point.x:.2f}, y={pt_world.point.y:.2f}")
                subprocess.run("bash /home/nuc/catkin_ws/src/farmbeast_2024/sounds/play.sh /home/nuc/catkin_ws/src/farmbeast_2024/sounds/detection.mp3", shell=True)
            except Exception as e:
                print(f"TF napaka: {e}")
        except Exception as e:
            print(f"Napaka pri obdelavi slike: {e}")

    def map_to_grid(self, x, y, width, height):
        u = x / width
        v = y / height

        left_x = 150 * (1 - v) + 30 * v
        left_y = -80 * (1 - v) + (-30) * v

        right_x = 150 * (1 - v) + 30 * v
        right_y = 80 * (1 - v) + 30 * v

        real_x = left_x * (1 - u) + right_x * u - 40
        real_y = left_y * (1 - u) + right_y * u

        real_x  =real_x/100
        real_y  =real_y/100

        return real_x, real_y

if __name__ == '__main__':
    detector = ColorDetector()
    rospy.spin()
