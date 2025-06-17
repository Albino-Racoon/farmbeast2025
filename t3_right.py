#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import sys
from pathlib import Path
from message_filters import Subscriber, ApproximateTimeSynchronizer
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
import subprocess
import os
from datetime import datetime
import csv

# Load YOLO
sys.path.insert(0, str(Path(__file__).resolve().parent / 'yolov8'))
from ultralytics import YOLO

# Parameters
last_processed_time = 0
PROCESS_INTERVAL = 1.0
camera_frame = 'detection_left_color_optical_frame'  # Your camera frame
world_frame = 'map'  # SLAM world frame

# Load model
model = YOLO('/home/nuc/catkin_ws/src/farmbeast_2024/yolo/best-6.pt')

bridge = CvBridge()
rospy.init_node('yolo_strawberry_world_detector_right')

output_dir = '/home/nuc/catkin_ws/2025_detection'

csv_filename = os.path.join(output_dir, 'detections_right_side.csv')

# Create the file only if it doesn't exist
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Čas', 'Tip objekta', 'X', 'Y', 'Razdalja (mm)', 'Pot do slike'])


# TF setup
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

# Publishers
marker_pub = rospy.Publisher('/strawberry_markers', MarkerArray, queue_size=1)

marker_id_counter = 0  # Global marker ID

def is_orange(hsv_color):
    # Definiramo HSV območje za oranžno barvo
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])
    return cv2.inRange(hsv_color, lower_orange, upper_orange).any()

def is_lemon(hsv_color):
    # Definiramo HSV območje za rumeno barvo
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    return cv2.inRange(hsv_color, lower_yellow, upper_yellow).any()

def synced_callback(color_msg, depth_msg): 
    global last_processed_time, strawberry_bush_detected, marker_id_counter, take_picture
    take_picture = rospy.get_param("/take_picture")
    if (take_picture != True):
        return
    current_time = time.time()

    rospy.set_param("/take_picture", False)
    take_picture = False    
    if current_time - last_processed_time < PROCESS_INTERVAL:
        return

    last_processed_time = current_time
    color_img = bridge.imgmsg_to_cv2(color_msg, desired_encoding='passthrough')
    depth_img = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
    img_rgb = cv2.cvtColor(color_img.copy(), cv2.COLOR_BGR2RGB)

    results = model(img_rgb)
    detections = []

    depth_h, depth_w = depth_img.shape[:2]
    yolo_input_w, yolo_input_h = 1280, 720
    scale_x = depth_w / yolo_input_w
    scale_y = depth_h / yolo_input_h

    try:
        cam_info = rospy.wait_for_message('/detection_right/color/camera_info', CameraInfo, timeout=1.0)
        fx = cam_info.K[0]
        fy = cam_info.K[4]
        cx_intr = cam_info.K[2]
        cy_intr = cam_info.K[5]
    except:
        rospy.logwarn("Failed to get camera intrinsics")
        return

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0].cpu().numpy())
            class_name = model.names[cls].strip().lower()
            conf = float(box.conf[0])

            if class_name == 'pomaranca' or class_name == 'jabolka' or class_name == 'banana' or class_name == 'grozdje' or class_name == 'limona':
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Izrežemo območje objekta in pretvorimo v HSV
                roi = img_rgb[y1:y2, x1:x2]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
                
                # Preverimo barvo za pomarančo/limono
                if class_name == 'pomaranca' or class_name == 'limona':
                    if is_orange(hsv_roi):
                        class_name = 'pomaranca'
                    elif is_lemon(hsv_roi):
                        class_name = 'limona'

                # Prilagojeni pogoji za različne vrste sadja
                if (class_name == 'grozdje' and conf >= 0.2) or \
                   (class_name == 'banana' and conf >= 0.75) or \
                   (class_name in ['pomaranca', 'limona', 'jabolka'] and conf >= 0.3):
                    
                # Nariši bounding box
                if(conf >=0.3):
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Dodaj tekst
                    label = f"{result.names[cls]} {conf:.2f}"
                    cv2.putText(img_rgb, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                depth_val = depth_img[cy, cx]
                if depth_img.dtype == np.uint16:
                    depth_m = depth_val / 1000.0
                elif depth_img.dtype == np.float32:
                    depth_m = float(depth_val)
                else:
                    depth_m = -1.0

                if depth_m <= 0.3 or depth_m > 3.0:
                    continue

                # Convert pixel to 3D camera frame
                x = (cx - cx_intr) * depth_m / fx
                y = (cy - cy_intr) * depth_m / fy
                z = depth_m

                pt_cam = PointStamped()
                pt_cam.header.stamp = rospy.Time(0)
                pt_cam.header.frame_id = camera_frame
                pt_cam.point.x = x
                pt_cam.point.y = y
                pt_cam.point.z = z

                if class_name == 'pomaranca':
                    print("pomaranca")
                    print(conf)
                    subprocess.run("bash " "/home/nuc/catkin_ws/src/farmbeast_2024/sounds/play.sh " "/home/nuc/catkin_ws/src/farmbeast_2024/sounds/orange.mp3", shell=True)
                elif class_name == 'grozdje':
                    print("grozdje")
                    print(conf)
                    subprocess.run("bash " "/home/nuc/catkin_ws/src/farmbeast_2024/sounds/play.sh " "/home/nuc/catkin_ws/src/farmbeast_2024/sounds/grape.mp3", shell=True)
                elif class_name == 'banana':
                    print("banana")
                    print(conf)
                    subprocess.run("bash " "/home/nuc/catkin_ws/src/farmbeast_2024/sounds/play.sh " "/home/nuc/catkin_ws/src/farmbeast_2024/sounds/banana.mp3", shell=True)
                elif class_name == 'limona':
                    print("limona")
                    print(conf)
                    subprocess.run("bash " "/home/nuc/catkin_ws/src/farmbeast_2024/sounds/play.sh " "/home/nuc/catkin_ws/src/farmbeast_2024/sounds/lemon.mp3", shell=True)
                elif class_name == 'jabolka' and conf <= 0.3:
                    print("jabolko")
                    print(conf)
                    subprocess.run("bash " "/home/nuc/catkin_ws/src/farmbeast_2024/sounds/play.sh " "/home/nuc/catkin_ws/src/farmbeast_2024/sounds/apple.mp3", shell=True)

                try:
                    pt_world = tf_buffer.transform(pt_cam, world_frame, rospy.Duration(0.5))
                    detections.append(pt_world)

                    # Zapiši v CSV
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    with open(csv_filename, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([
                            timestamp,
                            class_name,
                            abs(pt_world.point.y) + 1.0,
                            pt_world.point.x +1.0,
                            pt_world.point.z,
                            image_filename,
                        ])

                        # Shrani sliko z detekcijo
                    image_filename = os.path.join(output_dir, f'detection_{timestamp}.jpg')
                    cv2.imwrite(image_filename, img_rgb)
                    

                    rospy.loginfo(f"Detection world pos: ({pt_world.point.x:.2f}, {pt_world.point.y:.2f}, {pt_world.point.z:.2f})")
                except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
                    rospy.logwarn(f"TF transform failed: {e}")
                    continue

if __name__ == '__main__':
    rospy.loginfo("Starting YOLO strawberry detection with persistent markers")

    color_sub = Subscriber('/detection_right/color/image_raw', Image)
    depth_sub = Subscriber('/detection_right/depth/image_rect_raw', Image)

    ats = ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=5, slop=0.1)
    ats.registerCallback(synced_callback)

    rospy.spin()
    cv2.destroyAllWindows()