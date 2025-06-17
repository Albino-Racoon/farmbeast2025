#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import subprocess
import os

# Inicializacija ROS node-a
rospy.init_node('barvna_seg_rdece', anonymous=True)
bridge = CvBridge()

latest_color_image = None
last_sound_time = 0  # Za preprečevanje prevečkratnega predvajanja zvoka

def color_callback(msg):
    global latest_color_image
    try:
        latest_color_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        print(f"Napaka pri pretvorbi slike: {e}")

# Prijava na ROS topic za barvno sliko
rospy.Subscriber("/detection_right/color/image_raw", Image, color_callback)

print("Začenjam barvno segmentacijo za rdečo barvo...")

try:
    while not rospy.is_shutdown():
        if latest_color_image is not None:
            frame = latest_color_image.copy()
            
            # Pretvori sliko v HSV barvni prostor
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Definiraj območje za rdečo barvo v HSV
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            # Ustvari maski za rdečo barvo
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            
            # Združi maski
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Odstrani šum
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Najdi konture
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Izpiši informacije o detekciji
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimalna velikost območja
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        print(f"Zaznana rdeča barva na poziciji: ({cx}, {cy})")
            
            # Predvajaj zvok, če je rdeča barva detektirana in je minilo dovolj časa
            current_time = rospy.get_time()
            if len(contours) > 0 and (current_time - last_sound_time) > 2.0:  # Počakaj 2 sekundi med zvoki
                try:
                    subprocess.run("bash " "/home/nuc/catkin_ws/src/farmbeast_2024/sounds/play.sh " "/home/nuc/catkin_ws/src/farmbeast_2024/sounds/detection_right.mp3", shell=True)
                    last_sound_time = current_time
                except Exception as e:
                    print(f"Napaka pri predvajanju zvoka: {e}")

        rospy.sleep(0.05)

except Exception as e:
    print(f"Napaka: {e}")
finally:
    print("Zaustavljam barvno segmentacijo.") 