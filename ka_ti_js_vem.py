#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ColorDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.depth_image = None
        rospy.init_node('color_detector_headless', anonymous=True)
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            print(f"Napaka pri obdelavi globinske slike: {e}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            lower = np.array([140, 180, 150])
            upper = np.array([170, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    height, width = cv_image.shape[:2]
                    real_x, real_y = self.map_to_grid(cx, cy, width, height)
                    print(f"Pozicija na mreži: X={real_x:.2f}, Y={real_y:.2f}")
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

        return real_x, real_y

if __name__ == '__main__':
    detector = ColorDetector()
    rospy.spin()
