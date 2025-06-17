#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import sys

class ColorDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Neon Oranžna detektor")
        self.setGeometry(100, 100, 800, 600)

        # Glavni widget in layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # Label za prikaz slike
        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        # Inicializacija ROS
        rospy.init_node('color_detector', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)
        
        # Shranjevanje globinske slike
        self.depth_image = None

        # Timer za posodobitev GUI-ja
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(30)

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            print(f"Napaka pri obdelavi globinske slike: {e}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            # Barvno območje za neon rožnato
            lower = np.array([140, 100, 100])
            upper = np.array([170, 255, 255])
            
            # Uporabi masko
            mask = cv2.inRange(hsv, lower, upper)
            
            # Odstrani šum
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            self.current_mask = mask  # za debug
            cv2.imwrite("debug_mask.png", mask)  # za ročno preverjanje

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
                    cv2.circle(cv_image, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.putText(cv_image, f"X:{real_x:.2f} Y:{real_y:.2f}", 
                                (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Nariši grid (mrežo) na sliko
            grid_color = (200, 200, 200)
            thickness = 1
            height, width = cv_image.shape[:2]
            # Navpične črte
            for i in range(1, 4):
                x = i * width // 4
                cv2.line(cv_image, (x, 0), (x, height), grid_color, thickness)
            # Vodoravne črte
            for i in range(1, 4):
                y = i * height // 4
                cv2.line(cv_image, (0, y), (width, y), grid_color, thickness)
            self.current_image = cv_image
        except Exception as e:
            print(f"Napaka pri obdelavi slike: {e}")

    def update_gui(self):
        if hasattr(self, 'current_image'):
            # Prikaz maske za debug (odkomentiraj spodnje vrstice za test)
            # mask_rgb = cv2.cvtColor(self.current_mask, cv2.COLOR_GRAY2RGB)
            # rgb_image = mask_rgb

            rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setFixedSize(width, height)
            self.image_label.setPixmap(pixmap)

    def map_to_grid(self, x, y, width, height):
        # Koordinate mreže (v tvojem sistemu)
        # (0,0) -> zgornji levi (150, -80)
        # (width,0) -> zgornji desni (150, 80)
        # (0,height) -> spodnji levi (30, -30)
        # (width,height) -> spodnji desni (30, 30)
        u = x / width
        v = y / height

        # Interpolacija po y (zgoraj/spodaj)
        left_x = 150 * (1 - v) + 30 * v
        left_y = -80 * (1 - v) + (-30) * v

        right_x = 150 * (1 - v) + 30 * v
        right_y = 80 * (1 - v) + 30 * v

        # Interpolacija po x (levo/desno)
        real_x = left_x * (1 - u) + right_x * u
        real_y = left_y * (1 - u) + right_y * u

        return real_x, real_y

def main():
    app = QApplication(sys.argv)
    window = ColorDetectorGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 