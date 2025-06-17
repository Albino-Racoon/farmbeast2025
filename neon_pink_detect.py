#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QSpinBox, QHBoxLayout, QGroupBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import sys

class NeonPinkDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neon Pink Detector")
        self.setGeometry(100, 100, 1200, 800)

        # Glavni widget in layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        main_widget.setLayout(layout)

        # Leva stran - prikaz slike
        left_layout = QVBoxLayout()
        self.image_label = QLabel()
        left_layout.addWidget(self.image_label)
        layout.addLayout(left_layout)

        # Desna stran - kontrole
        right_layout = QVBoxLayout()
        
        # Koordinate za vsak kvadrant
        self.coordinates = {
            'bottom_middle': {'x': 30, 'y': 0},
            'bottom_left': {'x': 30, 'y': -30},
            'bottom_right': {'x': 30, 'y': 30},
            'middle_top': {'x': 160, 'y': 0},
            'top_right': {'x': 150, 'y': 80},
            'top_left': {'x': 150, 'y': -80}
        }

        # Ustvari kontrole za vsak kvadrant
        for key in self.coordinates:
            group = QGroupBox(key.replace('_', ' ').title())
            group_layout = QHBoxLayout()
            
            # X koordinata
            x_spin = QSpinBox()
            x_spin.setRange(-200, 200)
            x_spin.setValue(self.coordinates[key]['x'])
            x_spin.valueChanged.connect(lambda v, k=key: self.update_coordinate(k, 'x', v))
            group_layout.addWidget(QLabel('X:'))
            group_layout.addWidget(x_spin)
            
            # Y koordinata
            y_spin = QSpinBox()
            y_spin.setRange(-200, 200)
            y_spin.setValue(self.coordinates[key]['y'])
            y_spin.valueChanged.connect(lambda v, k=key: self.update_coordinate(k, 'y', v))
            group_layout.addWidget(QLabel('Y:'))
            group_layout.addWidget(y_spin)
            
            group.setLayout(group_layout)
            right_layout.addWidget(group)

        layout.addLayout(right_layout)

        # Inicializacija ROS
        rospy.init_node('neon_pink_detection', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        
        # Timer za posodobitev GUI-ja
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(30)

        self.current_image = None
        self.current_mask = None

    def update_coordinate(self, key, axis, value):
        self.coordinates[key][axis] = value

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
            
            self.current_mask = mask

            # Najdi konture
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > 100:
                    M = cv2.moments(largest)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.drawContours(cv_image, [largest], -1, (0,255,0), 2)
                        cv2.circle(cv_image, (cx, cy), 5, (0,0,255), -1)
                        
                        # Izračunaj realne koordinate
                        height, width = cv_image.shape[:2]
                        real_x, real_y = self.map_to_grid(cx, cy, width, height)
                        print(f"\nNeon pink zaznana na: ({real_x:.2f}, {real_y:.2f})")
                        cv2.putText(cv_image, f"X:{real_x:.1f} Y:{real_y:.1f}", 
                                  (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Nariši grid
            self.draw_grid(cv_image)
            self.current_image = cv_image

        except Exception as e:
            print(f"Napaka pri obdelavi slike: {e}")

    def draw_grid(self, image):
        height, width = image.shape[:2]
        grid_color = (200, 200, 200)
        thickness = 1
        
        # Navpične črte
        for i in range(1, 4):
            x = i * width // 4
            cv2.line(image, (x, 0), (x, height), grid_color, thickness)
        
        # Vodoravne črte
        for i in range(1, 4):
            y = i * height // 4
            cv2.line(image, (0, y), (width, y), grid_color, thickness)

    def map_to_grid(self, x, y, width, height):
        u = x / width
        v = y / height

        # Določi kvadrant
        if v > 0.75:  # Spodnji del
            if u < 0.25:  # Levo
                target = self.coordinates['bottom_left']
            elif u > 0.75:  # Desno
                target = self.coordinates['bottom_right']
            else:  # Sredina
                target = self.coordinates['bottom_middle']
        elif v < 0.25:  # Zgornji del
            if u < 0.25:  # Levo
                target = self.coordinates['top_left']
            elif u > 0.75:  # Desno
                target = self.coordinates['top_right']
            else:  # Sredina
                target = self.coordinates['middle_top']
        else:  # Srednji del
            if u < 0.25:  # Levo
                target = self.coordinates['top_left']
            elif u > 0.75:  # Desno
                target = self.coordinates['top_right']
            else:  # Sredina
                target = self.coordinates['middle_top']

        return target['x'], target['y']

    def update_gui(self):
        if self.current_image is not None:
            rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)

def main():
    app = QApplication(sys.argv)
    window = NeonPinkDetectorGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()