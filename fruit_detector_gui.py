#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import tkinter as tk
from tkinter import ttk
from PIL import Image as PILImage
from PIL import ImageTk

class FruitDetectorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Detektor Sadja")
        
        # Inicializacija ROS
        rospy.init_node('fruit_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # HSV barvne meje za različno sadje
        self.color_ranges = {
            'pomaranca': {'lower': np.array([5, 100, 100]), 'upper': np.array([15, 255, 255])},
            'limona': {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])},
            'banana': {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])},
            'jabolko': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
            'grozdje': {'lower': np.array([130, 50, 50]), 'upper': np.array([170, 255, 255])}
        }
        
        # GUI elementi
        self.setup_gui()
        
        # ROS subscriber
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", ROSImage, self.image_callback)
        
    def setup_gui(self):
        # Glavni frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video prikaz
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.grid(row=0, column=0, columnspan=2)
        
        # Detekcijska informacija
        self.detection_label = ttk.Label(self.main_frame, text="Detektirano sadje: ")
        self.detection_label.grid(row=1, column=0, columnspan=2)
        
    def image_callback(self, msg):
        try:
            # Pretvorba ROS sporočila v OpenCV sliko
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Pretvorba v HSV
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Detekcija sadja
            detected_fruits = self.detect_fruits(hsv, cv_image)
            
            # Posodobitev GUI
            self.update_gui(cv_image, detected_fruits)
            
        except Exception as e:
            rospy.logerr(e)
            
    def detect_fruits(self, hsv, original):
        detected = []
        
        for fruit, ranges in self.color_ranges.items():
            # Ustvarjanje maske za barvo
            mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
            
            # Iskanje kontur
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Filtriranje premajhnih območij
                    # Analiza oblike
                    if fruit == 'banana':
                        # Preverjanje razmerja stranic za banano
                        x, y, w, h = cv2.boundingRect(contour)
                        if w/h > 2:  # Banana je podolgovata
                            detected.append(fruit)
                    elif fruit == 'limona':
                        # Preverjanje okrogle oblike
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.8:  # Limona je bolj okrogla
                            detected.append(fruit)
                    else:
                        detected.append(fruit)
                        
        return list(set(detected))  # Odstranitev duplikatov
        
    def update_gui(self, cv_image, detected_fruits):
        # Pretvorba OpenCV slike v format za Tkinter
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = PILImage.fromarray(image)
        image = ImageTk.PhotoImage(image)
        
        # Posodobitev GUI
        self.video_label.configure(image=image)
        self.video_label.image = image
        
        # Posodobitev detekcijske informacije
        if detected_fruits:
            self.detection_label.configure(text=f"Detektirano sadje: {', '.join(detected_fruits)}")
        else:
            self.detection_label.configure(text="Detektirano sadje: Nič")
            
    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    try:
        detector = FruitDetectorGUI()
        detector.run()
    except rospy.ROSInterruptException:
        pass 