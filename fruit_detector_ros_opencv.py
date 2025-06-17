#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class FruitDetector:
    def __init__(self):
        # Inicializacija ROS
        rospy.init_node('fruit_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # HSV barvne meje za različno sadje - prilagojene za boljšo detekcijo
        self.color_ranges = {
            'pomaranca': {'lower': np.array([0, 150, 150]), 'upper': np.array([20, 255, 255])},  # Širši razpon za oranžno
            'limona': {'lower': np.array([20, 100, 150]), 'upper': np.array([40, 255, 255])},    # Širši razpon za rumeno
            'banana': {'lower': np.array([15, 50, 150]), 'upper': np.array([35, 255, 255])},     # Širši razpon za rumeno
            'jabolko': {'lower': np.array([0, 100, 50]), 'upper': np.array([10, 255, 150])},     # Temnejša rdeča
            'grozdje': {'lower': np.array([120, 50, 50]), 'upper': np.array([180, 255, 150])}    # Širši razpon za vijolično
        }
        
        # Barve za prikaz imen sadja
        self.colors = {
            'pomaranca': (0, 165, 255),  # BGR format
            'limona': (0, 255, 255),
            'banana': (0, 255, 255),
            'jabolko': (0, 0, 255),
            'grozdje': (255, 0, 255)
        }
        
        # ROS subscriber za sliko iz kamere
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        
        # ROS publisher za označeno sliko
        self.image_pub = rospy.Publisher("/fruit_detector/output", Image, queue_size=1)
        
        print("Začenjam detekcijo sadja...")
        print("Pritisni Ctrl+C za izhod")

    def image_callback(self, msg):
        try:
            # Pretvorba ROS sporočila v OpenCV sliko
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Detekcija sadja
            output = self.detect_fruits(cv_image)
            
            # Pretvorba nazaj v ROS sporočilo in objava
            output_msg = self.bridge.cv2_to_imgmsg(output, "bgr8")
            self.image_pub.publish(output_msg)
            
            # Prikaz rezultatov
            cv2.imshow('Detektor Sadja', output)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(e)

    def detect_fruits(self, frame):
        # Pretvorba v HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Kopija originalne slike za risanje
        output = frame.copy()
        
        for fruit, ranges in self.color_ranges.items():
            # Ustvarjanje maske za barvo
            mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
            
            # Dodatno čiščenje maske
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Iskanje kontur
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Zmanjšana minimalna velikost za boljšo detekcijo
                    # Analiza oblike
                    if fruit == 'banana':
                        # Preverjanje razmerja stranic za banano
                        x, y, w, h = cv2.boundingRect(contour)
                        if w/h > 1.5:  # Zmanjšano razmerje za boljšo detekcijo
                            self.draw_detection(output, contour, fruit)
                    elif fruit == 'limona':
                        # Preverjanje okrogle oblike
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.6:  # Zmanjšana zahtevnost za okroglost
                            self.draw_detection(output, contour, fruit)
                    else:
                        self.draw_detection(output, contour, fruit)
        
        return output

    def draw_detection(self, image, contour, fruit_name):
        # Risanje obrobe
        cv2.drawContours(image, [contour], -1, self.colors[fruit_name], 2)
        
        # Izračun centra za prikaz imena
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Prikaz imena sadja
            cv2.putText(image, fruit_name, (cx - 20, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors[fruit_name], 2)

def main():
    try:
        detector = FruitDetector()
        rospy.spin()
    except KeyboardInterrupt:
        print("Program zaključen.")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 