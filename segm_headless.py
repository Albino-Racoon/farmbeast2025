#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import torch
from ultralytics import YOLO

# Inicializacija ROS node-a
rospy.init_node('color_depth_detection', anonymous=True)
bridge = CvBridge()

# Globalne spremenljivke za shranjevanje zadnjih prejetih slik in podatkov
latest_color_image = None
latest_depth_image = None
latest_pointcloud = None

# HSV območja za različne barve
color_ranges = {
    'orange': (np.array([5, 50, 50]), np.array([15, 255, 255])),
    'blue': (np.array([100, 50, 50]), np.array([130, 255, 255])),
    'yellow': (np.array([20, 50, 50]), np.array([30, 255, 255]))
}

# Naloži YOLO model
model = YOLO('/home/rakun/Desktop/YOLO-formula/runs/fruit_train/exp/weights/best.pt')

def color_callback(msg):
    global latest_color_image
    latest_color_image = bridge.imgmsg_to_cv2(msg, "bgr8")

def depth_callback(msg):
    global latest_depth_image
    latest_depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

def pointcloud_callback(msg):
    global latest_pointcloud
    latest_pointcloud = msg

# Prijava na ROS teme
rospy.Subscriber("/camera/color/image_raw", Image, color_callback)
rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_callback)
# rospy.Subscriber("/camera/depth/color/points", PointCloud2, pointcloud_callback)  # če želiš pointcloud

def get_depth(depth_image, x, y, kernel_size=5):
    """Pridobi povprečno globino iz območja okoli točke (x,y)"""
    half_kernel = kernel_size // 2
    height, width = depth_image.shape
    
    if x >= width or y >= height or x < 0 or y < 0:
        return 0
    
    x_start = max(0, x - half_kernel)
    x_end = min(width, x + half_kernel + 1)
    y_start = max(0, y - half_kernel)
    y_end = min(height, y + half_kernel + 1)
    
    region = depth_image[y_start:y_end, x_start:x_end]
    region = region.astype(float)
    
    valid_depths = region[region > 0]
    valid_depths = valid_depths[valid_depths < 10000]
    
    if len(valid_depths) > 0:
        return float(np.median(valid_depths))
    return 0

def process_color(color_image, depth_image, color_name, hsv_range):
    """Procesira sliko za določeno barvo in vrne središče največjega objekta"""
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_range[0], hsv_range[1])
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 100:  # Minimalna velikost območja
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy), area
    
    return None, None

print("Začenjam detekcijo barv in YOLO detekcijo...")

try:
    while not rospy.is_shutdown():
        if latest_color_image is not None and latest_depth_image is not None:
            try:
                color_image = latest_color_image.copy()
                depth_image = latest_depth_image.copy()
                
                color_height, color_width = color_image.shape[:2]
                depth_height, depth_width = depth_image.shape[:2]
                
                # --- YOLO DETEKCIJA ---
                results = model(color_image, verbose=False)
                if results[0].boxes is not None:
                    result = results[0]
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        if conf > 0.3:
                            coords = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = map(int, coords)
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            # Pridobi globino
                            depth = get_depth(depth_image, center_x, center_y)
                            depth_mm = depth
                            cls = int(box.cls[0])
                            class_name = result.names[cls]
                            if depth_mm > 0:
                                print(f"\nYOLO zaznan objekt: {class_name}")
                                print(f"Pozicija (x,y): ({center_x}, {center_y})")
                                print(f"Razdalja: {depth_mm:.0f}mm")
                
                # --- BARVNA SEGMENTACIJA (ostane kot prej) ---
                for color_name, hsv_range in color_ranges.items():
                    center, area = process_color(color_image, depth_image, color_name, hsv_range)
                    if center is not None:
                        cx, cy = center
                        depth = get_depth(depth_image, cx, cy)
                        depth_mm = depth
                        if depth_mm > 0:
                            print(f"\nZaznana barva: {color_name}")
                            print(f"Pozicija (x,y): ({cx}, {cy})")
                            print(f"Površina: {area:.0f} pikslov")
                            print(f"Razdalja: {depth_mm:.0f}mm")
                            
            except Exception as e:
                print(f"Napaka v glavni zanki: {e}")
                continue

        rospy.sleep(0.1)  # Dodaj malo zamika, da ne obremenjujemo procesorja

except Exception as e:
    print(f"Napaka: {e}")
finally:
    print("Zaustavljam detekcijo barv in YOLO detekcijo...") 