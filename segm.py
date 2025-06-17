#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2

# Inicializacija ROS node-a
rospy.init_node('color_depth_detection', anonymous=True)
bridge = CvBridge()

# Globalne spremenljivke za shranjevanje zadnjih prejetih slik in podatkov
latest_color_image = None
latest_depth_image = None
latest_pointcloud = None

# HSV območja za različne barve (lahko prilagodite glede na vaše potrebe)
color_ranges = {
    'orange': (np.array([5, 50, 50]), np.array([15, 255, 255])),
    'blue': (np.array([100, 50, 50]), np.array([130, 255, 255])),
    'yellow': (np.array([20, 50, 50]), np.array([30, 255, 255]))
}

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
rospy.Subscriber("/camera/depth/image_rect_raw", Image, depth_callback)
rospy.Subscriber("/camera/depth/color/points", PointCloud2, pointcloud_callback)

def get_depth(depth_image, x, y, kernel_size=5):
    """Pridobi povprečno globino iz območja okoli točke (x,y)"""
    half_kernel = kernel_size // 2
    height, width = depth_image.shape
    
    # Preveri, če so koordinate znotraj mej
    if x >= width or y >= height or x < 0 or y < 0:
        return 0
    
    x_start = max(0, x - half_kernel)
    x_end = min(width, x + half_kernel + 1)
    y_start = max(0, y - half_kernel)
    y_end = min(height, y + half_kernel + 1)
    
    # Pridobi območje globinske slike
    region = depth_image[y_start:y_end, x_start:x_end]
    
    # Pretvori v float in iz milimetrov v metre
    region = region.astype(float)
    
    # Filtriraj neveljavne vrednosti
    valid_depths = region[region > 0]
    valid_depths = valid_depths[valid_depths < 10000]
    
    if len(valid_depths) > 0:
        return float(np.median(valid_depths))
    return 0

def process_color(color_image, depth_image, color_name, hsv_range):
    """Procesira sliko za določeno barvo in vrne masko ter središče največjega objekta"""
    # Pretvori v HSV
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    
    # Ustvari masko za določeno barvo
    mask = cv2.inRange(hsv, hsv_range[0], hsv_range[1])
    
    # Odstrani šum
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Najdi konture
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Najdi največjo konturo
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 100:  # Minimalna velikost območja
            # Najdi središče
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Pridobi globino
                depth = get_depth(depth_image, cx, cy)
                
                return mask, (cx, cy), depth, largest_contour
    
    return mask, None, None, None

try:
    while not rospy.is_shutdown():
        if latest_color_image is not None and latest_depth_image is not None:
            try:
                # Naredi kopijo slik
                color_image = latest_color_image.copy()
                depth_image = latest_depth_image.copy()
                
                # Pridobi dimenzije slik
                color_height, color_width = color_image.shape[:2]
                depth_height, depth_width = depth_image.shape[:2]
                
                # Procesiraj vsako barvo
                for color_name, hsv_range in color_ranges.items():
                    mask, center, depth, contour = process_color(color_image, depth_image, color_name, hsv_range)
                    
                    if center is not None:
                        cx, cy = center
                        
                        # Skaliraj koordinate za globinsko sliko
                        depth_x = int(cx * depth_width / color_width)
                        depth_y = int(cy * depth_height / color_height)
                        
                        # Pridobi globino
                        depth = get_depth(depth_image, depth_x, depth_y)
                        depth_mm = depth  # Globina je že v milimetrih
                        
                        if depth_mm > 0:  # Izpiši samo če je globina veljavna
                            # Izriši konturo in informacije
                            cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 2)
                            cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
                            text = f'{color_name}: {depth_mm:.0f}mm'
                            cv2.putText(color_image, text, (cx - 10, cy - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Izpiši v konzolo
                            print(f"\nZaznana barva: {color_name}")
                            print(f"Pozicija (x,y): ({cx}, {cy})")
                            print(f"Razdalja: {depth_mm:.0f}mm")
                        
                        # Prikaži masko za to barvo
                        cv2.imshow(f'Mask {color_name}', mask)
                
                # Prikaži originalno sliko z označbami
                cv2.imshow('Color Detection', color_image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"Napaka v glavni zanki: {e}")
                continue

except Exception as e:
    print(f"Napaka: {e}")
finally:
    cv2.destroyAllWindows() 