#!/usr/bin/env python3
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import csv
import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2

# Inicializacija YOLO modela
model = YOLO('/home/rakun/Desktop/Yolo_det_once/last-3.pt')

# Inicializacija ROS node-a
rospy.init_node('depth_detection', anonymous=True)
bridge = CvBridge()

# Globalne spremenljivke za shranjevanje zadnjih prejetih slik in podatkov
latest_color_image = None
latest_depth_image = None
latest_pointcloud = None

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

# Pripravi CSV datoteko
csv_filename = f'depth_tracking_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Objekt', 'X', 'Y', 'Razdalja_mm', 'X3D', 'Y3D', 'Z3D', 'Čas'])

print("Čakam na slike iz RealSense kamere...")

def get_point_3d(pointcloud, x, y):
    """Pridobi 3D točko iz point clouda za dano 2D točko"""
    width = pointcloud.width
    points = pc2.read_points(pointcloud, field_names=("x", "y", "z"), skip_nans=True)
    point_idx = y * width + x
    for i, point in enumerate(points):
        if i == point_idx:
            return point
    return None

def get_depth(depth_image, x, y, kernel_size=5):
    """Pridobi povprečno globino iz območja okoli točke (x,y)"""
    half_kernel = kernel_size // 2
    height, width = depth_image.shape
    
    # Preveri meje
    x_start = max(0, x - half_kernel)
    x_end = min(width, x + half_kernel + 1)
    y_start = max(0, y - half_kernel)
    y_end = min(height, y + half_kernel + 1)
    
    # Pridobi območje globinske slike
    region = depth_image[y_start:y_end, x_start:x_end]
    
    # Filtriraj neveljavne vrednosti (0 ali zelo velike vrednosti)
    valid_depths = region[region > 0]
    valid_depths = valid_depths[valid_depths < 10000]  # Ignoriraj vrednosti večje od 10m
    
    if len(valid_depths) > 0:
        # Vrni mediano veljavnih vrednosti
        return np.median(valid_depths)
    return 0

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
                
                # YOLO detekcija
                results = model(color_image, verbose=False)
                
                if results[0].boxes is not None:
                    result = results[0]
                    
                    # Za vsak zaznan objekt
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        if conf > 0.3:  # Minimalno zaupanje
                            # Pridobi koordinate okvirja
                            coords = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = map(int, coords)
                            
                            # Izračunaj središče objekta
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # Skaliraj koordinate za globinsko sliko
                            depth_x = int(center_x * depth_width / color_width)
                            depth_y = int(center_y * depth_height / color_height)
                            
                            # Preveri meje
                            depth_x = min(max(0, depth_x), depth_width - 1)
                            depth_y = min(max(0, depth_y), depth_height - 1)
                            
                            # Pridobi razred
                            cls = int(box.cls[0])
                            class_name = result.names[cls]
                            
                            try:
                                # Pridobi globino na središču objekta (v milimetrih)
                                depth = get_depth(depth_image, depth_x, depth_y)
                                
                                # Če je globina 0, poskusi z večjim območjem
                                if depth == 0:
                                    depth = get_depth(depth_image, depth_x, depth_y, kernel_size=11)
                                
                                # Če še vedno ni veljavne globine, preskoči
                                if depth == 0:
                                    continue
                                
                                # Pretvori v milimetre
                                depth = depth * 1000
                                
                                # Poskusi pridobiti 3D pozicijo iz point clouda
                                point_3d = None
                                if latest_pointcloud is not None:
                                    point_3d = get_point_3d(latest_pointcloud, depth_x, depth_y)
                                
                                # Izpiši podatke v konzolo samo če je globina veljavna
                                print(f"\nZaznan objekt: {class_name}")
                                print(f"2D pozicija (x,y): ({center_x}, {center_y})")
                                print(f"Globina: {depth:.0f}mm")
                                if point_3d:
                                    print(f"3D pozicija (x,y,z): ({point_3d[0]:.3f}m, {point_3d[1]:.3f}m, {point_3d[2]:.3f}m)")
                                
                                # Izriši okvir in informacije
                                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                text = f'{class_name} {conf:.2f} - {depth:.0f}mm'
                                cv2.putText(color_image, text, (x1, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                # Zapiši v CSV
                                with open(csv_filename, 'a', newline='') as csvfile:
                                    writer = csv.writer(csvfile)
                                    row = [
                                        class_name,
                                        center_x,
                                        center_y,
                                        f'{depth:.0f}'
                                    ]
                                    if point_3d:
                                        row.extend([f'{p:.3f}' for p in point_3d])
                                    else:
                                        row.extend(['', '', ''])
                                    row.append(datetime.now().strftime('%H:%M:%S.%f'))
                                    writer.writerow(row)
                                    
                            except (IndexError, ValueError) as e:
                                print(f"Napaka pri merjenju globine: {e}")
                                continue

                # Prikaži sliko
                cv2.imshow('RealSense Depth Detection', color_image)
                
                # Izhod s tipko 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"Napaka v glavni zanki: {e}")
                continue

finally:
    cv2.destroyAllWindows() 