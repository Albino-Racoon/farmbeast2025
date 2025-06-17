#!/usr/bin/env python3
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from datetime import datetime
import csv

# Inicializacija ROS node-a
rospy.init_node('sadje_detection', anonymous=True)
bridge = CvBridge()

# Globalne spremenljivke
latest_color_image = None

# Naloži YOLO model
model = YOLO('/home/rakun/Desktop/Yolo_det_once/last-3.pt')

# Izpiši vse razrede, ki jih model pozna
print("\nRazredi v modelu:")
for i, name in enumerate(model.names.values()):
    print(f"{i}: {name}")

def color_callback(msg):
    global latest_color_image
    latest_color_image = bridge.imgmsg_to_cv2(msg, "bgr8")

# Prijava na ROS temo za barvno sliko
rospy.Subscriber("/camera/color/image_raw", Image, color_callback)

# Pripravi CSV datoteko za beleženje detekcij
csv_filename = f'sadje_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Objekt', 'X', 'Y', 'Širina', 'Višina', 'Zaupanje', 'Čas'])

print("Začenjam detekcijo sadja...")

try:
    while not rospy.is_shutdown():
        if latest_color_image is not None:
            try:
                # Naredi kopijo slike
                frame = latest_color_image.copy()
                
                # YOLO detekcija
                results = model(frame, verbose=False)
                
                if results[0].boxes is not None:
                    result = results[0]
                    
                    # Za vsak zaznan objekt
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        if conf > 0.1:  # Minimalno zaupanje
                            # Pridobi koordinate okvirja
                            coords = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = map(int, coords)
                            
                            # Pridobi razred in zaupanje
                            cls = int(box.cls[0])
                            class_name = result.names[cls]
                            
                            # Izpiši vse razrede in njihovo zaupanje
                            print("\nVse možne razrede in zaupanje:")
                            for i, name in enumerate(result.names.values()):
                                print(f"{name}: {float(box.conf[i]) if i < len(box.conf) else 0:.2f}")
                            
                            # Izračunaj središče objekta
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # Izračunaj širino in višino
                            width = x2 - x1
                            height = y2 - y1
                            
                            # Izriši okvir in informacije
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            text = f'{class_name} {conf:.2f}'
                            cv2.putText(frame, text, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Zapiši v CSV
                            with open(csv_filename, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow([
                                    class_name,
                                    center_x,
                                    center_y,
                                    width,
                                    height,
                                    f'{conf:.2f}',
                                    datetime.now().strftime('%H:%M:%S.%f')
                                ])
                            
                            # Izpiši v konzolo
                            print(f"\nZaznan objekt: {class_name}")
                            print(f"Pozicija (x,y): ({center_x}, {center_y})")
                            print(f"Velikost (širina,višina): ({width}, {height})")
                            print(f"Zaupanje: {conf:.2f}")
                
                # Prikaži sliko
                cv2.imshow('Sadje Detection', frame)
                
                # Izhod s tipko 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"Napaka v glavni zanki: {e}")
                continue

        rospy.sleep(0.1)  # Dodaj malo zamika

except Exception as e:
    print(f"Napaka: {e}")
finally:
    cv2.destroyAllWindows()
    print("Zaustavljam detekcijo sadja...") 