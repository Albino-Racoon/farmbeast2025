#!/usr/bin/env python3
import cv2
import torch
from ultralytics import YOLO
import csv
from datetime import datetime
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os

# Inicializacija ROS node-a
rospy.init_node('object_detection_gui', anonymous=True)
bridge = CvBridge()

# Naloži YOLO model
model = YOLO('/home/rakun/Desktop/Yolo_det_once/best-6.pt')

latest_color_image = None

def color_callback(msg):
    global latest_color_image
    try:
        latest_color_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        print(f"Napaka pri pretvorbi slike: {e}")

# Prijava na ROS topic za barvno sliko
rospy.Subscriber("/camera/color/image_raw", Image, color_callback)

print("Začenjam detekcijo objektov...")

# Ustvari mapo za shranjevanje slik
output_dir = '/home/rakun/Desktop/Yolo_det_once/t3_valid'
os.makedirs(output_dir, exist_ok=True)

# Pripravi CSV datoteko
csv_filename = os.path.join(output_dir, 'detekcije_t3.csv')
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Čas', 'Tip objekta', 'X', 'Y', 'Razdalja (mm)', 'Pot do slike'])

try:
    while not rospy.is_shutdown():
        if latest_color_image is not None:
            frame = latest_color_image.copy()
            
            # Izvedi detekcijo
            results = model(frame, conf=0.2)
            
            # Obdelaj rezultate
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Pridobi koordinate
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Pridobi klaso in zaupanje
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Nariši bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Dodaj tekst
                    label = f"{result.names[cls]} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Izpiši informacije
                    print(f"\nZaznan objekt: {result.names[cls]}")
                    print(f"Pozicija: ({x1}, {y1}) - ({x2}, {y2})")
                    print(f"Zaupanje: {conf:.2f}")

            # Prikaži rezultate
            cv2.imshow("Object Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if results[0].boxes is not None:
                result = results[0]
                current_detections = set()
                
                # Za vsak zaznan objekt
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = result.names[cls]

                    # Preveri confidence threshold
                    if class_name == "banana" and conf <= 0.75:
                        continue
                    elif class_name != "banana" and conf <= 0.3:
                        continue

                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, coords)
                    
                    # Preverimo, da so koordinate veljavne
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    if x2 > frame.shape[1]: x2 = frame.shape[1]
                    if y2 > frame.shape[0]: y2 = frame.shape[0]
                    
                    # Preverimo, da je ROI veljaven
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    roi = frame[y1:y2, x1:x2]
                    
                    # Preverimo, da ROI ni prazen
                    if roi.size == 0:
                        continue

                    # Pretvori ROI v HSV
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                    # Definiraj HSV meje za limono (rumena) in pomarančo (oranžna)
                    yellow_lower = np.array([20, 100, 100])
                    yellow_upper = np.array([35, 255, 255])
                    orange_lower = np.array([10, 100, 100])
                    orange_upper = np.array([20, 255, 255])

                    # Ugotovi prevladujočo barvo v ROI
                    yellow_mask = cv2.inRange(hsv_roi, yellow_lower, yellow_upper)
                    orange_mask = cv2.inRange(hsv_roi, orange_lower, orange_upper)
                    yellow_ratio = cv2.countNonZero(yellow_mask) / (roi.size / 3)
                    orange_ratio = cv2.countNonZero(orange_mask) / (roi.size / 3)

                    # Določi razred glede na barvo
                    if yellow_ratio > 0.2 and orange_ratio <= 0.2:
                        class_name = "lemon"
                    elif orange_ratio > 0.2 and yellow_ratio <= 0.2:
                        class_name = "orange"
                    else:
                        continue  # Preskoči če ni jasna barva

                    # Izpiši informacije samo za objekte, ki so prešli preverjanje
                    print(f"\nZaznan objekt: {class_name}")
                    print(f"Pozicija: ({x1}, {y1}) - ({x2}, {y2})")
                    print(f"Zaupanje: {conf:.2f}")
                    print(f"Rumena barva: {yellow_ratio:.2f}, Oranžna barva: {orange_ratio:.2f}")

                    # Izračunaj središče objekta
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    current_center = (center_x, center_y)
                    
                    try:
                        # Shrani sliko z detekcijo
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_filename = os.path.join(output_dir, f'detection_{timestamp}.jpg')
                        cv2.imwrite(image_filename, frame)
                        
                        # Zapiši v CSV
                        with open(csv_filename, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([
                                timestamp,
                                class_name,
                                center_x,
                                center_y,
                                'N/A',  # Razdalja do objekta (mm)
                                image_filename
                            ])
                        
                        current_detections.add(current_center)
                        
                        # Izriši okvir in pot
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Napaka pri risanju/zapisovanju: {e}")
                        continue

        rospy.sleep(0.05)

except Exception as e:
    print(f"Napaka: {e}")
finally:
    cv2.destroyAllWindows()
    print("Zaustavljam detekcijo objektov.")