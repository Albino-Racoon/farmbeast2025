#!/usr/bin/env python3
import cv2
import torch
from ultralytics import YOLO
import csv
from datetime import datetime, timedelta
import numpy as np
from pygame import mixer
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Inicializacija pygame mixer-ja za zvok
mixer.init()
sound = mixer.Sound('/home/rakun/Desktop/Yolo_det_once/man-scream-121085.mp3')

# Inicializacija ROS node-a
rospy.init_node('object_detection', anonymous=True)
bridge = CvBridge()

# Naloži YOLO model
model = YOLO('/home/rakun/Desktop/Yolo_det_once/last-3.pt')

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

# Slovar za sledenje objektom
# {(x,y): {'start_pos': (x,y), 'current_pos': (x,y), 'start_time': timestamp}}
tracked_objects = {}

# Dodaj slovar za začasno hranjenje izginulih objektov
# {(x,y): {'objekt': tracked_object_data, 'last_seen': timestamp}}
temporary_lost = {}

# Pripravi CSV datoteko
csv_filename = f'object_tracking_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Objekt', 'Začetni_X', 'Začetni_Y', 'Končni_X', 'Končni_Y', 'Začetni_čas', 'Končni_čas'])

# Maksimalna razdalja za sledenje istemu objektu
MAX_TRACKING_DISTANCE = 50  # pikslov

# Nastavi maksimalni čas za timeout in maksimalno razdaljo za ponovno povezavo
MAX_TIMEOUT = timedelta(seconds=2)
MAX_RECONNECT_DISTANCE = 30  # pikslov

try:
    while not rospy.is_shutdown():
        if latest_color_image is not None:
            frame = latest_color_image.copy()
            
            # Izvedi detekcijo
            results = model(frame, conf=0.3)
            
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
                    if conf > 0.3:
                        # Pridobi koordinate okvirja
                        coords = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, coords)
                        
                        # Izračunaj središče objekta
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        current_center = (center_x, center_y)
                        
                        # Pridobi razred
                        cls = int(box.cls[0])
                        class_name = result.names[cls]
                        
                        # Poišči najbližji sledeči objekt (vključno z začasno izgubljenimi)
                        min_dist = float('inf')
                        closest_tracked = None
                        is_temporary = False
                        
                        # Preveri aktivne objekte
                        for tracked_center in tracked_objects.keys():
                            dist = np.sqrt((center_x - tracked_center[0])**2 + 
                                         (center_y - tracked_center[1])**2)
                            if dist < min_dist and dist < MAX_TRACKING_DISTANCE:
                                min_dist = dist
                                closest_tracked = tracked_center
                                is_temporary = False
                        
                        # Preveri začasno izgubljene objekte
                        current_time = datetime.now()
                        for temp_center, temp_data in list(temporary_lost.items()):
                            if (current_time - temp_data['last_seen']) < MAX_TIMEOUT:
                                dist = np.sqrt((center_x - temp_data['objekt']['current_pos'][0])**2 + 
                                             (center_y - temp_data['objekt']['current_pos'][1])**2)
                                if (dist < min_dist and dist < MAX_RECONNECT_DISTANCE and 
                                    temp_data['objekt']['class_name'] == class_name):
                                    min_dist = dist
                                    closest_tracked = temp_center
                                    is_temporary = True
                            else:
                                # Odstrani prestare začasno izgubljene objekte
                                del temporary_lost[temp_center]
                        
                        if closest_tracked is None:
                            # Nov objekt
                            tracked_objects[current_center] = {
                                'start_pos': current_center,
                                'current_pos': current_center,
                                'start_time': current_time,
                                'class_name': class_name
                            }
                            sound.play()
                        else:
                            # Posodobi pozicijo obstoječega objekta
                            if is_temporary:
                                tracked_objects[current_center] = temporary_lost[closest_tracked]['objekt']
                                del temporary_lost[closest_tracked]
                            else:
                                tracked_objects[current_center] = tracked_objects.pop(closest_tracked)
                            tracked_objects[current_center]['current_pos'] = current_center
                        
                        current_detections.add(current_center)
                        
                        # Izriši okvir in pot
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Izriši pot
                        start_pos = tracked_objects[current_center]['start_pos']
                        cv2.line(frame, start_pos, current_center, (255, 0, 0), 2)
                
                # Preveri za izginule objekte
                tracked_centers = set(tracked_objects.keys())
                disappeared = tracked_centers - current_detections
                
                # Premakni izginule objekte v začasno shrambo ali jih zapiši v CSV
                if disappeared:
                    current_time = datetime.now()
                    for center in disappeared:
                        obj = tracked_objects[center]
                        # Shrani v začasno shrambo
                        temporary_lost[center] = {
                            'objekt': obj,
                            'last_seen': current_time
                        }
                        del tracked_objects[center]

                # Preveri začasno izgubljene objekte za timeout
                for temp_center, temp_data in list(temporary_lost.items()):
                    if (current_time - temp_data['last_seen']) >= MAX_TIMEOUT:
                        # Zapiši v CSV in odstrani
                        with open(csv_filename, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            obj = temp_data['objekt']
                            writer.writerow([
                                obj['class_name'],
                                obj['start_pos'][0],
                                obj['start_pos'][1],
                                obj['current_pos'][0],
                                obj['current_pos'][1],
                                obj['start_time'].strftime('%H:%M:%S.%f'),
                                current_time.strftime('%H:%M:%S.%f')
                            ])
                        del temporary_lost[temp_center]

        rospy.sleep(0.05)

except Exception as e:
    print(f"Napaka: {e}")
finally:
    # Zapiši še preostale objekte v CSV
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for center, obj in tracked_objects.items():
            writer.writerow([
                obj['class_name'],
                obj['start_pos'][0],
                obj['start_pos'][1],
                obj['current_pos'][0],
                obj['current_pos'][1],
                obj['start_time'].strftime('%H:%M:%S.%f'),
                datetime.now().strftime('%H:%M:%S.%f')
            ])

    cv2.destroyAllWindows()
    print("Zaustavljam detekcijo objektov.") 