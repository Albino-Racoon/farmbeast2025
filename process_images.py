#!/usr/bin/env python3
import cv2
import os
import numpy as np
from ultralytics import YOLO
import json

class ImageProcessor:
    def __init__(self, image_dir, model_path):
        self.image_dir = image_dir
        self.model = YOLO(model_path)
        self.images = []
        self.current_idx = 0
        self.results = {}
        self.window_name = "YOLO Detekcija"
        
        # Naloži vse slike
        self.load_images()
        
        # Obdelaj vse slike
        self.process_all_images()
        
        # Shrani rezultate
        self.save_results()
        
        # Ustvari okno z možnostjo spreminjanja velikosti
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Nastavi začetno velikost okna
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        # Zaženi GUI
        self.run_gui()
    
    def load_images(self):
        """Naloži vse slike iz mape"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.images = [f for f in os.listdir(self.image_dir) 
                      if os.path.splitext(f)[1].lower() in valid_extensions]
        self.images.sort()
        print(f"Našel {len(self.images)} slik")
    
    def process_all_images(self):
        """Obdelaj vse slike z YOLO modelom"""
        for img_name in self.images:
            img_path = os.path.join(self.image_dir, img_name)
            img = cv2.imread(img_path)
            
            # Izvedi detekcijo
            results = self.model(img, conf=0.3)
            
            # Shrani rezultate
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Pridobi koordinate
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Pridobi klaso in zaupanje
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    detections.append({
                        'class': result.names[cls],
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
            
            self.results[img_name] = detections
    
    def save_results(self):
        """Shrani rezultate v JSON datoteko"""
        with open('detection_results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
    
    def show_current_image(self):
        """Prikaži trenutno sliko z detekcijami"""
        if not self.images:
            return
        
        img_name = self.images[self.current_idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = cv2.imread(img_path)
        
        # Nariši detekcije
        for detection in self.results[img_name]:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            cls = detection['class']
            
            # Nariši bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Dodaj tekst
            label = f"{cls} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Dodaj informacije o sliki
        cv2.putText(img, f"Slika {self.current_idx + 1}/{len(self.images)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Prikaži sliko
        cv2.imshow(self.window_name, img)
    
    def run_gui(self):
        """Zaženi GUI za pregledovanje slik"""
        self.show_current_image()
        
        print("\nKontrole:")
        print("n - naslednja slika")
        print("p - prejšnja slika")
        print("i - izpiši informacije")
        print("q - izhod")
        print("Okno lahko spreminjate velikost z miško")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):  # Izhod
                break
            elif key == ord('n'):  # Naslednja slika
                self.current_idx = (self.current_idx + 1) % len(self.images)
                self.show_current_image()
            elif key == ord('p'):  # Prejšnja slika
                self.current_idx = (self.current_idx - 1) % len(self.images)
                self.show_current_image()
            elif key == ord('i'):  # Izpiši informacije o trenutni sliki
                self.print_current_info()
        
        cv2.destroyAllWindows()
    
    def print_current_info(self):
        """Izpiši informacije o trenutni sliki"""
        if not self.images:
            return
        
        img_name = self.images[self.current_idx]
        print(f"\nInformacije za sliko: {img_name}")
        print(f"Številka slike: {self.current_idx + 1}/{len(self.images)}")
        print("Zaznani objekti:")
        
        for i, detection in enumerate(self.results[img_name], 1):
            print(f"\nObjekt {i}:")
            print(f"  Razred: {detection['class']}")
            print(f"  Zaupanje: {detection['confidence']:.2f}")
            print(f"  Koordinate: {detection['bbox']}")

if __name__ == "__main__":
    # Nastavi poti
    image_dir = "/media/rakun/3853-1F57/slike_t3"
    model_path = "/home/rakun/Desktop/Yolo_det_once/last-3.pt"
    
    # Zaženi procesor
    processor = ImageProcessor(image_dir, model_path) 