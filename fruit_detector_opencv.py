#!/usr/bin/env python3
import cv2
import numpy as np

class FruitDetector:
    def __init__(self):
        # HSV barvne meje za različno sadje
        self.color_ranges = {
            'pomaranca': {'lower': np.array([5, 100, 100]), 'upper': np.array([15, 255, 255])},
            'limona': {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])},
            'banana': {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])},
            'jabolko': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
            'grozdje': {'lower': np.array([130, 50, 50]), 'upper': np.array([170, 255, 255])}
        }
        
        # Barve za prikaz imen sadja
        self.colors = {
            'pomaranca': (0, 165, 255),  # BGR format
            'limona': (0, 255, 255),
            'banana': (0, 255, 255),
            'jabolko': (0, 0, 255),
            'grozdje': (255, 0, 255)
        }

    def detect_fruits(self, frame):
        # Pretvorba v HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Kopija originalne slike za risanje
        output = frame.copy()
        
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
                            self.draw_detection(output, contour, fruit)
                    elif fruit == 'limona':
                        # Preverjanje okrogle oblike
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.8:  # Limona je bolj okrogla
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
    # Inicializacija kamere
    cap = cv2.VideoCapture(0)  # Uporabite 0 za privzeto kamero
    
    # Inicializacija detektorja
    detector = FruitDetector()
    
    print("Začenjam detekcijo sadja...")
    print("Pritisni 'q' za izhod")
    print("Pritisni 'f' za preklop med fullscreen in normalnim načinom")
    
    fullscreen = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Napaka pri branju kamere")
            break
            
        # Detekcija sadja
        output = detector.detect_fruits(frame)
        
        # Prikaz rezultatov
        if fullscreen:
            cv2.namedWindow('Detektor Sadja', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Detektor Sadja', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Detektor Sadja', output)
        
        # Preverjanje tipk
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            fullscreen = not fullscreen
            
    # Čiščenje
    cap.release()
    cv2.destroyAllWindows()
    print("Program zaključen.")

if __name__ == '__main__':
    main() 