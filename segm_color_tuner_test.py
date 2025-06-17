#!/usr/bin/env python3
import cv2
import numpy as np
import time

# Barve za detekcijo
colors = {
    'rumena': {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])},
    'rdeča': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
    'oranžna': {'lower': np.array([5, 100, 100]), 'upper': np.array([15, 255, 255])},
    'vijolična': {'lower': np.array([130, 100, 100]), 'upper': np.array([170, 255, 255])}
}

def nothing(x):
    pass

def create_windows():
    """Ustvari okna za vsako barvo"""
    # Ustvari eno okno za nastavitve
    cv2.namedWindow('Nastavitve', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Nastavitve', 400, 800)
    
    # Ustvari okna za detekcijo in maske
    for color in colors:
        cv2.namedWindow(f'Detekcija {color}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'Detekcija {color}', 800, 600)
        
        cv2.namedWindow(f'Mask {color}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'Mask {color}', 800, 600)
        
        # Počakaj, da se okna ustvarijo
        cv2.waitKey(1)

def create_trackbars():
    """Ustvari drsnike za vsako barvo v enem oknu"""
    for color in colors:
        # H drsniki
        cv2.createTrackbar(f'{color}_H_min', 'Nastavitve', colors[color]['lower'][0], 180, nothing)
        cv2.createTrackbar(f'{color}_H_max', 'Nastavitve', colors[color]['upper'][0], 180, nothing)
        # S drsniki
        cv2.createTrackbar(f'{color}_S_min', 'Nastavitve', colors[color]['lower'][1], 255, nothing)
        cv2.createTrackbar(f'{color}_S_max', 'Nastavitve', colors[color]['upper'][1], 255, nothing)
        # V drsniki
        cv2.createTrackbar(f'{color}_V_min', 'Nastavitve', colors[color]['lower'][2], 255, nothing)
        cv2.createTrackbar(f'{color}_V_max', 'Nastavitve', colors[color]['upper'][2], 255, nothing)
        
        # Počakaj, da se drsniki ustvarijo
        cv2.waitKey(1)

def update_color_ranges():
    """Posodobi območja barv glede na drsnike"""
    for color in colors:
        colors[color]['lower'] = np.array([
            cv2.getTrackbarPos(f'{color}_H_min', 'Nastavitve'),
            cv2.getTrackbarPos(f'{color}_S_min', 'Nastavitve'),
            cv2.getTrackbarPos(f'{color}_V_min', 'Nastavitve')
        ])
        colors[color]['upper'] = np.array([
            cv2.getTrackbarPos(f'{color}_H_max', 'Nastavitve'),
            cv2.getTrackbarPos(f'{color}_S_max', 'Nastavitve'),
            cv2.getTrackbarPos(f'{color}_V_max', 'Nastavitve')
        ])

def process_frame(frame, color_name, color_range):
    """Obdelaj sliko za določeno barvo"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    result = frame.copy()
    
    # Ustvari masko
    mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
    
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
                
                # Nariši konturo in informacije
                cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)
                cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
                
                # Prikaži informacije
                text = f'{color_name} ({cx}, {cy})'
                cv2.putText(result, text, (cx - 10, cy - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result, mask

def main():
    try:
        # Inicializacija kamere
        cap = cv2.VideoCapture(0)  # 0 za laptop kamero
        
        if not cap.isOpened():
            print("Napaka: Ne morem odpreti kamere!")
            return
        
        # Ustvari okna
        create_windows()
        time.sleep(1)  # Počakaj, da se okna ustvarijo
        
        # Ustvari drsnike
        create_trackbars()
        time.sleep(1)  # Počakaj, da se drsniki ustvarijo
        
        print("Začenjam detekcijo barv...")
        print("Vse nastavitve so v enem oknu")
        print("Pritisni 'q' za izhod")
        print("Pritisni 'f' za preklop med fullscreen in normalnim načinom")
        
        while True:
            # Preberi sliko iz kamere
            ret, frame = cap.read()
            if not ret:
                print("Napaka: Ne morem prebrati slike iz kamere!")
                break
            
            # Posodobi območja barv
            update_color_ranges()
            
            # Obdelaj vsako barvo posebej
            for color_name, color_range in colors.items():
                result, mask = process_frame(frame.copy(), color_name, color_range)
                
                # Prikaži rezultate
                cv2.imshow(f'Detekcija {color_name}', result)
                cv2.imshow(f'Mask {color_name}', mask)
            
            # Preveri za izhod in preklop fullscreen
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                cv2.setWindowProperty('Nastavitve', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                for color in colors:
                    cv2.setWindowProperty(f'Detekcija {color}', cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(f'Mask {color}', cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_NORMAL)
            
    except Exception as e:
        print(f"Napaka: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("Program zaključen.")

if __name__ == "__main__":
    main() 