#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Inicializacija ROS node-a
rospy.init_node('color_tuner_detection', anonymous=True)
bridge = CvBridge()

# Globalne spremenljivke
latest_color_image = None
latest_depth_image = None

# Barve za detekcijo
colors = {
    'rumena': {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])},
    'rdeča': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
    'oranžna': {'lower': np.array([5, 100, 100]), 'upper': np.array([15, 255, 255])},
    'vijolična': {'lower': np.array([130, 100, 100]), 'upper': np.array([170, 255, 255])}
}

def color_callback(msg):
    global latest_color_image
    latest_color_image = bridge.imgmsg_to_cv2(msg, "bgr8")

def depth_callback(msg):
    global latest_depth_image
    latest_depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

# Prijava na ROS teme
rospy.Subscriber("/camera/color/image_raw", Image, color_callback)
rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_callback)

def get_depth(depth_image, x, y, kernel_size=5):
    """Pridobi povprečno globino iz območja okoli točke (x,y)"""
    if depth_image is None:
        return 0
    half_kernel = kernel_size // 2
    height, width = depth_image.shape
    
    x_start = max(0, x - half_kernel)
    x_end = min(width, x + half_kernel + 1)
    y_start = max(0, y - half_kernel)
    y_end = min(height, y + half_kernel + 1)
    
    region = depth_image[y_start:y_end, x_start:x_end]
    valid_depths = region[(region > 0) & (region < 10000)]
    
    if len(valid_depths) > 0:
        return float(np.median(valid_depths))
    return 0

def nothing(x):
    pass

def create_trackbars():
    """Ustvari drsnike za vsako barvo"""
    cv2.namedWindow('Nastavitve barv')
    
    for color in colors:
        # H drsniki
        cv2.createTrackbar(f'{color}_H_min', 'Nastavitve barv', colors[color]['lower'][0], 180, nothing)
        cv2.createTrackbar(f'{color}_H_max', 'Nastavitve barv', colors[color]['upper'][0], 180, nothing)
        # S drsniki
        cv2.createTrackbar(f'{color}_S_min', 'Nastavitve barv', colors[color]['lower'][1], 255, nothing)
        cv2.createTrackbar(f'{color}_S_max', 'Nastavitve barv', colors[color]['upper'][1], 255, nothing)
        # V drsniki
        cv2.createTrackbar(f'{color}_V_min', 'Nastavitve barv', colors[color]['lower'][2], 255, nothing)
        cv2.createTrackbar(f'{color}_V_max', 'Nastavitve barv', colors[color]['upper'][2], 255, nothing)

def update_color_ranges():
    """Posodobi območja barv glede na drsnike"""
    for color in colors:
        colors[color]['lower'] = np.array([
            cv2.getTrackbarPos(f'{color}_H_min', 'Nastavitve barv'),
            cv2.getTrackbarPos(f'{color}_S_min', 'Nastavitve barv'),
            cv2.getTrackbarPos(f'{color}_V_min', 'Nastavitve barv')
        ])
        colors[color]['upper'] = np.array([
            cv2.getTrackbarPos(f'{color}_H_max', 'Nastavitve barv'),
            cv2.getTrackbarPos(f'{color}_S_max', 'Nastavitve barv'),
            cv2.getTrackbarPos(f'{color}_V_max', 'Nastavitve barv')
        ])

def process_frame(frame, depth_image):
    """Obdelaj sliko in prikaži rezultate"""
    if frame is None or depth_image is None:
        return None
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    result = frame.copy()
    
    # Obdelaj vsako barvo
    for color_name, color_range in colors.items():
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
                    
                    # Pridobi globino
                    depth = get_depth(depth_image, cx, cy)
                    
                    # Nariši konturo in informacije
                    cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)
                    cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # Prikaži informacije
                    text = f'{color_name}: {depth:.0f}mm'
                    cv2.putText(result, text, (cx - 10, cy - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Izpiši v konzolo
                    print(f"\nZaznana barva: {color_name}")
                    print(f"Pozicija (x,y): ({cx}, {cy})")
                    print(f"Razdalja: {depth:.0f}mm")
        
        # Prikaži masko
        cv2.imshow(f'Mask {color_name}', mask)
    
    return result

def main():
    create_trackbars()
    
    print("Začenjam detekcijo barv...")
    print("Uporabi drsnike za nastavljanje HSV vrednosti")
    print("Pritisni 'q' za izhod")
    
    try:
        while not rospy.is_shutdown():
            if latest_color_image is not None and latest_depth_image is not None:
                # Posodobi območja barv
                update_color_ranges()
                
                # Obdelaj sliko
                result = process_frame(latest_color_image.copy(), latest_depth_image.copy())
                
                if result is not None:
                    # Prikaži rezultat
                    cv2.imshow('Color Detection', result)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            rospy.sleep(0.05)
            
    except Exception as e:
        print(f"Napaka: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 