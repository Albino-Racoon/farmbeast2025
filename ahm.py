import os
import shutil
import yaml
from pathlib import Path

def organize_dataset(images_dir, labels_yaml, output_dir):
    # Ustvari izhodne mape za vsak razred
    with open(labels_yaml, 'r') as f:
        labels = yaml.safe_load(f)
    
    # Ustvari izhodno mapo
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Ustvari mape za vsak razred
    for class_name in labels['names']:
        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)
    
    # Poišči vse slike
    image_extensions = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]
    
    # Razvrstitev slik
    for image in images:
        image_path = os.path.join(images_dir, image)
        label_path = os.path.join(images_dir, image.rsplit('.', 1)[0] + '.txt')
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                # Preberi prvo vrstico, ki vsebuje razred
                first_line = f.readline().strip()
                if first_line:
                    class_id = int(first_line.split()[0])
                    class_name = labels['names'][class_id]
                    
                    # Kopiraj sliko v ustrezno mapo
                    dest_path = output_dir / class_name / image
                    shutil.copy2(image_path, dest_path)
                    print(f"Kopirano: {image} -> {class_name}/")

if __name__ == "__main__":
    # Nastavitve
    images_dir = "/home/rakun/Desktop/train/images"  # Mapo s slikami
    labels_yaml = "/home/rakun/Desktop/data.yaml"  # YAML datoteka z oznakami
    output_dir = "/home/rakun/Desktop/train/output"  # Izhodna mapa
    
    organize_dataset(images_dir, labels_yaml, output_dir)