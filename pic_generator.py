import numpy as np
import cv2
import os

def generate_custom_shapes_dataset(num_samples_per_class=500, output_dir='../dataset_created'):
    shapes = ['line', 'circle', 'ellipse', 'rectangle', 'rounded_rectangle', 'regular_polygon', 'star']
    
    os.makedirs(output_dir, exist_ok=True)
    
    for shape in shapes:
        shape_dir = os.path.join(output_dir, shape)
        os.makedirs(shape_dir, exist_ok=True)
        
        for i in range(num_samples_per_class):
            img = np.zeros((128, 128), dtype=np.uint8)
            
            if shape == 'line':
                pt1 = tuple(np.random.randint(0, 128, size=2))
                pt2 = tuple(np.random.randint(0, 128, size=2))
                cv2.line(img, pt1, pt2, 255, 2)
            
            elif shape == 'circle':
                radius = np.random.randint(10, 40)
                center = tuple(np.random.randint(radius, 128-radius, size=2))
                cv2.circle(img, center, radius, 255, 2)
            
            elif shape == 'ellipse':
                center = tuple(np.random.randint(20, 108, size=2))
                axes = tuple(np.random.randint(10, 50, size=2))
                angle = np.random.randint(0, 180)
                cv2.ellipse(img, center, axes, angle, 0, 360, 255, 2)
            
            elif shape == 'rectangle':
                pt1 = tuple(np.random.randint(0, 108, size=2))
                pt2 = (pt1[0] + np.random.randint(20, 40), pt1[1] + np.random.randint(20, 40))
                cv2.rectangle(img, pt1, pt2, 255, 2)
            
            elif shape == 'rounded_rectangle':
                pt1 = tuple(np.random.randint(0, 108, size=2))
                pt2 = (pt1[0] + np.random.randint(20, 40), pt1[1] + np.random.randint(20, 40))
                radius = np.random.randint(5, 15)
                cv2.rectangle(img, pt1, pt2, 255, 2)
                corners = [pt1, pt2, (pt1[0], pt2[1]), (pt2[0], pt1[1])]
                for corner in corners:
                    cv2.circle(img, corner, radius, 255, 2)
            
            elif shape == 'regular_polygon':
                points = np.random.randint(0, 128, size=(5, 2))
                cv2.polylines(img, [points], isClosed=True, color=255, thickness=2)
            
            elif shape == 'star':
                points = np.random.randint(0, 128, size=(5, 2))
                cv2.polylines(img, [points], isClosed=True, color=255, thickness=2)
            
            img_path = os.path.join(shape_dir, f'{i}.png')
            cv2.imwrite(img_path, img)

generate_custom_shapes_dataset()
