import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

model = load_model('model.h5')

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['circle', 'kite', 'parallelogram', 'rectangle', 'rhombus', 'square', 'trapezoid', 'triangle'])

def preprocess_image(image_path, target_size=(128, 128)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image could not be loaded.")
    
    img = cv2.resize(img, target_size)
    img = img.reshape((1, 128, 128, 1)) / 255.0  
    return img

def predict_image(model, image_path, label_encoder):
    try:
        img = preprocess_image(image_path)
        prediction = model.predict(img)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
        return predicted_class[0]
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def display_image_with_prediction(image_path, predicted_class):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load image for display.")
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f'Predicted: {predicted_class}', (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow(f'{predicted_class}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

user_image_path = '../problems/occlusion2_sol_rec.png'  

predicted_class = predict_image(model, user_image_path, label_encoder)
if predicted_class:
    print(f'Predicted Class: {predicted_class}')
    display_image_with_prediction(user_image_path, predicted_class)
else:
    print("Prediction failed.")
