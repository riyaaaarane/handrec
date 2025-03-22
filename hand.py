import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("asl_model (1).h5")

# Define class names (A-Z)
class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess image for prediction
    img = cv2.resize(frame, (100, 100))  # Resize to match model input shape
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict the ASL letter
    prediction = model.predict(img)

    # Debugging prints
    print("Raw Predictions:", prediction)  # Print raw model output
    predicted_class = np.argmax(prediction)
    print("Predicted Class Index:", predicted_class)
    print("Class Names:", class_names)  # Ensure class names are correctly mapped

    # Get the predicted letter
    predicted_letter = class_names[predicted_class]
    print("Predicted Letter:", predicted_letter)

    # Display prediction on the video feed
    cv2.putText(frame, f"Prediction: {predicted_letter}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
