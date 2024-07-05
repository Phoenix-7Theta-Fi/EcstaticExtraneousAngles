import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
import tempfile
import urllib.request

# Download and save the model
@st.cache_resource
def load_model():
    model_url = "https://storage.googleapis.com/tm-model/bBgPjLiYN/model.tflite"
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tflite') as fp:
        urllib.request.urlretrieve(model_url, fp.name)
        model_path = fp.name
    return model_path

model_path = load_model()

# Load the TensorFlow Lite model
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize variables for rep counting
rep_count = 0
previous_state = None
current_state = None

# Set up Streamlit app
st.title("AI-powered Workout App")
st.write("Dumbbell Shoulder Press Rep Counter")

# Create a placeholder for the webcam feed
video_placeholder = st.empty()

# Create a placeholder for the rep count
count_placeholder = st.empty()

# Streamlit doesn't support direct webcam access, so we'll use a file uploader for video
video_file = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])

if video_file is not None:
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())

    # Open the video file
    cap = cv2.VideoCapture(tfile.name)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        resized_frame = cv2.resize(frame, (200, 200))
        input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get the predicted class
        predicted_class = np.argmax(output_data)

        # Update rep count
        if predicted_class == 0:  # Assuming 0 is for "Extension"
            current_state = "Extension"
        else:
            current_state = "Contraction"

        if previous_state == "Extension" and current_state == "Contraction":
            rep_count += 1

        previous_state = current_state

        # Display the frame
        video_placeholder.image(frame, channels="BGR")

        # Update the rep count
        count_placeholder.markdown(f"## Rep Count: {rep_count}")

        # Add a small delay to control the video playback speed
        cv2.waitKey(25)

    # Release the video capture object
    cap.release()

else:
    st.write("Please upload a video file to begin.")