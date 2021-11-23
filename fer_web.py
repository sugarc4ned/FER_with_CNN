import cv2
import numpy as np
import streamlit as st
from keras.models import model_from_json
import tempfile


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
# load json and create model
json_file = open(r'model\50emotion_model.json', 'r') #change directory of model
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights(r"model\50emotion_model.h5") #change directory of model
print("Loaded model from disk")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default')

def prediction():
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#Main
st.title("Face Expression Recognition with Deeplearning")
st.text("Team: Kum Yu Kit and Yow Chin Choong")

page = st.selectbox("Choose your Mode", ["Run on Image", "Run on Video", "Real-time Camera"]) 

if page == "Run on Image":
    image = st.file_uploader("Choose a image file", type="jpg")
    
    if image is not None:
        st.image(image, caption='Uploaded Image')
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        frame = cv2.resize(frame, (1280, 720))
        prediction()
        img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(img2, caption='Detected Image')

elif page == "Run on Video":
    FRAME_WINDOW = st.image([])
    video_file = st.file_uploader('Choose a video file', type = 'mp4')
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1280, 720))
            if not ret:
                break
            prediction()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

elif page == "Real-time Camera":
    run = st.checkbox('Use Webcam')
    FRAME_WINDOW = st.image([])
    while run:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1280, 720))
            if not ret:
                break
            prediction()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
    else:
        st.write('Camera Stopped')





