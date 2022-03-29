#importing libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import img_to_array
import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase,RTCConfiguration

#page title
st.set_page_config(page_title="Emotion Detection")
#load model
model = tf.keras.models.load_model("emotion_detection.h5")
#face detection classifier
try:
    face_haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        #image gray
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #detect faces
        faces = face_haar_cascade.detectMultiScale(image=gray_image)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x+w, y+h), color=(255, 0, 0), thickness=2)
            roi_gray = gray_image[y-5:y+h+5,x-5:x+w+5]
            roi_gray=cv2.resize(roi_gray,(48,48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis = 0)
            #normalize
            image_pixels /= 255
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            #map predictions
            emotion_detection = ('You seem Angry.', 'You seem Disgusted.', 'Fear Detected!!', "Yayy, You seem Happy.", 'You seem Sad.', 'Surprised!!!', 'You seem Neutral.')
            emotion_prediction = emotion_detection[max_index]
            font = cv2.FONT_HERSHEY_SIMPLEX
            lable_color = (0, 255, 0)
            cv2.putText(img,emotion_prediction, (int(x),int(y)),font,0.9, lable_color,2)
        return img

def main():
    # Application
    pages = ["Home","About"]
    with st.sidebar:
        st.title('Page Selection')
        page_name = st.selectbox("Select Page:", pages)
    st.title(page_name)

    if page_name == 'Home':
        home_html = """<body>
                    <h4 style="font-size:30px">Real Time Emotion Detection</h4>
                    <p>The application detects faces and predicts the face emotion using OpenCV and a customized CNN model trained on FER2013 dataset.</p>
                    </body>"""
        st.markdown(home_html,unsafe_allow_html=True)
        st.write("Click on start to use a webcam and detect your Facial Emotion.")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer,media_stream_constraints={
            "video": True,
            "audio": False
        },rtc_configuration=RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
))
    
    elif page_name == "About":
        about_html = """<body>
                       <h4 style="font-size:30px">Real time Face Emotion Detection Application</h4>
                                    <body>"""
        st.markdown(about_html, unsafe_allow_html=True)
        st.write("The project is part of the curriculum of AlmaBetter's Full Stack Data Science Program.")
        statement_html = '''<body>
        <h4 style="font-size:20px">Project Statement</h4>
        <p>One of many challenges is how to ensure quality learning for students. Digital platforms might overpower physical classrooms in terms of content quality but when it comes to understanding whether students are able to grasp the 
        content in a live class scenario is yet an open-end challenge. While digital platforms have limitations in terms of physical surveillance but it comes with the power of data and machines which can work for you. It provides data in the form of video, audio, and texts which can be analysed using deep learning algorithms. Deep learning backed system not only solves the surveillance issue, but it also removes the human bias from the system, and all information is no longer in the teacher’s 
        brain rather translated in numbers that can be analysed and tracked. The solution to this problem is by recognizing facial emotions. This is a live face emotion detection system. The model is able to real-time identify the emotions of students in a live class.
        The demo application is developed by Vithika Karan using OpenCV-Python, CNN and Streamlit Frameworks.<p>
         <body>'''
        st.markdown(statement_html,True)
        st.write("Contact Details:")
        st.write("Vithika Karan")
        st.write("LinkedIn Profile: https://www.linkedin.com/in/vithika-karan/")

    else:
        pass

if __name__ == "__main__":
    main()




