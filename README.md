# **Face-Emotion-Recognition**
### **Real Time Emotion Detection: Predicting emotions through a live webcam feed.**
Face Emotion Recognition is a CNN project. This project is a part of Full Stack Data Science curriculum at [AlmaBetter](https://www.almabetter.com/).

### **Project Statement**
The Indian education landscape has been undergoing rapid changes for the past 10 years owing to the advancement of web-based learning services, specifically, eLearning platforms. Global E-learning is estimated to witness an 8X over the next 5 years to reach USD 2B in 2021. India is expected to grow with a CAGR of 44% crossing the 10M users mark in 2021. Although the market is growing on a rapid scale, there are major challenges associated with digital learning when compared with brick and mortar classrooms. One of many challenges is how to ensure quality learning for students. Digital platforms might overpower physical classrooms in terms of content quality but when it comes to understanding whether students are able to grasp the content in a live class scenario is yet an open-end challenge. In a physical classroom during a lecturing teacher can see the faces and assess the emotion of the class and tune their lecture accordingly, whether he is going fast or slow. He can identify students who need special attention. Digital classrooms are conducted via video telephony software program (exZoom) where it’s not possible for medium scale class (25-50) to see all students and access the mood. Because of this drawback, students are not focusing on content due to lack of surveillance. While digital platforms have limitations in terms of physical surveillance but it comes with the power of data and machines which can work for you. It provides data in the form of video, audio, and texts which can be analysed using deep learning algorithms. Deep learning backed system not only solves the surveillance issue, but it also removes the human bias from the system, and all information is no longer in the teacher’s brain rather translated in numbers that can be analysed and tracked. 
The solution to this problem is by recognizing facial emotions. This is a live face emotion detection system. The model is able to real-time identify the emotions of students in a live class. The demo application is developed using OpenCV-Python, CNN and Streamlit Frameworks.

### **FER2013 Dataset**
The dataset used in training the CNN model for the above mentioned problem statement is FER2013 and can be downloaded from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) or Data folder of the repository. The data consists of grayscale images of faces at a resolution of 48x48 pixels. The faces have been automatically registered such that they are more or less centred in each image and take up around the same amount of area. The seven categories based on the emotion expressed in the facial expression are (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). There are 28,709 examples in the training set and 3,589 examples in the public test set.

### **Technologies**
* Pandas
* Numpy
* Matplotlib
* Seaborn
* PIL
* Keras 
* Tensorflow
* OpenCV-Python
* Streamlit

### **Approach**
* Exploratory Data Analysis
* Data Preprocessing
* Data Augmentation
* CNN Model Building
* Model Performance and Evaluation
* Real Time Emotion Detection
* Streamlit Web Application
* Web App Deployment on Heroku and AWS ec2 


### **Run Real Time Emotion Detection Locally**
* Create a project folder
* Create a virtual environment
* Download real_time_demo.py, emotion_detection.h5 and haarcascade_frontalface_default.xml in the same folder.
* Install the dependencies from requirements.txt 
(Note: Install opencv-python as well, opencv-python-headless in the requirements file is to adjust the size requirements for web app deployment purposes. To run app.py requirements.txt is enough.)
* Run 

### **Web App Deployment**
* Heroku URL - https://face-emo-recog1.herokuapp.com/
* AWS ec2 instance 
(streamlit external url) - http://54.84.63.103:8501/
(Note: Since Google Chrome doesnt allow media access from unsecure origins,to access the app, it is needed to exclude the link. Enable camera for insecure origins in chrome from ‘chrome://flags/#unsafely-treat-insecure-origin-as-secure’ by putting in the link to access the app)


