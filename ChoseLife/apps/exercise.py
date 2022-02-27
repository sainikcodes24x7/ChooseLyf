import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import json
import requests

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

##Function for Situp
def calculate_dist(a,b):
    a = np.array(a) 
    b = np.array(b) 
    dist = np.linalg.norm(a - b)   
    return dist 
##Function for Pushup
def calculate_space(a,b,c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    mid = (c[1]-b[1])/2
    diff = a[1]-mid   
    return diff  

##Function for Left and Right Hand
def calculate_angle(a,b,c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle        

def app():
    
    st.title('Exercise')
    st.subheader("Lets start some exercises!!")
    st.write('Navigate through the type of exercise you want to do and check your calories burnt using our webpage and navigation bar present on the left of the page.')
    rad = st.sidebar.radio("Do exercise",["Situps","Pushups","Left Hand Exercise","Right Hand Exercise"])

##Situps##    
    if rad == "Situps":
        lottie_situp = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_ocGoFt.json")
        st_lottie(lottie_situp)
        run = st.checkbox("Run")
        FRAME_WINDOW =st.image([])
        cam = cv2.VideoCapture(0)
        counter = 0 
        stage = None
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while run:
                ret,frame=cam.read()
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                results = pose.process(img)
                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                try:
                    landmarks = results.pose_landmarks.landmark
                    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                    Rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    space = calculate_dist(nose, Rknee)
                    cv2.putText(img, str(space), tuple(np.multiply(nose, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    if space > 0.7:
                        stage = "down"
                    if space < 0.4 and stage =='down':
                        stage="up"
                        counter +=1
                except:
                    pass
                
                cv2.rectangle(img, (0,0), (230,73), (245,117,16), -1)
                cv2.putText(img, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(img, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(img, 'CALORIES BURNT', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(img, str(counter*0.45), (100,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )
                cv2.imshow('Mediapipe Feed', img)
                FRAME_WINDOW.image(img)        
        
            st.balloons()                      
            cam.release()
            cv2.destroyAllWindows()
                                            
##Pushups##
    if rad == "Pushups":
        lottie_pushup = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_fhtmgptq.json")
        st_lottie(lottie_pushup)
        run = st.checkbox("Run")
        FRAME_WINDOW =st.image([])
        cam = cv2.VideoCapture(0)
        counter = 0
        stage = None
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while run:
                ret,frame=cam.read()
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                results = pose.process(img)
                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                try:
                    landmarks = results.pose_landmarks.landmark
                    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                    Rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    Lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    space = calculate_space(nose, Lwrist, Rwrist)
                    cv2.putText(img, str(space), tuple(np.multiply(nose, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    if space > 0.7:
                        stage = "down"
                    if space < 0.4 and stage =='down':
                        stage="up"
                        counter +=1    
                except:
                    pass
                
                cv2.rectangle(img, (0,0), (230,73), (245,117,16), -1)
                cv2.putText(img, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(img, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(img, 'CALORIES BURNT', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(img, str(counter*0.5), (100,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )
                cv2.imshow('Mediapipe Feed', img)
                FRAME_WINDOW.image(img)
                       
        st.balloons()
        cam.release()
        cv2.destroyAllWindows()    
                      
##LeftHand    
    if rad == "Left Hand Exercise":
        lottie_left = load_lottieurl("https://assets1.lottiefiles.com/private_files/lf30_i5o0xxk6.json")
        st_lottie(lottie_left)
        run = st.checkbox("Run")
        FRAME_WINDOW =st.image([])
        cap = cv2.VideoCapture(0)
        counter = 0 
        stage = None
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while run:
                ret, frame = cap.read()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
                    angle = calculate_angle(shoulder, elbow, wrist)
                    
                    cv2.putText(image, str(angle), 
                                tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    if angle > 160:
                        stage = "down"
                    if angle < 30 and stage =='down':
                        stage="up"
                        counter +=1                            
                except:
                    pass
                
                cv2.rectangle(image, (0,0), (350,73), (245,117,16), -1)
                
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, 'CALORIES BURNT', (205,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter*0.5), (210,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)            
                
                cv2.putText(image, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )               
                
                cv2.imshow('Mediapipe Feed', image)
                FRAME_WINDOW.image(image) 
        
        st.balloons()
        cap.release()
        cv2.destroyAllWindows()
    
##RightHand    
    if rad == "Right Hand Exercise":
        lottie_right = load_lottieurl("https://assets1.lottiefiles.com/private_files/lf30_i5o0xxk6.json")
        st_lottie(lottie_right)
        run = st.checkbox("Run")
        FRAME_WINDOW =st.image([])
        cap = cv2.VideoCapture(0)
        counter = 0 
        stage = None
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while run:
                ret, frame = cap.read()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    
                    angle = calculate_angle(shoulder, elbow, wrist)
                    
                    cv2.putText(image, str(angle), 
                                tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    if angle > 160:
                        stage = "down"
                    if angle < 30 and stage =='down':
                        stage="up"
                        counter +=1                            
                except:
                    pass
                
                cv2.rectangle(image, (0,0), (350,73), (245,117,16), -1)
                
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, 'CALORIES BURNT', (205,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter*0.5), (210,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)            
                
                cv2.putText(image, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )               
                
                cv2.imshow('Mediapipe Feed', image)
                FRAME_WINDOW.image(image) 

        st.balloons()
        cap.release()
        cv2.destroyAllWindows()            

