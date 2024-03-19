from flask import Flask, render_template, request
import json
import math
from flask_cors import CORS
import numpy as np
import cv2
import mediapipe as mp                              # Library for image processing
from math import floor
import imutils
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('shirt.html')
@app.route('/shirt.html')
def plot():
    return render_template('shirt.html')
@app.route('/pant.html')
def ploty():
    return render_template('pant.html')

#######################################################
def mdpt(A, B):
    return ((A[0] + B[0]) * 0.5, (A[1] + B[1]) * 0.5) 

@app.route('/measurement', methods=['GET','POST'])
def measurement():
    cap = cv2.VideoCapture(0)

# Set mediapipe pose model
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Get reference image
    ref_image = cv2.imread("shirt2.png")
    ref_image_height, ref_image_width, _ = ref_image.shape

    # Calculate reference shoulder width
    mp_holistic = mp.solutions.holistic
    mp_drawing_styles = mp.solutions.drawing_styles
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        results = holistic.process(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))
        ref_l_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        ref_r_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        ref_shoulder_width = math.dist([ref_l_shoulder.x*ref_image_width, ref_l_shoulder.y*ref_image_height],
                                    [ref_r_shoulder.x*ref_image_width, ref_r_shoulder.y*ref_image_height])

    # Loop over frames from the video stream
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        # Flip the frame
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect the pose from the frame
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results = pose.process(frame)
            
            # Draw pose landmarks on the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # Calculate the shoulder width of the user
                l_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                shoulder_width = math.dist([l_shoulder.x*frame.shape[1], l_shoulder.y*frame.shape[0]],
                                        [r_shoulder.x*frame.shape[1], r_shoulder.y*frame.shape[0]])
                
                # Calculate the shirt size of the user based on the shoulder width
                shirt_size = "Not Determined"
                if shoulder_width < ref_shoulder_width * 0.9:
                    shirt_size = "S"
                elif shoulder_width < ref_shoulder_width * 1.1:
                    shirt_size = "M"
                elif shoulder_width < ref_shoulder_width * 1.3:
                    shirt_size = "L"
                else:
                    shirt_size = "XL"
                
                # Put the shirt size text on the frame
                cv2.putText(frame, "Shirt Size: " + shirt_size, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

    # Check for the 'q' key to exit
        if cv2.waitKey(1) == ord('q'):
            break

# Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    return render_template('index.html')
#########################################################
@app.route('/predict', methods=['GET','POST'])
def predict():
    shirtno = int(request.form["shirt"])
    # pantno = int(request.form["pant"])

    cv2.waitKey(1)
    cap=cv2.VideoCapture(0)
    ih=shirtno
    # i=pantno

    while True:
        imgarr=['shirt2.png','shirt6.png','10.png','shirtred.png']

        #ih=input("Enter the shirt number you want to try")
        imgshirt = cv2.imread(imgarr[ih-1],1) #original img in bgr
        if ih==3:
            shirtgray = cv2.cvtColor(imgshirt,cv2.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks_inv = cv2.threshold(shirtgray,200 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks = cv2.bitwise_not(orig_masks_inv)
        else:
            shirtgray = cv2.cvtColor(imgshirt,cv2.COLOR_BGR2GRAY) #grayscale conversion
            ret, orig_masks = cv2.threshold(shirtgray,0 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
            orig_masks_inv = cv2.bitwise_not(orig_masks)
        origshirtHeight, origshirtWidth = imgshirt.shape[:2]
        # imgarr=["pant7.jpg",'pant21.png','skirt.jpg']
        # i=input("Enter the pant number you want to try")
        # imgpant = cv2.imread(imgarr[i-1],1)
        # imgpant=imgpant[:,:,0:4]#original img in bgr
        # pantgray = cv2.cvtColor(imgpant,cv2.COLOR_BGR2GRAY) #grayscale conversion
        # if i==1:
        #     ret, orig_mask = cv2.threshold(pantgray,100 , 255, cv2.THRESH_BINARY) #there may be some issues with image threshold...depending on the color/contrast of image
        #     orig_mask_inv = cv2.bitwise_not(orig_mask)
        # else:
        #     ret, orig_mask = cv2.threshold(pantgray,50 , 255, cv2.THRESH_BINARY)
        #     orig_mask_inv = cv2.bitwise_not(orig_mask)
        # origpantHeight, origpantWidth = imgpant.shape[:2]
        face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        ret,img=cap.read()

        height = img.shape[0]
        width = img.shape[1]
        resizewidth = int(width*3/2)
        resizeheight = int(height*3/2)
        cv2.namedWindow("img",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img", (int(width*3/2), int(height*3/2)))
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img,(100,200),(312,559),(255,255,255),2)

            # Shirt part
            shirtWidth =  3 * w  #approx wrt face width
            shirtHeight = shirtWidth * origshirtHeight / origshirtWidth #preserving aspect ratio of original image..
            x1s = x-w
            x2s =x1s+3*w
            y1s = y+h
            y2s = y1s+h*4

            if x1s < 0:
                x1s = 0
            if x2s > img.shape[1]:
                x2s =img.shape[1]
            if y2s > img.shape[0] :
                y2s =img.shape[0]
            temp=0
            if y1s>y2s:
                temp=y1s
                y1s=y2s
                y2s=temp

            shirtWidth = int(abs(x2s - x1s))
            shirtHeight = int(abs(y2s - y1s))
            y1s = int(y1s)
            y2s = int(y2s)
            x1s = int(x1s)
            x2s = int(x2s)

            shirt = cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(orig_masks, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
            masks_inv = cv2.resize(orig_masks_inv, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
            rois = img[y1s:y2s, x1s:x2s]
            num=rois
            roi_bgs = cv2.bitwise_and(rois,num,mask = masks_inv)
            roi_fgs = cv2.bitwise_and(shirt,shirt,mask = mask)
            dsts = cv2.add(roi_bgs,roi_fgs)
            img[y1s:y2s, x1s:x2s] = dsts

        cv2.imshow("img1",img)

        if cv2.waitKey(100) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return render_template('shirt.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=5000)