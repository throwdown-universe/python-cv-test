import cv2
import time
# import pandas
import numpy as np
import math
from collections import deque
from operator import itemgetter

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose





cap = cv2.VideoCapture("media/dbsnatch.mov")
# cap = cv2.VideoCapture("media/DB_Snatch_Workout.MP4")

#@title
keypoints = []

# how many landmarks to buffer for detecting high- and lowpoints
bufferLen=10

# frames buffer
fb = deque(maxlen=bufferLen)


def max_val(l, i):
    '''Returns the maximum value of a list of tuples, based on the value at index i of the tuple.'''
    return max(enumerate(map(itemgetter(i), l)),key=itemgetter(1))

def calculate_slope(a,b):
    a = np.array(a) # First
    b = np.array(b) # Second
    # slope, intercept = np.polyfit((a[0], b[0]), (a[1], b[1]), 1)
    slope, intercept = np.polyfit(a,b, 1)
    return slope

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


def calculate_angle_2 (a, b):
    a = np.array(a) # First
    b = np.array(b) # End
    x_diff = b[0] - a[0]
    y_diff = b[1] - a[1]
    return math.degrees(math.atan2(y_diff, x_diff))



def calculate_distance(a, b):
    """ Euclidean distance between two points a, b """
    diff_a = (a[0] - b[0]) ** 2
    diff_b = (a[1] - b[1]) ** 2
    return (diff_a + diff_b) ** 0.5




frame = 0;



leftArmCounter=0
rightArmCounter=0

prevFrameLeftArmSlope=0
prevFrameRightArmSlope=0

leftArmReset = 1
rightArmReset = 1


 
leftArmMaxSlopeAngle = 0
rightArmMaxSlopeAngle = 0

leftArmHighPositionReached = 0
rightArmHighPositionReached = 0

leftArmLowPositionReached = 0
rightArmLowPositionReached = 0
    
directionLeftArm = "unknown"
directionRightArm = "unknown"
    
# videoSaver = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('output.avi', fourcc, 25, (1280, 720))

# writing resulting video to file system
# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
videoSize = (frame_width, frame_height)
   
# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
out = cv2.VideoWriter('output.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, videoSize)
    



with mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      # continue
      break

    frame += 1


    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)


  # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark      
        leftShoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        leftElbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        leftWrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        leftAnkle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        leftEar = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
        
        rightShoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        rightElbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        rightWrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        rightAnkle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        rightEar = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
      
        # calculate angle
        leftArmBendAngle = calculate_angle(leftShoulder, leftElbow, leftWrist)
        rightArmAngle = calculate_angle(rightShoulder, rightElbow, rightWrist)
        
        # calculate slope
        # leftArmSlope = calculate_slope(leftShoulder, leftWrist)
        # rightArmSlope = calculate_slope(rightShoulder, rightWrist)
        leftArmSlope = calculate_angle_2(leftShoulder, leftWrist)
        rightArmSlope = calculate_angle_2(rightShoulder, rightWrist)
        
        # print ("LeftArmSlope: ", leftArmSlope)
      
        # Visualize angles
        # cv2.putText(image, str(leftArmBendAngle), 
        # # cv2.putText(image, str(leftArmSlope), 
        #     tuple(np.multiply(leftElbow, [640, 480]).astype(int)), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        # )
        cv2.putText(image, str(leftArmBendAngle), 
        # cv2.putText(image, str(leftArmSlope), 
            tuple(np.multiply(leftElbow, [640, 280]).astype(int)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        
        # cv2.putText(image, str(rightArmAngle), 
        # # cv2.putText(image, str(rightArmSlope), 
        #     tuple(np.multiply(rightElbow, [240, 280]).astype(int)), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        # )
        cv2.putText(image, str(rightArmAngle), 
        # cv2.putText(image, str(rightArmSlope), 
            tuple(np.multiply(rightElbow, [240, 480]).astype(int)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        )
      


        # Make an array of objects from landmakrs
        keypoints.append(landmarks)

     
        
            
            
        cv2.rectangle(image, (0,0), (150,73), (245,117,16), -1)
            
        # Print right arm reps to image
        cv2.putText(image, 'RIGHT', (15,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(rightArmCounter), 
                (10,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
        # Print left arm reps to image
        cv2.putText(image, 'LEFT', (65,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(leftArmCounter), 
                (60,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
                
    


        # print (results.pose_landmarks)
        
        #keypoints.append({
        #    'X': lm.x,
        #    'Y': lm.y,
        #    'Z': lm.z,
        #    'Visibility': results.visibility,
        #  })

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # mp_drawing.draw_landmarks(
        #    image,
        #    results.pose_landmarks,
        #    mp_pose.POSE_CONNECTIONS,
        #    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
            )               
            

   
        
        
        
        
        
        ##### detection of movement stadards #####
        
        # fill frame buffer with individual landmarks
        fb.appendleft((frame, leftShoulder, leftElbow, leftWrist, leftEar, leftArmBendAngle, rightShoulder, rightElbow, rightWrist, rightEar, rightArmAngle))
        # print ("Frame Buffer Length:", len(fb))  
        
        
        #print ("Left Wrist:", leftWrist)
        
        
        ## detect highpoint
        
        ### 1) take 10 frames from buffer and check for highest left/right arm y-coordinate

        leftArmYMaxHeightIndex = np.argmin([ f[3][1] for f in fb])
        rightArmYMaxHeightIndex = np.argmin([ f[8][1] for f in fb])
        # print("leftArmYMax:", leftArmYMaxIndex, fb[leftArmYMaxIndex][3]) #[1])
        
        ### 2) on highest point, check for rep conditions
        ###### left arm angle must be pressed out (angle > 170) 
        ###### and wrist must be over head (y-coordinate left wrist >= left ear) 
        ###### lowpoint must have been reached before
    
        
        if fb[leftArmYMaxHeightIndex][3][1] - fb[leftArmYMaxHeightIndex][4][1] < 0 and fb[leftArmYMaxHeightIndex][5] > 160 and leftArmLowPositionReached:
            print ("Frame:", frame, " - Left Arm Highpoint reached.")
            leftArmCounter += 1
            leftArmHighPositionReached = 1
            leftArmLowPositionReached = 0

        if fb[rightArmYMaxHeightIndex][8][1] - fb[rightArmYMaxHeightIndex][9][1] < 0 and fb[rightArmYMaxHeightIndex][10] > 160 and rightArmLowPositionReached:
            print ("Frame:", frame, " - Arm Highpoint reached.")
            rightArmCounter += 1
            rightArmHighPositionReached = 1
            rightArmLowPositionReached = 0


        ## detect lowpoint
        # print (leftAnkle[1]-leftWrist[1])
        if leftAnkle[1]-leftWrist[1] < 0.06 and leftArmLowPositionReached == 0:
            leftArmLowPositionReached = 1
            print ("Frame:", frame, " - Left Arm Lowpoint reached.", leftAnkle[1]-leftWrist[1], )
            
        if rightAnkle[1]-rightWrist[1] < 0.06 and rightArmLowPositionReached == 0:
            rightArmLowPositionReached = 1
            print ("Frame:", frame, " - Right Arm Lowpoint reached.", rightAnkle[1]-rightWrist[1])
        
       # print ("Frame:", frame, "Right / left wrist distance to ankle: ", rightAnkle[1]-rightWrist[1], leftAnkle[1]-leftWrist[1])
        
   
    except:
      pass
  
  
  
    # # Flip the image horizontally for a selfie-view display.
    # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    cv2.imshow('MediaPipe Pose', image)
    out.write(image)
  
  
    # slow down the video
    # time.sleep(0.05)
    # input()
    
    
  
    if cv2.waitKey(5) & 0xFF == 27:
        break
        

cap.release()
out.release()
cv2.destroyAllWindows()

