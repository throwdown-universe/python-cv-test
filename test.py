import cv2
import pafy
import time

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



# for calculating FPS
prev_frame_time = 0         # used to record the time when we processed last frame
new_frame_time = 0          # used to record the time at which we processed current frame


# For webcam input:

# # 0: MacBook Webcam
# # 4: USB Webcam (Elgato FaceCam)
# ## capture from webcam
# cap = cv2.VideoCapture("media/jan_snatch.mov")
cap = cv2.VideoCapture("media/DB_Snatch_Workout.MP4")



# # capture from youtube video
# url = "https://youtu.be/tgHeP2vGwIY"
# video = pafy.new(url)
# best = video.getbest(preftype="mp4")

# cap = cv2.VideoCapture(best.url)

with mp_pose.Pose(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.9) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      # continue
      break
  
  
  
    
  
  
  

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    # print (results.pose_landmarks["11"])

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    
      # Our operations on the frame come here
    gray = image
  
    # resizing the frame size according to our need
    gray = cv2.resize(gray, (500, 300))
  
    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()
  
    # Calculating the fps
  
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
  
    # converting the fps into integer
    fps = int(fps)
  
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
    

    # print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    cv2.putText(gray, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    
       
    
    # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()