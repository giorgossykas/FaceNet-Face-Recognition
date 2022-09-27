from __future__ import print_function
import cv2
from classification_L2_Norm import *

# Detect and extraxt face from camera frame
def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    found=False # Checks if a face was detected in the current frame
    for (x,y,w,h) in faces:
        start_points = (x, y)
        end_points = (x+w, y+h)
        frame = cv2.rectangle(frame, start_points, end_points, (255, 0, 0), 4)
        faceROI = frame[y:y+h, x:x+w]
        found = True

    cv2.imshow('Capture - Face detection', frame)
    # Now return the rectangle picture and save it to pass it to the FaceNet
    # If a face was not detected(found == False) the variable faceROI will not be used so I pass a zero
    if found == False:
        faceROI=0
        start_points = (0,0)
        end_points = (0,0)
    return faceROI, found, start_points, end_points

train_X, train_y, model = prepare_classifier() # from classication_L2_Norm.py

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Face detection algorithm

cap = cv2.VideoCapture(0)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

recognition = choose_rec() # from classication_L2_Norm.py to choose 'method' of recognition.
if recognition != 'live':

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        faceROI, found, _, _ = detectAndDisplay(frame)
        if found==True:
            cv2.imwrite(f"./data/test/frame_{frame_count}.jpg", faceROI)
            train_X, test_X = create_embeddings(model, train_X)
            prediction, dist = L2_norm(train_X, test_X, train_y, threshold=1)
            os.remove(f"./data/test/frame_{frame_count}.jpg")
            frame_count += 1
            print(f"Face detected #{frame_count}")
            for i in range(np.unique(train_y).shape[0]): # np.unique(train_y).shape[0] is number of different names
                print(f"Score for {train_y[i*5]} images: {dist[i*5:(i+1)*5]}")
            print('-----------------------')
            if prediction != 'Not recognised!':
                print(f"Prediction: {prediction}")
                break
            if frame_count==30:
                print(prediction)
                break


        if cv2.waitKey(10) == 27: ## 27 represents 'Escape' button
            break

else:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        faceROI, found, start_points, end_points = detectAndDisplay(frame)
        if found == True:
            cv2.imwrite(f"./data/test/frame_{frame_count}.jpg", faceROI)
            train_X, test_X = create_embeddings(model, train_X)
            prediction, dist = L2_norm(train_X, test_X, train_y, threshold=1)
            os.remove(f"./data/test/frame_{frame_count}.jpg")
            frame_count += 1

            cv2.rectangle(frame, start_points, end_points, color=(255, 0, 0), thickness=2)
            name_above = cv2.putText(frame, prediction, (start_points[0]+10, start_points[1]-10),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)
            cv2.imshow("Camera", name_above)

        if cv2.waitKey(10) == 27:  ## 27 represents 'Escape' button
            break