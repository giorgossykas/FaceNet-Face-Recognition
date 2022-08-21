import cv2
import os
import shutil
import tkinter as tk
from tkinter import simpledialog


###############################
####### Get name of user ######
def getName():
    ROOT = tk.Tk()

    ROOT.withdraw()
    # the input dialog
    USER_INP = simpledialog.askstring(title="",
                                      prompt="Insert Name")
    return USER_INP


#################################
####### Register new user #######
def register():
    # Make directory for images of new user
    parent_dir = "./data/train"
    name_directory = getName()
    path = os.path.join(parent_dir, name_directory)
    os.mkdir(path)

    # Open camera to take pictures
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Camera")
    img_counter = 0

    while img_counter<5:
        ret , frame = cam.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Define texts to appear
        if img_counter == 0:
            text = "Look directly at the camera and press 'Space' to take picture"
        elif img_counter == 1:
            text = "Tilt your head slightly to the right and press 'Space' to take picture"
        elif img_counter == 2:
            text = "Tilt your head slightly to the left and press 'Space' to take picture"
        elif img_counter == 3:
            text = "Tilt your head slightly upwards and press 'Space' to take picture"
        elif img_counter == 4:
            text = "Tilt your head slightly downwards and press 'Space' to take picture"

        # Draw box and text
        x,y = 0,0
        w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cv2.rectangle(frame , (x,y) , (int(x+w),int(y+h/11)) , (0,0,0) , -1)
        image = cv2.putText(frame , text, (10 , 25) ,
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

        # Open video camera
        cv2.imshow("Camera" , image)

        # Key to press for action
        k = cv2.waitKey(1)

        if k%256 == 27: # "Esc" key is hit
            print("Escape hit, closing app.")
            break
        elif k%256 == 32: # "Space" key is hit
            img_name = "picture_{}.jpg".format(img_counter+1)
            cv2.imwrite(os.path.join(path , img_name) , frame)
            print("Screenshot taken")
            img_counter += 1
            #if img_counter == 5:
                # Duplicate images to have more training data
                #copy_images(path,2)

    cam.release()
    cv2.destroyAllWindows()
    return path , name_directory


###########################
####### Copy images #######
def copy_images(src,num):

    def copy_img(src):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            name,extension = os.path.splitext(item)
            shutil.copy(s, os.path.join(src, "{Name} {Index}.jpg".format(Name = str(name) , Index = str(1))))


    for i in range(num):
        copy_img(src)
