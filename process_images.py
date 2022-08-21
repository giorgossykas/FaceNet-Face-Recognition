import os
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN

#############################################################
####### Extract a single face from a given photograph #######
def extract_face(filename , required_size = (160,160)):
    image = Image.open(filename)                # Open image
    image = image.convert('RGB')                # Convert to RGB
    pixels = np.asarray(image)                  # np array
    detector = MTCNN()                          # Creates detector from default weights
    results = detector.detect_faces(pixels)     # Detects face
    x1, y1, width, height = results[0]['box']   # Extract face bounding box
    x1, y1 = abs(x1), abs(y1)                   # Fix negative points
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]                 # Extract face
    image = Image.fromarray(face)
    image = image.resize(required_size)         # Resize image
    face_array = np.asarray(image)                 # np array

    return face_array

##########################################################################
####### Go through a single directory to extract faces - new entry #######
def load_faces(directory):
    faces = list()
    for filename in os.listdir(directory):
        path = os.path.join(directory,filename)
        face = extract_face(path)
        faces.append(face)
    return faces


#####################################################################################################
####### Go through multiple direrctories to get images from all files - train an many persons #######
def load_dataset(directory):
    X , y = list() , list()
    for subdir in os.listdir(directory):
        path = directory + subdir + '/'

        if not os.path.isdir(path):                             # Skip files that ae not in a folder
            continue

        faces = load_faces(path)                        # load faces in subdirectory
        labels = [subdir for _ in range(len(faces))]    # Create labels(name) for every person
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        X.extend(faces) # stores faces
        y.extend(labels)# stores labels(names)

    return np.asarray(X) , np.asarray(y)


#############################
####### Get embedding #######
def get_embedding(model , face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean , std = face_pixels.mean() , face_pixels.std()
    face_pixels = (face_pixels - mean)/std
    samples = np.expand_dims(face_pixels , axis = 0)
    yhat = model.predict(samples)
    return yhat[0]
