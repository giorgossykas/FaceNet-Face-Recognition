import os.path

from new_user import *
from process_images import *
from keras.models import load_model

# Create folders for images if they do not already exist
if not os.path.exists('data'):
    dir = os.getcwd()
    path = os.path.join(dir, 'data')
    os.mkdir(path)
    path2 = os.path.join(path, 'train')
    path3 = os.path.join(path, 'test')
    os.mkdir(path2)
    os.mkdir(path3)


# Input new user and get path to the images
directory , name = register() # from: new_user.py

# Align/crop images and store at "faces" (list of numpy arrays)
trainX = load_faces(directory) # from: process_images.py

# Target variable/label will for now be the name
trainy = np.array([name for _ in range(len(trainX))])

# Load the FaceNet model
model = load_model('./model/keras/facenet_keras.h5')

# Convert each face in the train set in an embedding
newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels) # from: process_images.py, returns 128 dimensional vector
    newTrainX.append(embedding)

newTrainX = np.asarray(newTrainX) # Now newTrainX will have 128-Dimensional vectors instead of images

# Store embeddings in compressed file
if os.path.exists('embeddings.npz'):
    temp_data = np.load('embeddings.npz')
    oldTrainX, oldTrainy = temp_data['arr_0'] , temp_data['arr_1']
    train_X , train_y = np.concatenate((oldTrainX , newTrainX)) , np.concatenate((oldTrainy , trainy))
    np.savez_compressed('embeddings.npz', train_X, train_y)
else:
    np.savez_compressed('embeddings.npz', newTrainX, trainy)
