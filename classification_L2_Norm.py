from process_images import *
from sklearn.preprocessing import Normalizer
import numpy as np
from PIL import Image
from keras.models import load_model
from scipy.spatial import distance
import tkinter as tk
from tkinter import simpledialog

############################################
####### Choose method of recognition #######
def choose_rec():
    win = tk.Tk()
    win.geometry('200x100')

    def on_click(text):
        global ress
        ress = text
        win.destroy()

    b1 = tk.Button(win, text="Continuous/Live", command=lambda: on_click('live'))
    b1.pack()

    b2 = tk.Button(win, text="Detect and close", command=lambda: on_click('not_live'))
    b2.pack()

    win.mainloop()

    return ress

###############################################
####### Load the trained images dataset #######
def prepare_classifier():
    # Load train embeddings
    data = np.load('embeddings.npz')
    train_X, train_y = data['arr_0'], data['arr_1']

    # Load FaceNet model
    model = load_model('./model/keras/facenet_keras.h5')

    return train_X, train_y, model

###########################################################
####### Create the embeddings of the detected faces #######
def create_embeddings(model, train_X):
    # Load detected faces and pass them through the FaceNet to create the test embeddings
    test_X = list()
    for filename in os.listdir('./data/test'):
        path = os.path.join('./data/test', filename)
        im = Image.open(path)
        im = im.resize((160,160))
        im = np.array(im)
        embedding = get_embedding(model, im) # from: process_images.py
        test_X.append(embedding)

    '''
    Dimensions will be:
        -train_X: (#trainedPics,128)
        -train_y: (#trainedPics,)
        -test_X:  (#newUserPics,128)
    '''

    # Normalize input vectors
    in_encoder = Normalizer(norm='l2')
    train_X = in_encoder.transform(train_X)
    test_X = in_encoder.transform(test_X)

    return train_X, test_X


############################################################
####### Calculate L2 Norm and compare with threshold #######
def L2_norm(train_X, test_X, train_y, threshold):
    d_max=10
    dist = list()
    prediction = 'Not recognised!'
    for i in range(test_X.shape[0]):
        for j in range(train_X.shape[0]):
            d = distance.euclidean(train_X[j] , test_X[i])
            dist.append(d)
            if d < threshold and d < d_max:
                d_max=d
                prediction = train_y[j]

    return prediction, dist