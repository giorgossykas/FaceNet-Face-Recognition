Files:
- register_user.py: Recording and saving user data.
- face_recognition.py: Starts face recognition.
- new_user.py : Functions for recording and saving user data.
- process_images.py : Functions for processing and feature/embedding extraction of user images.
- classification_L2_Norm.py: Functions for data processing at the face recognition stage.
- embeddings.npz: It will be created after running register_user.py once and will contain user embedding and id/name.
- model: Facenet model and haarcascade_frontalface_alt.xml.
- code: Constains the source code for the FaceNet model architecture, inception_resnet_v1.py .
- requirements.txt : Required packages and versions (for virtual environment).
- README.txt : Instructions and information.

: Instructions
1. Clone repository.
2. Conda environment FaceNet.yml contains all packages and dependancies (). In case of VirtualEnvironment packages are in requirements.txt.
3. Download FaceNet model from one of two links: 1) https://github.com/davidsandberg/facenet , 2)  .
4. Place the file downloaded "facenet_keras.h5" in the folder: /model/keras . If not placed there the algorithm will not find it.
5. Run register_user.py to take pictures of new user. Data will be saved in the right folders.
6. Run face_recognition.py and choose method, real time continuous or just one person and close.
7. Results printed: for each frame in which a face was detected the algorithm will print a score of comparison/distance with all the registered users.
   If the score is less than one it will choose that user as the prediction and return the name. Otherwise it will keep printing. 
   If no face has been detected
   in case of "detect and close" option the algorithm will stop after 30 non-satisfied detections.


Code was written in PyCharm Community Edition 2021.3.3 with Python 3.6 and
the pre-trained FaceNet model was created in Ubuntu16.04/Windows10, Python 3.6.2, tensorflow 1.3.0 and keras 2.1.2 .
(tensorflow version used in conda environment is 1.15.0 and keras is 2.3.1 due to some compatibility issues.)
Some changes are necessary in order to run on GPU.

FaceNet model was created by David Sandberg.
