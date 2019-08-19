# USAGE
# python classify.py --model svpp_stu.model --labelbin svpp_stu.pickle --image examples/example_01.jpg

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import pandas as pd
from pygame import mixer
from gtts import gTTS
from imutils.video import VideoStream
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade",type=str,default="haarcascade_frontalface_default.xml",
	help = "path to where the face cascade resides")
ap.add_argument("-o", "--output", type=str,default="examples",
	help="path to output directory")
ap.add_argument("-m", "--model",type=str,default="svpp_stu_16b.model",
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", type=str,default="svpp_stu_16b.pickle",
	help="path to label binarizer")
ap.add_argument("-i", "--image",type=str,default="examples/00000.png",
	help="path to input image")
args = vars(ap.parse_args())
#Image Capturing
# USAGE
# python build_face_dataset.py --cascade haarcascade_frontalface_default.xml --output dataset/adrian

# load OpenCV's Haar cascade for face detection from disk
detector = cv2.CascadeClassifier(args["cascade"])

# initialize the video stream, allow the camera sensor to warm up,
# and initialize the total number of example faces written to disk
# thus far
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
total = 0

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, clone it, (just
	# in case we want to write it to disk), and then resize the frame
	# so we can apply face detection faster
	frame = vs.read()
	orig = frame.copy()
	frame = imutils.resize(frame, width=400)

	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(
		cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30))

	# loop over the face detections and draw them on the frame
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	cv2.waitKey(5000) & 0xFF
 
	# if the `k` key was pressed, write the *original* frame to disk
	# so we can later process it and use it for face recognition
	#if key == ord("k"):
	p = os.path.sep.join([args["output"], "{}.png".format(
			str(total).zfill(5))])
	cv2.imwrite(p, orig)
	total += 1

	# if the `q` key was pressed, break from the loop
	#elif key == ord("q"):
	break

# do a bit of cleanup
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()

# load the image
image = cv2.imread(args["image"])
output = imutils.resize(image, width=400)
print("Hello",args["image"])
# pre-process the image for classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network and the multi-label
# binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
mlb = pickle.loads(open(args["labelbin"], "rb").read())

# classify the input image then find the indexes of the two class
# labels with the *largest* probability
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:2]

# loop over the indexes of the high confidence class labels
for (i, j) in enumerate(idxs):
	# build the label and draw the label on the image
	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
	cv2.putText(output, label, (10, (i * 30) + 25), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
'''if(p*100 > 20):'''
# show the probabilities for each of the individual labels
for (label, p) in zip(mlb.classes_, proba):
         temp=label
         print("{}: {:.2f}%".format(label, p * 100))
        
        
               
#Retriving Name from Pandas Using RollNo
df = pd.read_csv("batch1.csv")
xbox_one_filter =  (df["RollNo"] == temp)
filtered_reviews = df[xbox_one_filter]
temp_name=str(filtered_reviews["Name"])
temp_name = temp_name.split(" ") 
temp_name=temp_name[4].replace('Name:', '')
print("Hello",temp)

#Text To Speech to Verify whether Attendance is Marked or Not
mixer.init()
text=("welcome "+temp_name)
print(temp_name)
speech=gTTS(text,'en','slow')
speech.save("welcome.mp3")
mixer.music.load("welcome.mp3")
mixer.music.play()

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
df.loc[df['RollNo']==temp,['Present']]='1'
df.to_csv('/home/aidl1/Desktop/FaceID/batch1.csv',index=False)


