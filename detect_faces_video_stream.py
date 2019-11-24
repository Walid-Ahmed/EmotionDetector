# USAGE
# python detect_faces_video_stream.py 

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import tensorflow as tf
import pickle

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


labels = ['Angry' , 'Disgust' ,'Fear', 'Happy','Neutral', 'Sad','Surprise']



# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")

camera = 0
pathToModel="FacialExpression_Classifier.h5"
modelFacialExpression=tf.keras.models.load_model(pathToModel)
print("[INFO] model loaded  from file {}".format(pathToModel))
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer1 = cv2.VideoWriter("out.avi", fourcc, 10,(1000, 400), True)


# loop over the frames from the video stream
while True:
	predictions=np.zeros(len(labels))
	predictions=np.expand_dims(predictions,axis=1)
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	cap  = cv2.VideoCapture(camera)
	ret, frame = cap.read()
	if (frame is None):
		continue

	frame = imutils.resize(frame, width=400)

 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < args["confidence"]:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
		face=frame[startY:endY, startX:endX]

		face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

		if (face is None):
			continue
		face=cv2.resize(face,(48,48))
		face=np.expand_dims(face,2)
		face=np.expand_dims(face,0)
		face = np.array(face, dtype="float") / 255.0

		predictions=(modelFacialExpression.predict(face))
		emotion=labels[int(predictions.argmax(axis=1))]
		cv2.putText(frame, emotion, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		pathToSaveFace=os.path.join("Results","face.png")
		cv2.imwrite(pathToSaveFace,face)

	y_pos = np.arange(len(labels))
	objects=tuple(labels)
	y_pos = np.arange(len(objects))
	plt.figure()

	plt.bar(y_pos, predictions[0], align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.ylabel('Probabilty')
	plt.title('Emotion Probabilities')
	plt.savefig('gant.png')
	probs=cv2.imread("gant.png")
	cv2.imshow("Gant chart",probs)
	plt.close() 



	# show the output frame
	probs=cv2.resize(probs,(400,400))
	frame=cv2.resize(frame,(600,400))



	frame=np.hstack((frame,probs))
	cv2.imshow("Frame", frame)
	writer1.write(frame)

	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
writer1.release()
