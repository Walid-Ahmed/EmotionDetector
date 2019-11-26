# USAGE
# python emotionDetector.py

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt




#declare the labels for the classifier
labels = ['Angry' , 'Disgust' ,'Fear', 'Happy','Neutral', 'Sad','Surprise']
#set the color of bounding box for different emotions
colorDict={'Angry':(0,0,255) , 'Disgust':(0,255,0)  ,'Fear':(0,191,255) , 'Happy':(0,165,255) ,'Neutral':(255,255,255) , 'Sad':(255,0,0) ,'Surprise':(190,132,178) }



# load caffe  model from face detection
print("[INFO] loading caffe  model from face detection ...")
net = cv2.dnn.readNetFromCaffe(os.path.join("caffeModel","deploy.prototxt.txt"), os.path.join("caffeModel","res10_300x300_ssd_iter_140000.caffemodel"))



pathToModel="FacialExpression_Classifier.h5"
modelFacialExpression=tf.keras.models.load_model(pathToModel)
print("[INFO] model loaded  from file {}".format(pathToModel))


#prepare video writers
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer1 = cv2.VideoWriter(os.path.join("Results","out1.avi"), fourcc, 5,(1000, 400), True) #save the frame with bar graph
writer2 = cv2.VideoWriter(os.path.join("Results","out2.avi"), fourcc, 5,(600, 400), True) #save the frame without bar graph

# loop over the frames from the camera video stream
while True:

	#intialize predictions to zero if in a frame a face was not detected
	predictions=np.zeros(len(labels))
	predictions=np.expand_dims(predictions,axis=1)

	#grab frames from camera
	cap  = cv2.VideoCapture(0)    
	ret, frame = cap.read()
	if (frame is None):
		continue

    #resize frame to 400px width
	frame = imutils.resize(frame, width=400)

 
	# grab the frame dimensions 
	h, w,_ = frame.shape

	# convert frame  to a blob
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
 
	# Apply face detection model to blob
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence of this prediction
		confidence = detections[0, 0, i, 2]

		if confidence < 0.5:  #minimum confidence to accept is 0.5
			continue

		# get the coordinates of the detected face 
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		
		y = startY - 10 if startY - 10 > 10 else startY + 10

        #grab the face
		face=frame[startY:endY, startX:endX]

		face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

		if (face is None):
			continue

		#preprocess face to be suitablefor emotion prediction	
		face=cv2.resize(face,(48,48))
		face=np.expand_dims(face,2)
		face=np.expand_dims(face,0)
		face = np.array(face, dtype="float") / 255.0

        #Apply inference
		predictions=modelFacialExpression.predict(face)
		emotion=labels[int(predictions.argmax(axis=1))]

		#draw the bounding box of the face and write the emotion
		cv2.rectangle(frame, (startX, startY), (endX, endY),colorDict[emotion], 2)
		cv2.putText(frame, emotion, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	

    #prepare the bar graph
	y_pos = np.arange(len(labels))
	objects=tuple(labels)
	y_pos = np.arange(len(objects))
	plt.figure()
	plt.bar(y_pos, predictions[0], align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.ylabel('Probabilty')
	plt.title('Emotion Probabilities')
	plt.savefig('tmp.png') #this can be improved
	probs=cv2.imread("tmp.png")
	plt.close() 



	# show  and save the output frame
	probs=cv2.resize(probs,(400,400))
	frame=cv2.resize(frame,(600,400))
	writer2.write(frame) #save the frame without bar graph
	frame=np.hstack((frame,probs))
	cv2.imshow("Frame", frame)
	writer1.write(frame)  #save the frame with bar graph

	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# cleanup
cv2.destroyAllWindows()
writer1.release()
writer2.release()
if os.path.exists("tmp.png"):
  os.remove("tmp.png")
