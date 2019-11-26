# Emotion Detector

usage
'''
python emotionDetector.py
'''

This python script apply face detection to a live stream from camera and then each face is assigned a label based on emotion classifier.

The emotion classifier was trained to detect the following emotions: ['Angry' , 'Disgust' ,'Fear', 'Happy','Neutral', 'Sad','Surprise']

Face Detection is done by Caffe Resnet-SSD and  Face classification by CNN Keras. The face classification NN is trained on [Facial Expression Recognition dataset Challenge from Kaggle.](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
Thanks a lot to  Adrian Rosebrock  for his great inspiring article on  [Face  Detection](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)

The output will be automatically saved to folder Results as 2 videos, out1.avi which include the frames along with a bar graph for different emotions probabilities, the other file is out2.avi that  saves only the frame.  The code can handle multi faces however in this case the bar graph in out1.avi will not be relevant.

![](https://github.com/Walid-Ahmed/EmotionDetector/blob/master/sampleImage.png)
