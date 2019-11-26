# Emotion Detector

usage
python emotionDetector.py

This python script apply face detection to a live stream from camera and then each face is assigned a label based on emotion classifier.

The emotion classifier was trained to detect the following emotions: ['Angry' , 'Disgust' ,'Fear', 'Happy','Neutral', 'Sad','Surprise']

Face Detection done by Caffe Resnet-SSD and  Face classification by CNN Keras. The face classification NN is trained on [Facial Expression Recognition dataset Challenge from Kaggle.](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
Thanks a lot to  Adrian Rosebrock  for his great inspiring article on  Face  Detection

