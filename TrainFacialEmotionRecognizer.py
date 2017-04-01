import sys
import dlib
import math
from sklearn import svm
import cv2
import csv
REFERENCE_LANDMARK_POINT = 30

NEUTRAL=0
HAPPY=1 
SAD=2
SURPRISE=3 
ANGRY=4 
DISGUST=5 
FEAR=6

def getEmotion(label):
    
    if(label[0]==0):
        return 'Neutral'
    elif(label[0]==1):
        return 'Happy'
    elif(label[0]==2):
        return 'Sad'
    elif(label[0]==3):
        return 'Surprise'
    elif(label[0]==4):
        return 'Angry'
    elif(label[0]==5):
        return 'Disgust'
    elif(label[0]==6):
        return 'Fear'
    else:
        sys.exit(1)
    return

def recognizeEmotion():
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()



    
    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()
        #print ret
    
        dets = detector(frame, 1)
        Mood = 'Neutral'
        
        for d in dets:
            # Get the landmarks/parts for the face in box d.
            shape = predictor(frame, d)

            #print makeAttributeVector(shape)
            Mood = getEmotion(svm_clf.predict([makeAttributeVector(shape)]))
            
        cv2.putText(frame,Mood,(10,470), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2)
    
        # Display the resulting frame
        cv2.imshow('EmotionRecognition',frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return

def getLabel(sampleFilePath):
    
    if(sampleFilePath[12:14]=='NE'):
        return NEUTRAL
    
    elif(sampleFilePath[12:14]=='HA'):
        return HAPPY
    
    elif(sampleFilePath[12:14]=='SA'):
        return SAD
    
    elif(sampleFilePath[12:14]=='SU'):
        return SURPRISE
    
    elif(sampleFilePath[12:14]=='AN'):
        return ANGRY
    
    elif(sampleFilePath[12:14]=='DI'):
        return DISGUST
    
    elif(sampleFilePath[12:14]=='FE'):
        return FEAR
    
    else:
        sys.exit(1)
        
    return 

def calculateDistanceSquare(x1,y1,x2,y2):
    return math.pow(x2-x1, 2) + math.pow(y2-y1, 2)

def makeAttributeVector(shape):
    
    attributeVector = []
    
    for i in range(68):
        
        if(i!=REFERENCE_LANDMARK_POINT):
            attributeVector.append(calculateDistanceSquare(shape.part(i).x,shape.part(i).y,shape.part(REFERENCE_LANDMARK_POINT).x,shape.part(REFERENCE_LANDMARK_POINT).y))
    
    return attributeVector


predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

   

trainingSample = []
trainingSampleLabel = []

trainingFeatureMatrixFileName = 'training_sample_features.csv'
trainingLabelVectorFileName = 'training_sample_label.csv'

with open(trainingFeatureMatrixFileName, 'r') as csvfile:
    reader = csv.reader(csvfile)
    trainingSample = [[int(float(e)) for e in r] for r in reader]    
        
with open(trainingLabelVectorFileName, 'r') as csvfile:
    reader = csv.reader(csvfile)
    trainingSampleLabel = [int(float(r[0])) for r in reader]
    
svm_clf = svm.LinearSVC()
svm_clf.fit(trainingSample,trainingSampleLabel)
recognizeEmotion()