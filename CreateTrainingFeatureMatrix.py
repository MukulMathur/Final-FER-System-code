import os
import sys
import dlib
import glob
from skimage import io
import math
import csv
from sklearn import svm
from sklearn import cross_validation

REFERENCE_LANDMARK_POINT = 30

NEUTRAL=0
HAPPY=1 
SAD=2
SURPRISE=3 
ANGRY=4 
DISGUST=5 
FEAR=6

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


faces_folder_path = '../image'

predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

trainingImages = glob.glob(os.path.join(faces_folder_path, "*.jpg")) + glob.glob(os.path.join(faces_folder_path, "*.png"))

trainingSample = []
trainingSampleLabel = []

for sampleCounter, sampleFilePath in enumerate(trainingImages):
    
    print("Processing File {}: {}".format(sampleCounter,sampleFilePath))
    img = io.imread(sampleFilePath)
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    
    for d in dets:
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        
        trainingSample.append(makeAttributeVector(shape))
        trainingSampleLabel.append(getLabel(sampleFilePath))
    
    
        
trainingFeatureMatrixFileName = 'training_sample_features.csv'
trainingLabelVectorFileName = 'training_sample_label.csv'

with open(trainingFeatureMatrixFileName, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(trainingSample)
    
with open(trainingLabelVectorFileName, 'w') as csvfile:
    writer = csv.writer(csvfile)
    [writer.writerow([r]) for r in trainingSampleLabel]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainingSample, trainingSampleLabel, test_size=0.3, random_state=0)

svm_clf = svm.LinearSVC()
svm_clf.fit(X_train,y_train)
print svm_clf.score(X_test, y_test)
        

