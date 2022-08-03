import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
path = 'myData'
images = []
classNo = []
testRadio = 0.2
validRadio = 0.2
myList = os.listdir(path)
imageDimensions = (32,32,3)
batchSizeVal = 1
epochsVal = 20
stepsPerEpoch = 4
print("Total no of classes detected", len(myList))
noOfClasses = len(myList)
print("Importing classes.........")
for x in range (0, noOfClasses):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        print(path + '/' + str(x) + '/' + y)
        curImg = cv2.imread(path + '/' + str(x) + '/' + y)
        curImg = cv2.resize(curImg, (imageDimensions[0],imageDimensions[1]))
        images.append(curImg)
        classNo.append(x)
    print(x)

images = np.array(images)
classNo = np.array(classNo)
print(images.shape)
print(classNo.shape)
### Spliting the data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRadio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validRadio)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

numOfSamples = []
for x in range (0, noOfClasses):
    print(len(np.where(y_train==x)[0]))
    numOfSamples.append(len(np.where(y_train==x)[0]))
print(numOfSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0, noOfClasses), numOfSamples)
plt.title("No of images for each class")
plt.xlabel("Number of Images")
plt.ylabel("Number of Images")
# plt.show()

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img
# img = preProcessing(X_train[10])
# img = cv2.resize(img, (300,300))
# cv2.imshow("PreProcessed", img)
# cv2.waitKey(0)

X_train = np.array(list(map(preProcessing,X_train)))
X_test = np.array(list(map(preProcessing,X_test)))
X_validation = np.array(list(map(preProcessing,X_validation)))
# img = X_train[10]
# img = cv2.resize( img, (300,300))
# cv2.imshow("PreProcessed", img)
# cv2.waitKey(0)
print(X_test.shape)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1], X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1], X_validation.shape[2],1)
print(X_train.shape)

dataGen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,shear_range=0.1,rotation_range=10)
dataGen.fit(X_train)
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNode = 500
    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0], imageDimensions[1], 1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(learning_rate=.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
model = myModel()
print(model.summary())

history = model.fit_generator(dataGen.flow(X_train,y_train,
                    batch_size=batchSizeVal),
                    steps_per_epoch=stepsPerEpoch,
                    epochs=epochsVal,
                    validation_data=(X_validation, y_validation),
                    shuffle=1)
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy =', score[1])

pickle_out = open("A/model_trained.p","wb")
pickle.dump(model, pickle_out)
pickle_out.close()