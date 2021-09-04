import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import json
#import tensorflow as tf
import os
import glob

train_images=[]
train_labels= np.empty((0,2),int)

train_path = "/Users/shashwateshtripathi/Downloads/dataset/training"
labels = []
count=0
x_val=[]
y_val= np.empty((0,2),int)
print("reading training images and labels")
for root,dirs,files in os.walk(train_path):
    for d in dirs:
        labels.append(d)
print(labels)
for label in labels:
    path = os.path.join(train_path,label,'*')
    files = glob.glob(path)
    for f1 in files:
        image = (imread(f1).astype('uint8'))/255
        if count%4==0:
            x_val.append(image)
        train_images.append(image)
        if label == 'background':
            if count%4==0:
                y_val = np.append(y_val,np.array([[1,0]]),axis=0)
            train_labels = np.append(train_labels,np.array([[1,0]]),axis=0)
        else:
            if count%4==0:
                y_val = np.append(y_val,np.array([[0,1]]),axis=0)
            train_labels = np.append(train_labels,np.array([[0,1]]),axis=0)
        count+=1

train_images = np.asarray(train_images)
test_images=[]
test_path = "/Users/shashwateshtripathi/Downloads/dataset/test"
path = os.path.join(test_path,'*')
files = glob.glob(path)
print("reading test images ")
for f1 in files:
    image = (imread(f1).astype('uint8'))/255
    test_images.append(image)
test_images = np.asarray(test_images)
x_val = np.asarray(x_val)
print("training set image shape : {}".format(train_images.shape))
print("test set shape : {}".format(test_images.shape))
print("label shape: {} ".format(train_labels.shape))
print("x_val shape: {}".format(x_val.shape))
print("y_val shape: {}".format(y_val.shape))

from tensorflow import keras
import keras_tuner as kt

def build(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(64,64,3)))
    model.add(keras.layers.Dense(units=hp.Int("units",min_value=32,max_value=512,step =32),activation ="relu",))
    model.add(keras.layers.Dense(2,activation="softmax"))
    model.compile(optimizer =keras.optimizers.Adam(hp.Choice("learning_rate",values = [1e-2,1e-3,1e-4])),loss='categorical_crossentropy',metrics=["accuracy"])
    return model

tuner = kt.RandomSearch(build,objective = "val_accuracy",max_trials=10,executions_per_trial=2)

tuner.search(train_images,train_labels,epochs=10,validation_data=(x_val,y_val))

models = tuner.get_best_models(num_models=1)

prediction_model = keras.Sequential([models[0],keras.layers.Softmax()])
predictions = prediction_model.predict(test_images)

if __name__=='__main__':
    filename = './result2.json'
    res={}
    with open(filename,'w') as outfile:
        for i in range(98):
            answer = str(i+1) + '.jpg'
            res[answer] = int(np.argmax(predictions[i]))

        json.dump(res,outfile)

