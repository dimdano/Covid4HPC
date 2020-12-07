import tensorflow as tf

from tensorflow.keras.utils import to_categorical

import os
import cv2

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def data_preprocess(normalize=1):
    
    
    IMG_SIZE = 224

    imagePaths = []

    for dirname, _, filenames in os.walk('dataset/'):
        for filename in filenames:
            if (filename[-3:] == 'png'):
                imagePaths.append(os.path.join(dirname, filename))

    #Should return true for our dataset.
    len(imagePaths) == 219+1341+1345

    X = []
    y = []

    for img_path in imagePaths:
        label = img_path.split(os.path.sep)[-2]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X.append(img)
        y.append(label)

    if normalize == 1:
        X = np.array(X) / 255.0
    else:
        X = np.array(X)
    y = np.array(y)


    #View counts of different labels
    y_df = pd.DataFrame(y, columns=['Labels'])
    print(y_df['Labels'].value_counts())

    #Encode labels as integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    #Convert list of labels to one-hot format
    y_encoded = np_utils.to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, stratify=y_encoded, random_state=3)

    return (X_train,y_train), (X_test,y_test)   
