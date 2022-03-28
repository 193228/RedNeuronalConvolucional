import os
import numpy as np
from imutils import paths
import random
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report


def procesadoImagenes(directorio):
    # initialize the data and labels
    print("Cargando Imagenes")
    data = []
    labels = []

    imagePaths = sorted(list(paths.list_images(directorio)))
    random.seed(42)
    random.shuffle(imagePaths)

    # loop over the input images
    for imagePath in imagePaths:
        # load the image, resize it to 64x64 pixels (the required input spatial dimensions of SmallVGGNet),
        # and store the image in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image,(64, 64))  # we are not flattening our data for neural network, because it is convolutional
        data.append(image)
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    print("imagenes cargadas")
    return [data,labels]

def divisionDatos(data,labels):
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
    return [trainX,testX,trainY,testY]

def entrenamientoRed(datos,model,aug,trainX,trainY,testX,testY):
    print("Entrenando Red")
    opt = SGD(lr=datos['eta'], decay=datos['eta']/datos['epocas'])
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # train the network
    entrenamiento = model.fit(x=aug.flow(trainX, trainY, batch_size=datos['lote']), validation_data=(testX, testY),steps_per_epoch=len(trainX) // datos['lote'], epochs=datos['epocas'])
    return entrenamiento

def evaluarRed(model,testX,testY,lb):
    print("Evaluando Red")
    predictions = model.predict(x=testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=lb.classes_))