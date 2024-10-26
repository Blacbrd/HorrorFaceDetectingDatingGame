import numpy as np
import os
import cv2
import random

# Path to dataset file from Kaggle 
# https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

# Since there are thousands of training images, I will have them in a different directory
PATH_TO_IMAGES = "C:\\Users\\blacb\\Desktop\\Datasets\\archive\\train"

# Categories to distinguish
categories = ["happy", "sad"]

# Train neural network
trainingData = []

for category in categories:

    # Dataset/happy, Dataset/sad etc.
    path = os.path.join(PATH_TO_IMAGES, category)

    print(f"Path is: {path}")

    print(f"Category is: {category}")

    # 0: neutral, 1: sad, 2: happy
    categoryNumber = categories.index(category) 

    print(f"Category number is: {categoryNumber}")

    for img in os.listdir(path):

        # As some images may not process, we will skip them
        try:
            # Makes sure that the image is gray scale
            grayImg = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

            imgSize = 48
            # Makes sure all images are same size, 100x100 
            resisedGrayImage = cv2.resize(grayImg, (imgSize, imgSize))

            # Adds image and its category to training data
            trainingData.append([resisedGrayImage, categoryNumber])


        except Exception as e:
            print(e)
    
# This is to mix up the data, so its not all "0, 0, 0, 0, ...., 1, 1, 1" etc.
random.shuffle(trainingData)

x_train = [] # Image of face
y_train = [] # The label associated with the face

# Seperates feature (like happy, sad etc.) with its label
for feature, label in trainingData:

    x_train.append(feature)
    y_train.append(label)

# (Amount of images there are, height, width, colour depth)
# Since we don't know the amount of images, we use -1
x_train = np.array(x_train).reshape(-1, imgSize, imgSize, 1)

np.save("EmotionTrainingData.npy", x_train)
np.save("EmotionLabels.npy", y_train)


