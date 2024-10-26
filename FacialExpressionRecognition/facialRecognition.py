import cv2
import cv2.data
import numpy as np
import keras
import os

def faceRecognition():
    # Load trained facial expression neural network model
    model = keras.models.load_model('EmotionRecogniser.h5')

    # OpenCV's pre-trained face detector
    # This will make it easier to process each frame, as we can focus on the face on its own rather than the whole image
    # This code was partially taken from others in stackoverflow, as there is no other way to define this code
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # This prevents the log message from showing


    # Have to be in the same order as the network trained to provide correct index
    # Its all in the eyes, neutral works fine, happy you must be wide eyes and smiling, sad you must be squinted and frowning, for neutral, your eyes must also be wide
    emotions = ["happy", "sad"]

    # Captures an image stream of primary camera device
    capture = cv2.VideoCapture(0)

    counter = 0
    while True:

        # successful returns true if image stream recognised, false if not
        successful, frame = capture.read()

        # Ends loop if error occurs
        if not successful:
            break

        # Change image to grayscale to reduce input variables and to comply with the input of our neural network
        # (We trained the model on grayscale images, so their colour dimension is 1)
        grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face in image
        faces = faceCascade.detectMultiScale(grayImage, scaleFactor=1.3, minNeighbors=5)


        for(x, y, w, h) in faces:

            # Extract face region from the frame
            faceRegion = grayImage[y:y+h, x:x+w]

            # Resize image to fit with the input layer of trained data
            faceRegionResized = cv2.resize(faceRegion, (48, 48))

            # Change dimensions of image to fit the input layer of the neural network
            faceRegionResized = np.expand_dims(faceRegionResized, axis=-1)

            faceRegionResized = np.expand_dims(faceRegionResized, axis=0)

            # Normalise grayscale value, in keras we did this by doing .normalize()
            faceRegionResized = faceRegionResized / 255.0

            prediction = model.predict(faceRegionResized)

            print(prediction)

            highestPobabilityIndex = np.argmax(prediction)

            print(highestPobabilityIndex)

            label = emotions[highestPobabilityIndex]

            counter += 1

            print(label)

        
        
        cv2.imshow("Facial Expression Recognition", frame)

        if counter > 25:
            print(f"Final label: {label}")
            break

        # Break if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
    return label


faceRecognition()
