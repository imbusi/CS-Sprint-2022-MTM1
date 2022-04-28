import os
import cv2
import sys
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from MobileNetV2 import CreateMobileNet
from ResNet import CreateResNet

from time import time



ROOT_DIR = os.getcwd()

labelDictionary = dict()
model = input('Enter what model you want to use (MobileNet, ResNet): ')
if model == 'MobileNet':
    Savename = os.path.join(ROOT_DIR, 'SavedModels', 'MobileNetV2')
elif model == 'ResNet':
    Savename = os.path.join(ROOT_DIR, 'SavedModels', 'ResNet')
else:
    print('Wrong input.')
    sys.exit()

f = open(os.path.join(Savename, 'classes.txt'))
lines = f.readlines()
class_names = lines[0].split()
for i in range(len(class_names)):
    labelDictionary[i] = class_names[i]

print('Here is your class Dictionary:', labelDictionary)
print()
f.close()

class_counts = dict()

num_classes = len(class_names)
threshold = 1/float(num_classes)
print("Threshold: {:.2f}%".format(threshold*100))

saved_h5 = os.path.join(Savename, 'model.h5')

if model == 'MobileNet':
    Model = CreateMobileNet(labelDictionary)
    Model.load_weights(saved_h5)
    print('MobileNetV2 Loaded Succesfully.\n')
else:
    Model = CreateResNet(labelDictionary)
    Model.load_weights(saved_h5)
    print('ResNet50 Loaded Succesfully.\n')
start = time()


demo_path = os.path.join(ROOT_DIR, 'TestVideos','Eye_Travel', 'IMG_0876.MP4')


frameSize = (1280.0, 720.0)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('Demo_Eye_Travel_Success.mp4', fourcc, 15.0, (1280,720))

for cls in class_names:
    class_counts[cls] = 0
#Load the video

vidcap = cv2.VideoCapture(demo_path)
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
success, image = vidcap.read()
total_frames = 1
count = 1

width  = vidcap.get(3)  # float `width`
height = vidcap.get(4)  # float `height`
print('Dimemsions:', width, height)
fps = vidcap.get(5)
print('FPS:', fps)

#go frame by frame
while success:
    if count >= 30 or count < (length - 30):
        #Preprocess the image
        total_frames = total_frames + 1
        if model == 'ResNet':
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, (224, 224))
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            image = image[None, ...]
            image = image * 0.5 + 0.5
        else:
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, (224, 224))
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            image = image[None, ...]

    #get label for each frame
        image_probs = Model.predict(image)

        predicted_id = np.argmax(image_probs, axis=-1)
        test_label = class_names[predicted_id[0]]

    #add label to dictionary
    #class_counts[test_label] = class_counts[test_label] + 1
    #Add probabilities to dictionary


        for cls in class_names:
            if image_probs[0][class_names.index(cls)] > threshold:
                class_counts[cls] = class_counts[cls] + image_probs[0][class_names.index(cls)]

    success,image = vidcap.read()
    count = count + 1

    prob_denominator = 0
    for key, value in class_counts.items():
        prob_denominator = prob_denominator + value
    #for key, value in class_counts.items():
    #    class_counts[key] = "{:.2f}%".format(value*100.0/prob_denominator)
    key_list = []
    value_list = []
    for key, value in class_counts.items():
        key_list.append(key)
        value_list.append(value)
    key_list = [x for _,x in sorted(zip(value_list,key_list), reverse = True)]
    value_list = sorted(value_list, reverse = True)



    font = cv2.FONT_HERSHEY_SIMPLEX
    startX = 50
    startY = (720.0-100.0)/8.0
    output_text = ''
    for i in range(len(key_list)):
        output_text = key_list[i] + ": {:.2f}%".format(value_list[i]*100.0/prob_denominator)
        Y = int(50 + (startY * i))
        cv2.putText(image, output_text, (startX, Y), font, 0.5, (255, 0, 0), 2, cv2.LINE_8)
    out.write(image)
out.release()
print(class_counts)

end = time()
print('Total time of execution: {:.3f} seconds'.format(end-start))
