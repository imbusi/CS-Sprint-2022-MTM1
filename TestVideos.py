#Load the Model
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


#sys.exit()
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

test_path = os.path.join(ROOT_DIR, 'TestVideos')
#Go by class
total_frames = 0
total_correct = 0
total_checked = 0
motions = os.listdir(test_path)
for motion in motions:
    motion_path = os.path.join(test_path, motion)
    videos = os.listdir(motion_path)

    motion_correct = 0
    motion_checked = 0
    print('Motion:', motion)
    print()
    #For each video in class
    for video in videos:
        for cls in class_names:
            class_counts[cls] = 0
        #Load the video
        video_path = os.path.join(motion_path, video)
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 1
        #go frame by frame
        while success:

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
                if model == 'ResNet': #looks only at obvious frames
                    if image_probs[0][class_names.index(cls)] > 0.5:
                        class_counts[cls] = class_counts[cls] + image_probs[0][class_names.index(cls)]
                else:
                    class_counts[cls] = class_counts[cls] + image_probs[0][class_names.index(cls)]

            success,image = vidcap.read()
            count += 1
        #Find Max
        max = 0
        max_class = None
        #return max label
        for key, value in class_counts.items():
            if value >= max:
                max = value
                max_class = key
        print(video)
        print(class_counts)

        #check if correct
        if max_class == motion:
            motion_correct = motion_correct + 1
        motion_checked = motion_checked + 1

    #Add Accuracy Statistics
    if motion_checked == 0:
        motion_percent = float('inf')
    else:
        motion_percent = float(motion_correct)/float(motion_checked) * 100.0
    print('\n{} Accuracy : {}/{} = {:.3f}%\n'.format(motion, motion_correct, motion_checked, motion_percent))

    total_correct = total_correct + motion_correct
    total_checked = total_checked + motion_checked

if total_checked == 0:
    total_percent = float('inf')
else:
    total_percent = float(total_correct)/float(total_checked) * 100.0
print('\n\nOverall Accuracy : {}/{} = {:.3f}%'.format(total_correct, total_checked, total_percent))

end = time()
print('Total time of execution: {:.3f} seconds. {} frames. fps = {}'.format(end-start, total_frames, float(total_frames)/(end-start)))
print('\nDone!')
