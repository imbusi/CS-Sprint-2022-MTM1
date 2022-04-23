import cv2

import os
import sys

ROOT_DIR = os.getcwd()

videos_path = os.path.join(ROOT_DIR, 'Videos') # Videos are read from here
images_path = os.path.join(ROOT_DIR, 'Images') #Images are saved here

motions = os.listdir(videos_path)

print('The motions are:', motions)

#Walk through each motions videos
for motion in motions:
    motion_path = os.path.join(videos_path, motion)
    out_motion_path = os.path.join(images_path, motion)
    videos = os.listdir(motion_path)
    os.mkdir(out_motion_path)
    #walk through each video
    for video in videos:
        video_path = os.path.join(motion_path, video)
        vidcap = cv2.VideoCapture(video_path)
        success,image = vidcap.read()
        count = 1
        while success:
            out_name = video[:-4] + '_{}.jpg'.format(count)
            out_frame_path = os.path.join(out_motion_path, out_name)
            cv2.imwrite(out_frame_path, image)     # save frame as JPEG file
            success,image = vidcap.read()
            count += 1
    print(motion, 'Complete.')
print('All Motions Complete. {} frames saved.'.format(count))
