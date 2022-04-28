import cv2
from time import time
import os
import sys

start = time()
ROOT_DIR = os.getcwd()

videos_path = os.path.join(ROOT_DIR, 'Videos') # Videos are read from here
images_path = os.path.join(ROOT_DIR, 'Images') #Images are saved here

motions = os.listdir(videos_path)

print('The motions are:', motions)
start = time()
total_count = 1
#Walk through each motion's videos
for motion in motions:
    motion_path = os.path.join(videos_path, motion)
    out_motion_path = os.path.join(images_path, motion)
    videos = os.listdir(motion_path)
    os.mkdir(out_motion_path)
    print('\nStarting', motion)
    print(len(videos), 'total.')
    #walk through each video
    for video in videos:
        video_path = os.path.join(motion_path, video)
        vidcap = cv2.VideoCapture(video_path)
        success,image = vidcap.read()
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        count = 1
        save_Count = 1
        #Write Each frame to a jpg
        while success:
            if count >= 30 and count < (length - 30):
                out_name = video[:-4] + '_{}.jpg'.format(count)
                out_frame_path = os.path.join(out_motion_path, out_name)
                cv2.imwrite(out_frame_path, image)     # save frame as JPEG file
                save_Count = save_Count + 1
            success,image = vidcap.read()
            count += 1
        print(video, 'complete.', save_Count - 1, 'frames saved.')
        
        total_count = total_count + save_Count - 1
    print(motion, 'Complete.')
end = time()
print('All Motions Complete. {} frames saved.'.format(total_count-1))
print('Total time of execution: {:.3f} seconds'.format(end-start))
print('Time per frame = {} seconds'.format(float(end-start)/float(total_count-1)))
