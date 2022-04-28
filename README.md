



# CS-Sprint-2022-MTM1 - OpenCV/ML Implementation
We made use of OpenCV and Machine Learning to accuractely identify what motion is being performed in a video. Towards the goal of accurately identifying the correct MTM code. This project was worked on by Asit Singh and Vyas Padmanabhan.

**This project was performed as part of CS 499 and Mercedes-Benz U.S. International (MBUSI) was the collaborator.**


## How to Arrange Videos in the Video folder (same applies for TestVideos)
Videos/  
...Motion1/  
......Motion1_video_1.mov  
......Motion1_video_2.mov  
...Motion2/  
......Motion2_video_1.mov  
......Motion2_video_2.mov  


## To Train the Machine Learning Model
<ol>
  <li>Put your Videos into the correct folder (Videos)</li>
  <li>Turn videos into jpgs using python VideoToImage.py</li>
  <li>Train Classifier using python TrainModels.py</li>
</ol>

## To Test the classifier
<ol>
  <li>Put your Videos into the correct folder (TestVideos)</li>
  <li>Run Tests using python RunTests.py</li>
</ol>
