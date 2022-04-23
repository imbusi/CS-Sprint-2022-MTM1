# MTM study: OpenCV/ML team

We made use of OpenCV and Machine Learning to accuractely identify what motion is being performed in a video

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
  <li>Run Tests using python TestVideos.py</li>
</ol>
