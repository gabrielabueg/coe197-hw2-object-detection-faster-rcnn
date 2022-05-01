# Objection Detection using Faster R-CNN on the Drinks Dataset
Homework 2 submission for CoE197Z
by John Gabriel Porton , 201803016

## Link/Reference
This homework is based on [Sovit Ranjan Raths Tutorial series on object detection with python and torchvision](https://debuggercafe.com/ssdlite-mobilenetv3-backbone-object-detection-with-pytorch-and-torchvision/)

## Data preparation
### Data Structure

 -Dataset\
 &emsp;|--test\
	&emsp;&emsp;|--0000020.jpg\
	&emsp;&emsp;|--0000020.xml\
 	&emsp;&emsp;...\
 &emsp;|--train\
      &emsp;&emsp;|--0000100.jpg\
      &emsp;&emsp;|--0000100.xml\
      &emsp;&emsp;...
      
 &emsp;-outputs\
 	&emsp;&emsp;|--model.pth\
 	&emsp;&emsp;|--train_loss_10.png\
	&emsp;&emsp;|--valid_loss_10.png\
 	&emsp;&emsp;...

 &emsp;-src\
  &emsp;&emsp;|--config.py\
  &emsp;&emsp;|--datasets.py\
  &emsp;&emsp;|--engine.py\
  &emsp;&emsp;|--inf_single.py\
  &emsp;&emsp;|--inf_video.py\
  &emsp;&emsp;|--inf_batch.py
  
 -test_data\
  &emsp;|--0000984.jpg\
  &emsp;&emsp;...\
 -test_data_video\
  &emsp;...\
 -test_predictions\
  &emsp;...
  
 -requirements.txt
 
|Folder | Description |
| --- | --- |
| Dataset | contains the dataset |
| outputs | contains the trained models, loss graphs and weights |
| src | contains the main python code files |
| test_data | contains images for inference |
| test_data_video | contains videos for inference |
| test_predictions | contains inference results |
  
## Prerequisite/Setup
1. Clone the repository
2. Download necessary libraries/modules/packages using `pip install -r requirements.txt`
3. cd to src folder  `cd src`
4. Put images in the test_data folder for inference
  4.1 If running inference by batch, file name is unnecessary
  4.2 If running inference on a single photo, filename must be `input_img.jpg` , resulting output will be `Out_img`
5. Put videos in the test_data_videos folder for inference
  5.1 Video file must be named `input_video.mp4`, resultng output will be an mp4 file called `output_video.mp4`
  
## Code Execution
If all files and input files have their correct filenames and are in their respective folders\
run:\
    `python inf_batch.py` if doing inference per batch of photos\
    `python inf_single.py` if doing inference on a single photo\
    `python inf_video.py` if doing inference on a video\
    
## Outputs and Graphs
<p align="center">
	<img src="https://user-images.githubusercontent.com/67114171/166144863-4332bb26-8f4b-4e99-823c-9a2e78a81a46.png">
	<br
	<b>Model train loss by iteration</b><br>
	<br><br>
	<img src ="https://user-images.githubusercontent.com/67114171/166145903-9ff2eb30-cee7-4298-abb0-4ab13d4270ae.jpg">
	<br>
	<b> Sample Image input </b>
	<br><br>
	<img src ="https://user-images.githubusercontent.com/67114171/166145928-6dd11e16-912d-4be4-a402-50fa89a8c24c.jpg">
	<br>
	<b> Sample Image Output </b>


</p>






