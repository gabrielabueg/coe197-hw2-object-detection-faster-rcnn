# coe197-hw2
**Homework 2 submission for CoE197Z
by John Gabriel Porton , 201803016

# Link/Reference
This homework is based on [Sovit Ranjan Raths Tutorial series on object detection with python and torchvision](https://debuggercafe.com/ssdlite-mobilenetv3-backbone-object-detection-with-pytorch-and-torchvision/)

# Data preparation
## Data Structure

> -Dataset
>  |--test
>     |--0000020.jpg
>      |--0000020.xml
>      ...
>  |--train
>      |--0000100.jpg
>      |--0000100.xml
>      ...
>      
> -outputs
>  |--model.pth
>  |--train_loss_10.png
>  |--valid_loss_10.png
>  ...
>
> -src
>  |--config.py
>  |--datasets.py
>  |--engine.py
>  |--inf_single.py
>  |--inf_video.py
>  |--inf_batch.py
>
> -test_data
>  |--0000984.jpg
>  ...
> -test_data_video
> 
> -test_predictions
>  ...
>  
> -requirements.txt

Dataset - contains the dataset
outputs - contains the trained models, loss graphs and weights
src - contains the main python code files
test_data - contains images for inference
test_data_video - contains videos for inference
test_predictions - contains inference results
  
## Prerequisite/Setup
1. Clone the repository
2. Download necessary libraries/modules/packages using 'pip install -r requirements.txt'
3. cd to src folder
4. Put images in the test_data folder for inference
  4.1 If running inference by batch, file name is unnecessary
  4.2 If running inference on a single photo, filename must be 'input_img.jpg', resulting output will be 'Out_img'
5. Put videos in the test_data_videos folder for inference
  5.1 Video file must be named 'input_video.mp4', resultng output will be an mp4 file called 'output_video.mp4'
  
## Code Execution
If all files and input files have their correct filenames and are in their respective folders
run:
    'python inf_batch.py' if doing inference per batch of photos
    'python inf_single.py' if doing inference on a single photo
    'python inf_video.py' if doing inference on a video
    
## Outputs and Graphs




