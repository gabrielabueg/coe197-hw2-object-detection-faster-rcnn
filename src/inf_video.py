import cv2
import mediapipe as mp
import glob as glob
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from model import create_model


def predict(image, model, device, detection_threshold):
	image = transform(image).to(device)
	image = torch.unsqueeze(image, 0)
	with torch.no_grad():
		outputs = model(image)
	pred_scores = outputs[0]['scores'].detach().cpu().numpy()
	pred_boxes = outputs[0]['boxes'].detach().cpu().numpy()
	boxes = pred_boxes[pred_scores >= detection_threshold].astype(np.int32)
	pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
	labels = outputs[0]['labels'][:len(boxes)]

	draw_boxes = boxes.copy()
	print(pred_scores)

	return boxes, pred_classes, labels, pred_scores



def draw_boxes(boxes, classes, labels, image, scores):
	image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
	for j, box in enumerate(boxes):
		color = COLORS[labels[j]]
		cv2.rectangle(image,
				(int(box[0]), int(box[1])),
				(int(box[2]), int(box[3])),
				color, 2)
		cv2.putText(image, classes[j], 
				(int(box[0]), int(box[1]-5)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
				2, lineType=cv2.LINE_AA)
		cv2.putText(image, str(scores[j]),
				(int(box[0]+5), int(box[1]+15)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 
	                        1)
	return image 





# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load the model and the trained weights
model = create_model(num_classes=4).to(device)
model.load_state_dict(torch.load(
    '../outputs/model.pth', map_location=device), strict=False)
model.eval()

COLORS = np.random.uniform(0, 255, size=(4, 3))
print('------------------------------',COLORS)
CLASSES = [
    'background', 'Pine Juice', 'Summit Water', 'Cola']

detection_threshold = 0.8


cap = cv2.VideoCapture("../test_data_video/input_video.mp4")

if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

transform = transforms.Compose([
    transforms.ToTensor(),
])


out = cv2.VideoWriter(f"../test_predictions/output_video.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))
frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        # get the start time
        start_time = time.time()
        with torch.no_grad():
        	boxes, classes, labels, scores = predict(frame, model, device, detection_threshold)
        # 	frame = transform(frame).to(device)
        # 	frame = torch.unsqueeze(frame, 0)
        # 	with torch.no_grad():
        # 		outputs = model(frame)

        # 	pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        # 	pred_boxes = outputs[0]['boxes'].detach().cpu().numpy()
        # 	boxes = pred_boxes[pred_scores >= detection_threshold].astype(np.int32)
        # 	pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        # 	labels = outputs[0]['labels'][:len(boxes)]

        # 	draw_boxes = boxes.copy()
        image = draw_boxes(boxes, classes, labels, frame, scores)

        # # draw boxes and show current frame on screen
        # for j, box in enumerate(draw_boxes):
        # 	color = COLORS[j]
        # 	cv2.rectangle(frame,
		      #       (int(box[0]), int(box[1])),
		      #       (int(box[2]), int(box[3])),
		      #       color, 2)
        # 	cv2.putText(frame, pred_classes[j], 
		      #           (int(box[0]), int(box[1]-5)),
		      #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
		      #           2, lineType=cv2.LINE_AA)
        # 	cv2.putText(frame, str(pred_scores[j]),
		      #           (int(box[0]+5), int(box[1]+15)),
		      #           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 
		      #                   1)


        # get the end time
        end_time = time.time()
        # get the fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1
        # write the FPS on the current frame
        # cv2.putText(image, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #             1, (0, 255, 0), 2)
        # convert from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('image', image)
        out.write(image)
        # press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")