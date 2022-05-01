import cv2
import mediapipe as mp
import glob as glob
import torch
import torchvision
import numpy as np
from model import create_model


# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load the model and the trained weights
model = create_model(num_classes=4).to(device)
model.load_state_dict(torch.load(
    '../outputs/model.pth', map_location=device), strict=False)
model.eval()


COLORS = np.random.uniform(0, 255, size=(4, 3))
CLASSES = [
    'background', 'Pine Juice', 'Summit Water', 'Cola']

detection_threshold = 0.8

test_images = "../test_data/input_img.jpg"
image_name = test_images.split('/')[-1].split('.')[0]
image = cv2.imread(test_images)
orig_image = image.copy()

# BGR to RGB
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
# make the pixel range between 0 and 1
image /= 255.0
# bring color channels to front
image = np.transpose(image, (2, 0, 1)).astype(np.float)
# convert to tensor
if torch.cuda.is_available():
    image = torch.tensor(image, dtype=torch.float).cuda()
else:
    image = torch.tensor(image, dtype=torch.float)
# add batch dimension
image = torch.unsqueeze(image, 0)
with torch.no_grad():
    outputs = model(image)

pred_scores = outputs[0]['scores'].detach().cpu().numpy()

pred_boxes = outputs[0]['boxes'].detach().cpu().numpy()

boxes = pred_boxes[pred_scores >= detection_threshold].astype(np.int32)

labels = outputs[0]['labels'][:len(boxes)]


draw_boxes = boxes.copy()

pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

for i, box in enumerate(boxes):
	print('Image Detected: ', pred_classes[i])
	print('Prediction Score: ', str(pred_scores[i]))

for j, box in enumerate(draw_boxes):
	color = COLORS[labels[j]]
	cv2.rectangle(orig_image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2)
	cv2.putText(orig_image, pred_classes[j], 
                (int(box[0]), int(box[1]-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                2, lineType=cv2.LINE_AA)
	cv2.putText(orig_image, str(pred_scores[j]),
                (int(box[0]+5), int(box[1]+15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 
                        1)

cv2.imshow('Image', orig_image)
cv2.imwrite(f"../test_predictions/Out_img.jpg", orig_image)
cv2.waitKey(0)