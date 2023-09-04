import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

# Download image
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
filename = 'images/child.jpg'

# Load a model
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# move to gpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image for large or small model
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# load image and apply transforms
img = cv2.imread(filename)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)

# Predict and resize
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=img.shape[:2],
                                                 mode="bicubic", align_corners=False,).squeeze()    
    

output = prediction.cpu().numpy()
ret, img_thresh = cv2.threshold(output, 500, 255, cv2.THRESH_BINARY)
img_thresh = img_thresh.astype(np.uint8)

img_final = cv2.bitwise_and(img, img, mask=img_thresh)

# Show result
# create figure
fig = plt.figure(figsize=(10, 7))
  
# setting values to rows and column variables
rows = 1
columns = 5

fig.add_subplot(rows, columns, 1)
plt.imshow(img) 

fig.add_subplot(rows, columns, 2)
plt.imshow(output)

fig.add_subplot(rows, columns, 3)
plt.imshow(img_gray, cmap='gray')

fig.add_subplot(rows, columns, 4)
plt.imshow(img_thresh)

fig.add_subplot(rows, columns, 5)
plt.imshow(img_final)

plt.show()

