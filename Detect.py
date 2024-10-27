#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[2]:


import os
import random
import pandas as pd
from PIL import Image
import cv2
from ultralytics import YOLO
from IPython.display import Video
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
import pathlib
import glob
from tqdm.notebook import trange, tqdm
import warnings
warnings.filterwarnings('ignore')
import ultralytics

# In[4]:


import os
import random
import matplotlib.pyplot as plt

Image_dir = r'C:\Users\HONOR\OneDrive\Desktop\Object detection\train\images'

num_samples = 9
image_files = os.listdir(Image_dir)

# Randomly select num_samples images
rand_images = random.sample(image_files, num_samples)

fig, axes = plt.subplots(3, 3, figsize=(11, 11))

for i in range(num_samples):
    image = rand_images[i]
    ax = axes[i // 3, i % 3]
    ax.imshow(plt.imread(os.path.join(Image_dir, image)))
    ax.set_title(f'Image {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.show()


# In[6]:


import cv2

# Get the size of the image
image = cv2.imread(r"C:\Users\HONOR\OneDrive\Desktop\Object detection\train\images\00000_00000_00007_png.rf.a57de5d1a4e7ac7afa168838fec08c3a.jpg")
h, w, c = image.shape
print(f"The image has dimensions {w}x{h} and {c} channels.")


# In[25]:


from ultralytics import YOLO

# Use a pretrained YOLOv8n model
model = YOLO("yolov8n.pt") 

# Use the model to detect object
image = r"C:\Users\HONOR\OneDrive\Desktop\Object detection\train\images\00001_00030_00005_png.rf.ffdae2af318eebb9d2b4b3a18dded057.jpg"
result_predict = model.predict(source = image, imgsz=(416))

# show results
plot = result_predict[0].plot()
plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)

# Show the image using OpenCV
cv2.imshow('Result', plot)
cv2.waitKey(0)
cv2.destroyAllWindows()



# In[1]:


from ultralytics import YOLO

# Build from YAML and transfer weights
Final_model = YOLO('yolov8n.yaml').load('yolov8n.pt')  

# Training The Final Model
try:
    Result_Final_model = Final_model.train(data=r"C:\Users\HONOR\OneDrive\Desktop\Object detection\data.yaml",
                                           epochs=2, imgsz=416, batch=2, lr0=0.0001, dropout=0.15, device='cpu')
except Exception as e:
    print(f"An error occurred during training: {e}")


# In[2]:


list_of_metrics = ["P_curve.png","R_curve.png","confusion_matrix.png"]


# In[9]:


import cv2
import matplotlib.pyplot as plt

# Assuming list_of_metrics is defined somewhere before this code snippet
list_of_metrics = ["P_curve.png", "R_curve.png", "confusion_matrix.png"]

for i in list_of_metrics:
    # Construct the file path using f-string
    image = cv2.imread(f'C:\\Users\\HONOR\\OneDrive\\Desktop\\Object detection\\runs\\detect\\train6\\{i}')

    # Check if image is loaded successfully
    if image is None:
        print(f'Error: Unable to load image {i}')
    else:
        # Create a larger figure
        plt.figure(figsize=(16, 12))

        # Display the image
        plt.imshow(image)

        # Show the plot
        plt.show()


# In[12]:


import pandas as pd

Result_Final_model = pd.read_csv(r'C:\Users\HONOR\OneDrive\Desktop\Object detection\runs\detect\train6\results.csv')
Result_Final_model.tail(10)


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set column names without whitespace
Result_Final_model.columns = Result_Final_model.columns.str.strip()

# Create subplots
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))

# Plot the columns using seaborn
sns.lineplot(x='epoch', y='train/box_loss', data=Result_Final_model, ax=axs[0,0])
sns.lineplot(x='epoch', y='train/cls_loss', data=Result_Final_model, ax=axs[0,1])
sns.lineplot(x='epoch', y='train/dfl_loss', data=Result_Final_model, ax=axs[1,0])
sns.lineplot(x='epoch', y='metrics/precision(B)', data=Result_Final_model, ax=axs[1,1])
sns.lineplot(x='epoch', y='metrics/recall(B)', data=Result_Final_model, ax=axs[2,0])
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=Result_Final_model, ax=axs[2,1])
sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=Result_Final_model, ax=axs[3,0])
sns.lineplot(x='epoch', y='val/box_loss', data=Result_Final_model, ax=axs[3,1])
sns.lineplot(x='epoch', y='val/cls_loss', data=Result_Final_model, ax=axs[4,0])
sns.lineplot(x='epoch', y='val/dfl_loss', data=Result_Final_model, ax=axs[4,1])

# Set titles and axis labels for each subplot
axs[0,0].set(title='Train Box Loss')
axs[0,1].set(title='Train Class Loss')
axs[1,0].set(title='Train DFL Loss')
axs[1,1].set(title='Metrics Precision (B)')
axs[2,0].set(title='Metrics Recall (B)')
axs[2,1].set(title='Metrics mAP50 (B)')
axs[3,0].set(title='Metrics mAP50-95 (B)')
axs[3,1].set(title='Validation Box Loss')
axs[4,0].set(title='Validation Class Loss')
axs[4,1].set(title='Validation DFL Loss')

# Set the main title
plt.suptitle('Training Metrics and Loss', fontsize=24)

# Adjust layout and show plot
plt.subplots_adjust(top=0.8)
plt.tight_layout()
plt.show()


# In[16]:


# Loading the best performing model
Valid_model = YOLO(r'C:\Users\HONOR\OneDrive\Desktop\Object detection\runs\detect\train6\weights\best.pt')

# Evaluating the model on the testset
metrics = Valid_model.val(split = 'test')


# In[17]:


# final results 
print("precision(B): ", metrics.results_dict["metrics/precision(B)"])
print("metrics/recall(B): ", metrics.results_dict["metrics/recall(B)"])
print("metrics/mAP50(B): ", metrics.results_dict["metrics/mAP50(B)"])
print("metrics/mAP50-95(B): ", metrics.results_dict["metrics/mAP50-95(B)"])


# In[20]:


import os
import random
import cv2

# Path to the directory containing the images
image_dir = r'C:\Users\HONOR\OneDrive\Desktop\Object detection\test\images'

# Get a list of all image files in the directory
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.jpg')]

# Randomly select 10 images from the directory
random_images = random.sample(image_files, k=10)

for image_path in random_images:
    image = cv2.imread(image_path)  # Replace with your preferred method of reading the image
    results = Final_model.predict([image], save=True, imgsz=416, conf=0.5, iou=0.7)
    #results.append(result)


# In[22]:


for i in range(2,12):
    plt.imshow(plt.imread(r'C:\Users\HONOR\OneDrive\Desktop\Object detection\runs\detect\train63\image0.jpg'))
    plt.show()


# In[24]:


from ultralytics import YOLO

# Load or initialize your video model
video_model = YOLO()  # Example initialization, replace with your actual video model initialization

# Export the model to ONNX format
video_model.export(format='onnx')


# In[37]:


import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# Load ONNX model
ort_session = ort.InferenceSession("yolov8n.onnx")

# Open video file
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define output video writer
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Process each frame
for _ in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    # (Replace this with your actual inference code)
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(r'C:\Users\HONOR\OneDrive\Desktop\Object detection\train\images', axis=0)
    output = ort_session.run(None, {"input": r'C:\Users\HONOR\OneDrive\Desktop\Object detection\train\images'})  # Replace "input" with your model input name

    # Overlay results on frame
    # (Replace this with your actual visualization code)
    processed_frame = frame  # Placeholder for processed frame

    # Write processed frame to output video
    out.write(processed_frame)

# Release resources
cap.release()
out.release()


# In[40]:


# Import the Video object
from IPython.display import Video

# Process the video using FFmpeg
get_ipython().system('ffmpeg -y -loglevel panic -i C:\\Users\\HONOR\\OneDrive\\Desktop\\Object detection\\video.mp4 output.mp4')

# Display the video
Video("output.mp4", width=960, embed=True)


# In[43]:


# Load a pr-trained model
video_model = YOLO("yolov8n.onnx")
 
# Use the model to detect signs
video_model.predict(source=r"C:\Users\HONOR\OneDrive\Desktop\Object detection\output.mp4", show=True, save = True)

