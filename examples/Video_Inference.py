#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision


# In[2]:


print(torch.__version__)
print(torchvision.__version__)


# In[3]:


# import matplotlib
# matplotlib.use('TkAgg')


# In[4]:


import matplotlib.pyplot as plt


# In[6]:


import sys

sys.path.append('..')
sys.path.append('../examples')


# # Data Loading

# First, let's take a look at how to load our dataset in the format that Yolov7 expects.

# ## Selecting a dataset

# Throughout this article, we shall use the [Kaggle cars object detection dataset](https://www.kaggle.com/datasets/sshikamaru/car-object-detection); however, as our aim is to demonstrate how Yolov7 can be applied to any problem, this is really the least important part of this work. Additionally, as the images are quite similar to COCO, it will enable us to experiment with a pretrained model before we do any training.
# 

# In[7]:


from pathlib import Path
import os
import pandas as pd


# In[8]:


data_path = "../data/papilla"
data_path = Path(data_path)
images_path = data_path / "training_images"
annotations_file_path = data_path / "annotations3.csv"


# The annotations for this dataset are in the form of a .csv file, which associates the image name with the corresponding annotations; where each row represents one bounding box. Whilst there are around 1000 images in the training set, only those with annotations are included in this file. 
# 
# We can view the format of this by loading it into a pandas DataFrame. 

# In[9]:


df= pd.read_csv(annotations_file_path).drop(columns='Unnamed: 0')

# As it is not usually the case that all images in our dataset contain instances of the objects that we are trying to detect, we would also like to include some images that do not contain cars. To do this, we can define a function to load the annotations which also includes 100 'negative' images. Additionally, as the designated test set is unlabelled, let's randomly take 20% of these images to use as our validation set. 

# In[11]:


import pandas as pd
import random

def load_cars_df(annotations_file_path, images_path):
    # all_images = sorted(set([p.parts[-1] for p in images_path.iterdir()]))
    image_id_to_image = {i: im for i, im in zip(df.image_id, df.image)}
    image_to_image_id = {v: k for k, v, in image_id_to_image.items()}
    
    class_id_to_label = dict(
        enumerate(df.query("has_annotation == True").class_name.unique())
    )
    class_label_to_id = {v: k for k, v in class_id_to_label.items()}

    # df["image_id"] = df.image.map(image_to_image_id)
    # df["class_id"] = df.class_name.map(class_label_to_id)

#     file_names = tuple(df.image.unique())
    
#     random.seed(42)
#     validation_files = set(random.sample(file_names, int(len(df) * 0.2)))
#     train_df = df[~df.image.isin(validation_files)]
#     valid_df = df[df.image.isin(validation_files)]
#     test_files= set(random.sample(file_names, int(len(df)* 10)))
#     test_df= df[~(df.image.isin(validation_files)) && ]
    
    from sklearn.model_selection import train_test_split

    # Split into train (70%), validation (30%), and test (10%)
    train_df, remaining_data = train_test_split(df, test_size=0.3, random_state=42)
    valid_df, test_df = train_test_split(remaining_data, test_size=0.1, random_state=42)

    
    lookups = {
        "image_id_to_image": image_id_to_image,
        "image_to_image_id": image_to_image_id,
        "class_id_to_label": class_id_to_label,
        "class_label_to_id": class_label_to_id,
    }
    return train_df, valid_df, test_df, lookups


# We can now use this function to load our data:

# In[12]:


train_df, valid_df, test_df, lookups = load_cars_df(annotations_file_path, images_path)


# In[13]:


train_df.head()


# In[14]:


valid_df.head()


# In[15]:


test_df.head()


# In[16]:


print(train_df.image.nunique(), valid_df.image.nunique(), test_df.image.nunique())


# In[17]:


# Splitting Validation into Val and Test


# To make it easier to associate predictions with an image, we have assigned each image a unique id; in this case it is just an incrementing integer count. Additionally, we have added an integer value to represent the classes that we want to detect, which is a single class - 'car' - in this case.
# 
# Generally, object detection models reserve `0` as the background class, so class labels should start from `1`. This is **not** the case for Yolov7, so we start our class encoding from `0`. For images that do not contain a car, we do not require a class id. We can confirm that this is the case by inspecting the lookups returned by our function.

# In[18]:


lookups.keys()


# In[19]:


lookups['class_label_to_id'], lookups['class_id_to_label']


# Finally, let's see the number of images in each class for our training and validation sets. As an image can have multiple annotations, we need to make sure that we account for this when calculating our counts:

# In[20]:


print(f"Num. annotated images in training set: {len(train_df.query('has_annotation == True').image.unique())}")
print(f"Num. Background images in training set: {len(train_df.query('has_annotation == False').image.unique())}")
print(f"Total Num. images in training set: {len(train_df.image.unique())}")
print('------------')

print(f"Num. annotated images in validation set: {len(valid_df.query('has_annotation == True').image.unique())}")
print(f"Num. Background images in validation set: {len(valid_df.query('has_annotation == False').image.unique())}")
print(f"Total Num. images in validation set: {len(valid_df.image.unique())}")


# ## Create a Dataset Adaptor

# Usually, at this point, we would create a PyTorch dataset specific to the model that we shall be training. 
# 
# However, we often use the pattern of first creating a dataset 'adaptor' class, with the sole responsibility of wrapping the underlying data sources and loading this appropriately. This way, we can easily switch out adaptors when using different datasets, without changing any pre-processing logic which is specific to the model that we are training.
# 
# Therefore, let’s focus for now on creating a `CarsDatasetAdaptor` class, which converts the specific raw dataset format into an image and corresponding annotations. Additionally, let's load the image id that we assigned, as well as the height and width of our image, as they may be useful to us later on.
# 
# An implementation of this is presented below:

# In[21]:


from train_cars import CarsDatasetAdaptor


# Notice that, for our background images, we are just returning an empty array for our bounding boxes and class ids.

# Using this, we can confirm that the length of our dataset is the same as the total number of training images that we calculated earlier.

# In[22]:


train_ds = CarsDatasetAdaptor(images_path, train_df)
valid_ds= CarsDatasetAdaptor(images_path, valid_df)
test_ds= CarsDatasetAdaptor(images_path, test_df)


# In[23]:


train_ds


# Now, we can use this to visualise some of our images, as demonstrated below.

# In[24]:


from yolov7.plotting import show_image


# Let's wrap our data adaptor using this dataset and inspect some of the outputs:

# ### Transforms

# In[25]:


from yolov7.dataset import Yolov7Dataset
from yolov7.dataset import create_yolov7_transforms


# In[26]:


target_image_size = 640


# In[27]:


train_yds = Yolov7Dataset(train_ds, transforms=create_yolov7_transforms(image_size=(target_image_size, target_image_size)))
eval_yds= Yolov7Dataset(valid_ds, transforms=create_yolov7_transforms(image_size=(target_image_size, target_image_size)))
test_yds= Yolov7Dataset(test_ds, transforms=create_yolov7_transforms(image_size=(target_image_size, target_image_size)))


# Using these transforms, we can see that our image has been resized to our target size and padding has been applied. The reason that padding is used is so that we can maintain the aspect ratio of the objects in the images, but have a common size for images in our dataset; enabling us to batch them efficiently!

# ### Run Training

# In[28]:


from yolov7.trainer import Yolov7Trainer



# In[30]:


import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from func_to_script import script
from PIL import Image
from pytorch_accelerated.callbacks import (
    EarlyStoppingCallback,
    SaveBestModelCallback,
    get_default_callbacks,
)
from pytorch_accelerated.schedulers import CosineLrScheduler
from torch.utils.data import Dataset

from yolov7 import create_yolov7_model
from yolov7.dataset import Yolov7Dataset, create_yolov7_transforms, yolov7_collate_fn
from yolov7.evaluation import CalculateMeanAveragePrecisionCallback
from yolov7.loss_factory import create_yolov7_loss
from yolov7.trainer import Yolov7Trainer, filter_eval_predictions


# In[31]:


# Defining model 
best_model = create_yolov7_model('yolov7', num_classes=1, pretrained=True)
best_model.eval();


# In[32]:


# Loading Weights
best_model_path= '/Users/longquanchen/Desktop/Work/Versatile/Yolov7-training-main/examples/v7_annotations_finetune.pt'
checkpoint = torch.load(best_model_path, map_location='cpu')
state_dict = checkpoint['model_state_dict']
best_model.load_state_dict(state_dict)


# ## Running inference on Video:

# In[33]:


my_test_df= pd.read_csv("/Users/longquanchen/Desktop/Work/Versatile/Yolov7-training-main/data/papilla/annotations_testing.csv").drop(columns='Unnamed: 0').reset_index().drop(columns='index')


# In[34]:


my_test_df['image_id']= my_test_df['image'].apply(lambda x: int(x.split('.')[0]))


# In[35]:


final_df = my_test_df.sort_values(by='image_id')


# In[36]:


final_df


# In[37]:


os.getcwd()


# In[38]:


my_test_ds= CarsDatasetAdaptor("/Users/longquanchen/Desktop/Work/Versatile/Yolov7-training-main/data/papilla/test_video_frames", final_df)


# In[39]:


my_test_yds= Yolov7Dataset(my_test_ds, transforms=create_yolov7_transforms(image_size=(target_image_size, target_image_size)))


# In[40]:


image_tensor, labels, image_id, image_size = test_yds[0]


# In[41]:


#List of image tensors
image_tensor_collection= []
startIndex = 500
for i in range(startIndex,startIndex+200): # 874 maximum
    image_tensor, labels, image_id, image_size = my_test_yds[i]
    image_tensor_collection.append(image_tensor)


# In[42]:


import cv2
import torch

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 640))
# Loop over the image tensors
for image_tensor in image_tensor_collection:
    # Perform inference on the image tensor
    with torch.no_grad():
        model_outputs = best_model(image_tensor[None])
        # Postprocess the output to get the predictions
        preds = best_model.postprocess(model_outputs, conf_thres=0., multiple_labels_per_box=False)

    # Filter the predictions using NMS and a confidence threshold
    nms_predictions = filter_eval_predictions(preds, confidence_threshold=0.1)

    if len(nms_predictions[0])>1:
        nms_predictions= [nms_predictions[0][1].reshape(1,6)]
        
    # Get the predicted boxes
    pred_boxes = nms_predictions[0][:, :4].cpu().numpy()

    # Load the image as a NumPy array
    img = image_tensor.permute(1, 2, 0).cpu().numpy()

    # Scale pixel values to range of 0 to 255
    img = (img * 255).astype(np.uint8)# This was an important step, because all pixel values 
    #in the image_tensor were normalized so we need to scale it up first and then convert the format into uint8 
    #(uint8 format is important for the writing into video file)
    
    # Resizing just to make sure
    img= cv2.resize(img, (640, 640))

    # Convert color space to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    
    # Draw the predicted boxes on the image
    for box in pred_boxes:
        xmin, ymin, xmax, ymax = box.astype(int)
        print(xmin, ymin, xmax, ymax )
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    ## Visualization
#     window_name = 'image'
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # Show the image with the predicted boxes
#     cv2.imshow(window_name, img)
#     cv2.waitKey(0)

    # Write frame to video
    #out.write(img)
    cv2.imwrite('testFrame' + str(startIndex) + '.png', img)
    startIndex= startIndex + 1


# Release the video writer and close all windows
out.release()
cv2.destroyAllWindows()


# In[43]:


# import cv2
# import torch

# with torch.no_grad():
#     # Loop over the image tensors
#     for image_tensor in image_tensor_collection:
#         # Perform inference on the image tensor
#         model_outputs = best_model(image_tensor[None])
        
#         # Postprocess the output to get the predictions
#         preds = best_model.postprocess(model_outputs, conf_thres=0., multiple_labels_per_box=False)
        
#         # Filter the predictions using NMS and a confidence threshold
#         nms_predictions = filter_eval_predictions(preds, confidence_threshold=0.1)
                
#         if len(nms_predictions[0])>1:
#             nms_predictions= [nms_predictions[0][1].reshape(1,6)]
#         # Get the predicted boxes
#         pred_boxes = nms_predictions[0][:, :4].cpu().numpy()
        
#         # Load the image as a NumPy array
#         img = image_tensor.permute(1, 2, 0).cpu().numpy()
        
#         # Convert BGR to RGB
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # Draw the predicted boxes on the image
#         for box in pred_boxes:
#             xmin, ymin, xmax, ymax = box.astype(int)
#             cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
#         window_name = 'image'
#         # Show the image with the predicted boxes
#         cv2.imshow(window_name, img)
#         cv2.waitKey(0)
        
# #         break
#     # Close all windows
#     cv2.destroyAllWindows()


# In[44]:


# import cv2
# import torch

# with torch.no_grad():
#     # Loop over the image tensors
#     for image_tensor in image_tensor_collection:
#         # Perform inference on the image tensor
#         model_outputs = best_model(image_tensor[None])
        
#         # Postprocess the output to get the predictions
#         preds = best_model.postprocess(model_outputs, conf_thres=0., multiple_labels_per_box=False)
        
#         # Filter the predictions using NMS and a confidence threshold
#         nms_predictions = filter_eval_predictions(preds, confidence_threshold=0.1)
                
#         if len(nms_predictions[0])>1:
#             nms_predictions= [nms_predictions[0][1].reshape(1,6)]
#         # Get the predicted boxes
#         pred_boxes = nms_predictions[0][:, :4].cpu().numpy()
        
#         # Load the image as a NumPy array
#         img = image_tensor.permute(1, 2, 0).cpu().numpy()
        
#         # Convert BGR to RGB
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # Draw the predicted boxes on the image
#         for box in pred_boxes:
#             xmin, ymin, xmax, ymax = box.astype(int)
#             cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
#         window_name = 'image'
#         # Show the image with the predicted boxes
#         cv2.imshow(window_name, img)
#         cv2.waitKey(0)
        
#         break
#     # Close all windows
#     cv2.destroyAllWindows()


# In[45]:


# import cv2
# import torch

# # Set up video writer
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))

# with torch.no_grad():
#     # Loop over the image tensors
#     for image_tensor in image_tensor_collection:
#         # Perform inference on the image tensor
#         model_outputs = best_model(image_tensor[None])
        
#         # Postprocess the output to get the predictions
#         preds = best_model.postprocess(model_outputs, conf_thres=0., multiple_labels_per_box=False)
        
#         # Filter the predictions using NMS and a confidence threshold
#         nms_predictions = filter_eval_predictions(preds, confidence_threshold=0.1)
                
#         if len(nms_predictions[0])>1:
#             nms_predictions= [nms_predictions[0][1].reshape(1,6)]
#         # Get the predicted boxes
#         pred_boxes = nms_predictions[0][:, :4].cpu().numpy()
        
#         # Load the image as a NumPy array
#         img = image_tensor.permute(1, 2, 0).cpu().numpy()
        
#         # Draw the predicted boxes on the image
#         for box in pred_boxes:
#             xmin, ymin, xmax, ymax = box.astype(int)
#             cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
#         # Write frame to video
#         out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

#     # Release the video writer and close all windows
#     out.release()
#     cv2.destroyAllWindows()


# In[46]:


# import cv2
# # # Create a simple image with a red rectangle
# img1 = np.zeros((640, 640, 3), dtype=np.uint8)
# img1[:, :, 0] = 255
# cv2.rectangle(img1, (100, 100), (200, 200), (0, 0, 255), 2)

# # Set up video writer
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 640))

# # # Write frame to video
# out.write(img1)

# # # Release the video writer and close all windows
# out.release()
# cv2.destroyAllWindows()


# In[47]:


# import cv2
# # import numpy as np

# # # Create a simple image with a red rectangle
# img = image_tensor.permute(1, 2, 0).cpu().numpy()

# # Scale pixel values to range of 0 to 255
# img = (img * 255).astype(np.uint8)

# img= cv2.resize(img, (640, 640))
# cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 255), 2)

# # Convert color space to BGR
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# # Set up video writer
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output_trial11.mp4', fourcc, 30.0, (640, 640))


# # # Write frame to video
# out.write(img)

# # # Release the video writer and close all windows
# out.release()
# cv2.destroyAllWindows()


# In[ ]:




