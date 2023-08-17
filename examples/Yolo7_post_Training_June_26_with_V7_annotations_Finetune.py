#!/usr/bin/env python
# coding: utf-8
import torch
import torchvision

print(torch.__version__)
print(torchvision.__version__)

import matplotlib.pyplot as plt

import sys

sys.path.append('..')
sys.path.append('../examples')

from pathlib import Path
import os
import pandas as pd

data_path = "../data/papilla"
data_path = Path(data_path)
images_paths = [os.path.join(data_path, "anatomical-model-papilla-860x860-1", "anatomical_model_papilla_1"), os.path.join(data_path, "anatomical-model-papilla-860x860-2", "anatomical_model_papilla_2")] # all_images
annotations_file_paths = [os.path.join(data_path,"anatomical-model-papilla-860x860-1", "anatomical-model-papilla-860x860-1.json"), os.path.join(data_path, "anatomical-model-papilla-860x860-2", "anatomical-model-papilla-860x860-2.json")] # annotations_final.csv

import json

#Creating dataframe
xmin = 0  # example xmin value
ymin = 0  # example ymin value
xmax = 860  # example xmax value
ymax = 860  # example ymax value
my_df= {'image':[], 'xmin':[],'ymin':[], 'xmax':[], 'ymax':[], 'class_name':[],
        'has_annotation':[], 'image_id':[], 'class_id':[] }



def load_cars_df(df):
    # all_images = sorted(set([p.parts[-1] for p in images_path.iterdir()]))
    image_id_to_image = {i: im for i, im in zip(df.image_id, df.image)}
    image_to_image_id = {v: k for k, v, in image_id_to_image.items()}

    class_id_to_label = dict(
        enumerate(df.query("has_annotation == True").class_name.unique())
    )
    class_label_to_id = {v: k for k, v in class_id_to_label.items()}

    from sklearn.model_selection import train_test_split
    # first, split into X_train, X_valid_test, y_train, y_valid_test
    # `test_size=0.3` split into 70% and 30%
    train_df, valid_test_df = train_test_split(df, test_size=0.3, random_state=42)

    # second, split into X_valid, X_test, y_valid, y_test
    # `test_size=0.5` split into 50% and 50%. The original data set is 30%,
    # so, it will split into 15% equally.
    valid_df, test_df = train_test_split(valid_test_df, test_size=0.5, random_state=42)

    lookups = {
        "image_id_to_image": image_id_to_image,
        "image_to_image_id": image_to_image_id,
        "class_id_to_label": class_id_to_label,
        "class_label_to_id": class_label_to_id,
    }
    return train_df, valid_df, test_df, lookups

for i_sub in range(len(images_paths)):
  annotations_file_path = annotations_file_paths[i_sub]
  images_path = images_paths[i_sub]
  with open(annotations_file_path, 'r') as f:
          data = json.load(f)

  # REmoving segmentation, area, iscrowd, extra keys from the annotations:
  keys_to_remove = ['segmentation', 'area', 'iscrowd', 'extra']

  for item in data['annotations']:
      for key in keys_to_remove:
          if key in item:
              del item[key]

  data['annotations']

  import re
  import numpy as np
  annotations= data['annotations']

  # Iterate over the images in different scenario
  final_train_df = pd.DataFrame()
  final_valid_df = pd.DataFrame()
  final_test_df = pd.DataFrame()
  lookups = {}
  for image_filename in os.listdir(images_path):
      #print(image_filename)
      # Find the annotation for the current image filename
      new_image_filename = int(image_filename.split('.')[0])
  #     mayching_data = next((annotation for annotation in annotations if annotation['image_id'] == new_image_filename), None)
      matching_data= [d for d in data['annotations'] if d['image_id']== new_image_filename]
      ## Inputting data into dictionary for the dataframe building
      my_df['image'].append(os.path.join(images_path,image_filename))
      my_df['image_id'].append(np.nan)

  #     print(matching_data)

      if len(matching_data)==0:

          my_df['class_name'].append('background')
          my_df['class_id'].append(np.nan)
          my_df['has_annotation'].append(False)## Im assuming if its a background (category id 168) there is no annotation

          my_df['xmin'].append(np.nan)
          my_df['ymin'].append(np.nan)
          my_df['xmax'].append(np.nan)
          my_df['ymax'].append(np.nan)


      else:

          original_bbox= matching_data[0]['bbox']# The bbox is in the format [xmin, ymin, w, h]

          crop_width = xmax - xmin
          crop_height = ymax - ymin
          # Converting bbox according to the cropped image and converting the format to [xmin, ymin, xmax, ymax]
          new_xmin = max(0, original_bbox[0] - xmin)
          new_ymin = max(0, original_bbox[1] - ymin)
          new_xmax = min(crop_width, original_bbox[0] + original_bbox[2] - xmin)
          new_ymax = min(crop_height, original_bbox[1] + original_bbox[3] - ymin)

  #         print(bbox)
          my_df['class_name'].append('papilla')
          my_df['class_id'].append(0.0)
          my_df['has_annotation'].append(True)

          my_df['xmin'].append(new_xmin)
          my_df['ymin'].append(new_ymin)
          my_df['xmax'].append(new_xmax)# The bounding boxes are in the format x,y,w,g, from coco annotater
          my_df['ymax'].append(new_ymax)# Hence converting w & h to xmax, ymax
  df = pd.DataFrame.from_dict(my_df)
  df.image_id = list(range(1, len(df) + 1))
  train_df, valid_df, test_df, lookups = load_cars_df(df)
  final_train_df = pd.concat([final_train_df, train_df])
  final_valid_df = pd.concat([final_valid_df, valid_df])
  final_test_df = pd.concat([final_test_df, test_df])

temp_df= final_train_df.iloc[0:15]
temp_df

"""
## Visualize Images from the Dataset"""

from PIL import Image, ImageDraw
for i in temp_df.index.tolist():
    my_img_path= temp_df.image[i]
    # Load the image
#     image_path = 'path/to/your/image.jpg'
    image = Image.open(my_img_path)

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Define the BBox coordinates
    xmin, ymin, xmax, ymax = temp_df['xmin'][i], temp_df['ymin'][i], temp_df['xmax'][i], temp_df['ymax'][i]

    print(my_img_path)
    print(xmin, ymin, xmax, ymax)
    # Draw the bounding box rectangle
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=2)

    # Show or save the image
    image.show()
    #image.save('/content/drive/MyDrive/2023/Datasets/Anatomical_Model/image_with_bounding_box.jpg')
    break


final_train_df.head()

final_valid_df.head()

final_test_df.head()

print(final_train_df.image.nunique(), final_valid_df.image.nunique(), final_test_df.image.nunique())

# final_lookups.keys()
#
# final_lookups['class_label_to_id'], lookups['class_id_to_label']


# Finally, let's see the number of images in each class for our training and validation sets. As an image can have multiple annotations, we need to make sure that we account for this when calculating our counts:


print(f"Num. annotated images in training set: {len(train_df.query('has_annotation == True').image.unique())}")
print(f"Num. Background images in training set: {len(train_df.query('has_annotation == False').image.unique())}")
print(f"Total Num. images in training set: {len(train_df.image.unique())}")
print('------------')

print(f"Num. annotated images in validation set: {len(valid_df.query('has_annotation == True').image.unique())}")
print(f"Num. Background images in validation set: {len(valid_df.query('has_annotation == False').image.unique())}")
print(f"Total Num. images in validation set: {len(valid_df.image.unique())}")



from train_cars import CarsDatasetAdaptor

# Notice that, for our background images, we are just returning an empty array for our bounding boxes and class ids.

# Using this, we can confirm that the length of our dataset is the same as the total number of training images that we calculated earlier.

# Make sure to plug in right variable for the path(images)
train_ds = CarsDatasetAdaptor(final_train_df)
valid_ds= CarsDatasetAdaptor(final_valid_df)
test_ds= CarsDatasetAdaptor(final_test_df)

from yolov7.plotting import show_image


# Let's wrap our data adaptor using this dataset and inspect some of the outputs:

# ### Transforms

from yolov7.dataset import Yolov7Dataset
from yolov7.dataset import create_yolov7_transforms


target_image_size = 640


train_yds = Yolov7Dataset(train_ds, transforms=create_yolov7_transforms(image_size=(target_image_size, target_image_size)))
eval_yds= Yolov7Dataset(valid_ds, transforms=create_yolov7_transforms(image_size=(target_image_size, target_image_size)))
test_yds= Yolov7Dataset(test_ds, transforms=create_yolov7_transforms(image_size=(target_image_size, target_image_size)))

idx = 3

image_tensor, labels, image_id, image_size = eval_yds[idx]

print(f'Image: {image_tensor.shape}')
print(f'Labels: {labels}')

# denormalize boxes
boxes = labels[:, 2:]
boxes[:, [0, 2]] *= target_image_size #image_size[1]## Multiplying with targetimagesize becasue, padding was applied (using transforms above)
boxes[:, [1, 3]] *= target_image_size #image_size[0]

show_image(image_tensor.permute( 1, 2, 0), boxes.tolist(), [lookups['class_id_to_label'][int(c)] for c in labels[:, 1]], 'cxcywh')
plt.show()
print(f'Image id: {image_id}')
print(f'Image size: {image_size}')

from yolov7.trainer import Yolov7Trainer

import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from func_to_script import script
from PIL import Image
from pytorch_accelerated.callbacks import (
    ModelEmaCallback,
    ProgressBarCallback,
    SaveBestModelCallback,
    get_default_callbacks,
    EarlyStoppingCallback
)
from pytorch_accelerated.schedulers import CosineLrScheduler
from torch.utils.data import Dataset

from yolov7 import create_yolov7_model
from yolov7.dataset import (
    Yolov7Dataset,
    create_base_transforms,
    create_yolov7_transforms,
    yolov7_collate_fn,
)
from yolov7.evaluation import CalculateMeanAveragePrecisionCallback
from yolov7.loss_factory import create_yolov7_loss
from yolov7.mosaic import MosaicMixupDataset, create_post_mosaic_transform
from yolov7.trainer import Yolov7Trainer, filter_eval_predictions
from yolov7.utils import SaveBatchesCallback, Yolov7ModelEma

DATA_PATH =data_path

def finetune_training(
    train_ds, valid_ds,
    data_path: str = DATA_PATH,
    image_size: int = 640,
    pretrained: bool = True,
    num_epochs: int = 50,
    batch_size: int = 16,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',# Add this line to set the device to use
):


##############################################################################
    # CHecking on which device the training is going to run
    print(device)
    #Code added by Mike: print name of GPU
    #print(torch.cuda.get_device_name(torch.cuda.current_device()))

##############################################################################

##############################################################################

    num_classes= 1

##############################################################################

    train_yds = Yolov7Dataset(
        train_ds,
        create_yolov7_transforms(training=True, image_size=(image_size, image_size)),
    )
    eval_yds = Yolov7Dataset(
        valid_ds,
        create_yolov7_transforms(training=False, image_size=(image_size, image_size)),
    )
##############################################################################

    # Create model, loss function and optimizer
    model = create_yolov7_model(
        architecture="yolov7-tiny", num_classes=num_classes, pretrained=pretrained, pretrainedWeights="../examples/best_model_1.pt"
    ).to(device) # Dheeraj added .to(torch.device("cpu"))

##############################################################################

    loss_func = create_yolov7_loss(model, image_size=image_size)

##############################################################################

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, nesterov=True
    )

##############################################################################

    # create evaluation callback and trainer
    calculate_map_callback = (
        CalculateMeanAveragePrecisionCallback.create_from_targets_df(
            targets_df=valid_df.query("has_annotation == True")[
                ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id"]
            ],
            image_ids=set(valid_df.image_id.unique()),
            iou_threshold=0.2,
        )
    )

##############################################################################
    # Create trainer and train
    trainer = Yolov7Trainer(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        filter_eval_predictions_fn=partial(
            filter_eval_predictions, confidence_threshold=0.01, nms_threshold=0.3
        ),
        callbacks=[
            calculate_map_callback,
            SaveBestModelCallback(watch_metric="eval_loss_epoch", greater_is_better=False),
            EarlyStoppingCallback(
                early_stopping_patience=10,
                watch_metric="eval_loss_epoch",
                greater_is_better=False,
                early_stopping_threshold=0.001,
            ),
            *get_default_callbacks(progress_bar=True),
        ],
    )

##############################################################################


    trainer.train(
        num_epochs=num_epochs,
        train_dataset=train_yds,
        eval_dataset=eval_yds,
        per_device_batch_size=batch_size,
        create_scheduler_fn=CosineLrScheduler.create_scheduler_fn(
            num_warmup_epochs=5,
            num_cooldown_epochs=5,
            k_decay=2,
        ),
        collate_fn=yolov7_collate_fn,
    )

finetune_training(train_ds, valid_ds)

# The notable differences between the "Training from Scratch" and "Fine-tuning" scenarios in the provided script:
# 
# Fine-tuning the model:
# 
# Dataset: The script defines train_yds and eval_yds using the Yolov7Dataset and create_yolov7_transforms functions for both training and evaluation datasets.
# 
# Optimizer: The optimizer is created using model.parameters() instead of model.get_parameter_groups().
# Early Stopping: The script includes the EarlyStoppingCallback in the list of trainer callbacks, which helps stop training early if the metric does not improve.
# 
# No Gradient Accumulation Steps: Unlike the "Training from Scratch" script, there is no calculation or mention of gradient accumulation steps in the "Fine-tuning" script.

# ### Calculating mAP, Precision and Recall on Test set

# ##### I have created my own code for calculation of mAP, Precison, Recall, and Precision-Recall Curve
# ##### But this might not be necessary if we can use the functions provided in the evaluation folder correctly
# ##### Path for evaluation folder: C:\Users\endo\Desktop\Yolov7-training-main\Yolov7-training-main\yolov7\evaluation
# ##### Some idea on how to use it can be learnt from the above cell (Training Code), where the callbacks for calcualtion mAP were used

# In[40]:


# Defining model 
best_model = create_yolov7_model('yolov7-tiny', num_classes=1)
best_model.eval();


# In[41]:


# Loading Weights
# Remember to change the .pt file as per the trained weights you have or want to test with
best_model_path= 'C:\\Users\\endo\\Desktop\\Yolov7-training-main\\Yolov7-training-main\\examples\\v7_annotations_finetune_tiny.pt'
checkpoint = torch.load(best_model_path)
state_dict = checkpoint['model_state_dict']
best_model.load_state_dict(state_dict)


# ## Running inference on test_yds

# In[42]:


import torch
for idx in range(min(10,len(test_yds))):
    image_tensor, labels, image_id, image_size = test_yds[idx]
    with torch.no_grad():
        model_outputs = best_model(image_tensor[None])
        preds = best_model.postprocess(model_outputs, conf_thres=0., multiple_labels_per_box=False)

        # Inference
        nms_predictions = filter_eval_predictions(preds, confidence_threshold=0.1)
        nms_predictions[0].shape
        pred_boxes = nms_predictions[0][:, :4]
        class_ids = nms_predictions[0][:, -1]

        show_image(image_tensor.permute( 1, 2, 0), pred_boxes.tolist(), class_ids.tolist())
        plt.show()
        print(preds)
#         print(f'Image id: {image_id}')
#         print(f'Original Image size: {image_size}')
#         print(f'Resized Image size: {image_tensor.shape[1:]}')


# In[ ]:





# In[44]:


def intersection_over_union(boxes_preds, boxes_labels, box_format= "corners"):
    # (N,4): N--number of bboxes
    # boxes_labels shape is (N,4)
    
    '''Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
        '''       
    if box_format == "corners":
        
        #Converting cx,cy,w,h (center_x, center_y,w,h) into (xmin, ymin, xmax, ymax)
        
#         print(boxes_labels)
#         print(boxes_preds)
        
        center_x= boxes_labels[..., 0:1]
        center_y = boxes_labels[..., 1:2] 
        width = boxes_labels[..., 2:3] 
        height= boxes_labels[..., 3:4] 
        new_boxes_labels = torch.zeros_like(boxes_labels) # Initializing the tensor

        new_boxes_labels[..., 0:1]= center_x - (width / 2)
        new_boxes_labels[..., 1:2]= center_y - (height / 2)
        new_boxes_labels[..., 2:3]= center_x + (width / 2)
        new_boxes_labels[..., 3:4]= center_y + (height / 2)        
        
        
        pred_xmin= boxes_preds[..., 0:1]
        pred_ymin= boxes_preds[..., 1:2]
        pred_xmax= boxes_preds[..., 2:3]
        pred_ymax= boxes_preds[..., 3:4]

        label_xmin= new_boxes_labels[..., 0:1]
        label_ymin= new_boxes_labels[..., 1:2]
        label_xmax= new_boxes_labels[..., 2:3]
        label_ymax= new_boxes_labels[..., 3:4]

    inter_area = max(0, min(pred_xmax, label_xmax) - max(pred_xmin, label_xmin)) *              max(0, min(pred_ymax, label_ymax) - max(pred_ymin, label_ymin))

    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    label_area = (label_xmax - label_xmin) * (label_ymax - label_ymin)
    union_area = pred_area + label_area - inter_area

    iou = inter_area / union_area
    conf= boxes_preds[...,4:5]
    return iou, conf
        


# In[45]:


# Validation Image
df_list=[]
iou_thres= 0.5
for i in range(len(test_df)):
    
    # Taking a test image
    image_tensor, labels, image_id, image_size = test_yds[i]
    boxes_labels = labels[:, 2:]
    boxes_labels[:, [0, 2]] *= target_image_size
    boxes_labels[:, [1, 3]] *= target_image_size
    
    # Predicting 
    with torch.no_grad():
        model_outputs = best_model(image_tensor[None])
        preds = best_model.postprocess(model_outputs, conf_thres=0., multiple_labels_per_box=False)  
        
    # This has the bounding boxes with confidence and class label
    nms_predictions = filter_eval_predictions(preds, confidence_threshold=0.1)
    data= {'image_id':[],'gt_flag':[],'pd_flag':[], 'confidence':[], 'iou':[], 'tp':[], 'fp':[], 'fn':[], 'tn':[]}
    
    
    # Chec if the Ground Truth Bbox is available:
    if boxes_labels.numel()==0:
        # Now check if Predicted Bounding boxes are zero or any got predicted:
        if nms_predictions[0].numel()==0:
#             print(" We dont care about this case")
            data['image_id']= image_id.tolist()
            data['gt_flag'].append(1)
            data['pd_flag'].append(1)
            data['confidence'].append(np.nan)
            data['iou'].append(np.nan)
            data['tp'].append(np.nan)
            data['fn'].append(np.nan)
            data['fp'].append(np.nan)
            data['tn'].append(1)
        elif nms_predictions[0].numel()!=0:
            data['image_id']= image_id.tolist()
            data['gt_flag'].append(1)
            data['pd_flag'].append(0)
            data['confidence'].append(np.nan)
            data['iou'].append(np.nan)
            data['tp'].append(np.nan)
            data['fn'].append(np.nan)
            data['fp'].append(1)
            data['tn'].append(np.nan)
    # this is when the ground truth bounding box is available:
    else:
        # Now we chck if the prediction are done or not:
        # Checking if there are no predictions:
        if nms_predictions[0].numel()==0:
            data['image_id']= image_id.tolist()
            data['gt_flag'].append(0)
            data['pd_flag'].append(1)
            data['confidence'].append(np.nan)
            data['iou'].append(np.nan)
            data['tp'].append(np.nan)
            data['fn'].append(1)
            data['fp'].append(np.nan)
            data['tn'].append(np.nan)
        # Checking ig the prediction is done
        elif nms_predictions[0].numel()!=0:
            # In this case we check for IOU:
            # First we check of the number of predictions are 1 or more:
#             my_iou_list=[]
#             confidence= []
            
            if len(nms_predictions[0])>1:
                for j in range(len(nms_predictions[0])):
                    my_iou, conf= intersection_over_union(nms_predictions[0][j], boxes_labels, box_format= "corners")
#                     print(conf)
                    if my_iou> iou_thres:
                        data['image_id']= image_id.tolist()
                        data['gt_flag'].append(0)
                        data['pd_flag'].append(0)
                        data['confidence'].append(conf[0].tolist())
                        data['iou'].append(my_iou[0][0].tolist())
                        data['tp'].append(1)
                        data['fn'].append(np.nan)
                        data['fp'].append(np.nan)
                        data['tn'].append(np.nan)
                    else:
                        data['image_id']= image_id.tolist()
                        data['gt_flag'].append(0)
                        data['pd_flag'].append(0)
                        data['confidence'].append(conf[0].tolist())
                        data['iou'].append(my_iou[0][0].tolist())
                        data['tp'].append(np.nan)
                        data['fn'].append(np.nan)
                        data['fp'].append(1)
                        data['tn'].append(np.nan)
                        
#                     my_iou_list.append(my_iou)
#                     confidence.append(conf)
                
            else:
                
                my_iou, conf= intersection_over_union(nms_predictions[0], boxes_labels, box_format= "corners")
                
#                 print(conf)
                
                if my_iou> iou_thres:
                    data['image_id']= image_id.tolist()
                    data['gt_flag'].append(0)
                    data['pd_flag'].append(0)
                    data['confidence'].append(conf[0][0].tolist())
                    data['iou'].append(my_iou[0][0].tolist())
                    data['tp'].append(1)
                    data['fn'].append(np.nan)
                    data['fp'].append(np.nan)
                    data['tn'].append(np.nan)
                else:
                    data['image_id']= image_id.tolist()
                    data['gt_flag'].append(0)
                    data['pd_flag'].append(0)
                    data['confidence'].append(conf[0][0].tolist())
                    data['iou'].append(my_iou[0][0].tolist())
                    data['tp'].append(np.nan)
                    data['fn'].append(np.nan)
                    data['fp'].append(1)
                    data['tn'].append(np.nan)
#                 my_iou_list.append(my_iou)
#                 confidence.append(conf)

    df= pd.DataFrame(data)
#     display(df)
    df_list.append(df)
#     break


# In[47]:


final_df= pd.concat(df_list).reset_index().drop(columns='index')


# In[48]:


final_df


# In[49]:


final_df_sort= final_df.sort_values('confidence', ascending=False).reset_index().drop(columns='index')


# In[50]:


final_df_sort['tp_fp_fn'] = np.where((final_df_sort['tp'] ==1.0) , 'TP', 
                        np.where((final_df_sort['fp'] ==1.0) , 'FP', 
                                 np.where((final_df_sort['tn'] ==1.0) , 'TN',
                                          np.where((final_df_sort['fn'] ==1.0) , 'FN', np.nan))))

###############################################

final_df_sort['tp'] = final_df_sort['tp_fp_fn'].apply(lambda x: 1 if x == 'TP' else 0)
final_df_sort['fp'] = final_df_sort['tp_fp_fn'].apply(lambda x: 1 if x == 'FP' else 0)
final_df_sort['fn'] = final_df_sort['tp_fp_fn'].apply(lambda x: 1 if x == 'FN' else 0)
final_df_sort['tn'] = final_df_sort['tp_fp_fn'].apply(lambda x: 1 if x == 'TN' else 0)


# In[51]:


final_df_sort


# In[52]:


from sklearn.metrics import auc

tp = np.cumsum(final_df_sort['tp_fp_fn'] == 'TP')
fp = np.cumsum(final_df_sort['tp_fp_fn'] == 'FP')
fn = np.sum(final_df_sort['tp_fp_fn'] == 'FN')

# Calculate precision and recall at each threshold
precision = tp / (tp + fp)
recall = tp / (tp + fn)

final_df_sort['precision']= precision
final_df_sort['recall']= recall
auc_pr = auc(final_df_sort['recall'], final_df_sort['precision'])

# Or can use the:
auc= torch.trapz(torch.tensor(final_df_sort['precision'].values), torch.tensor(final_df_sort['recall'].values))


# In[53]:


precision


# In[54]:


recall


# In[55]:


auc_pr, auc


# In[56]:


plt.plot(recall, precision, marker='.', label='Logistic')
plt.title('Precision vs Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')


# In[57]:


auc


# In[58]:


TP = len(final_df_sort[final_df_sort['tp_fp_fn'] == 'TP'])
FP = len(final_df_sort[final_df_sort['tp_fp_fn'] == 'FP'])
FN = len(final_df_sort[final_df_sort['tp_fp_fn'] == 'FN'])
TN = len(final_df_sort[final_df_sort['tp_fp_fn'] == 'TN'])


# In[59]:


confusion_matrix = np.array([[TP, FP], [FN, TN]])
confusion_matrix

