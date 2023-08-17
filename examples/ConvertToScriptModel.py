import cv2, os, time
import numpy as np
import argparse
from typing import Callable, List
import cv2
import torch
from catalyst.utils import load_checkpoint
from PIL import Image
import torch
from torch.autograd import Variable

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




def convertToTracedScriptModel(original_model):
    with torch.no_grad():
        device = torch.device('cpu')

        # run the tracing
        generated_input = Variable(
            torch.zeros(1, 3, 640, 640)
        )
        traced_script_module = torch.jit.trace(original_model, generated_input, strict=False)
        # save the converted model
        traced_script_module.save("ercp_traced_best_tiny.pt")
        model = torch.jit.load('ercp_traced_best_tiny.pt')
        model.to(device)
        model.eval()

        for i in range(2):
            out = model(generated_input)
            ori_out = original_model(generated_input)
            #print(out)
            print(ori_out)
        img = cv2.imread("01-DEC-21_01-22AM_002_processed_022943.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.astype(np.float)
        img = img/255
        img = cv2.resize(img, (640, 640))

        img_tensor = torch.from_numpy(img);
        img_tensor = img_tensor.swapaxes(0, 2)
        img_tensor = img_tensor.swapaxes(1, 2)
        img_tensor = img_tensor.view(1,3,640,640).float()
        out = model(img_tensor)
        preds = original_model.postprocess(out, conf_thres=0., multiple_labels_per_box=False)
        nms_predictions = filter_eval_predictions(preds, confidence_threshold=0.1)

        if len(nms_predictions[0]) > 1:
            nms_predictions = [nms_predictions[0][1].reshape(1, 6)]

        # Get the predicted boxes
        pred_boxes = nms_predictions[0][:, :4].cpu().numpy()


# Defining model
best_model = create_yolov7_model('yolov7-tiny', num_classes=1, pretrained=False)


# In[32]:


# Loading Weights
best_model_path= '/Users/longquanchen/Desktop/Work/Versatile/Yolov7-training-main/examples/best_model_1.pt'
checkpoint = torch.load(best_model_path, map_location='cpu')
state_dict = checkpoint['model_state_dict']
best_model.load_state_dict(state_dict)
best_model.eval()


convertToTracedScriptModel(best_model)