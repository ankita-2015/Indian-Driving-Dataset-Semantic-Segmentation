import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import tensorflow as tf
from glob import glob
from tqdm import tqdm
import pandas as pd
import keras
from tensorflow.keras import backend as k
import gc

AUTOTUNE = tf.data.experimental.AUTOTUNE

def convert_to_rgb(predi,_cmap={}):
  pred_image = np.zeros((predi.shape[0], predi.shape[1], 3),dtype=np.uint8) + 255
  for i in np.unique(predi):
    pred_image[predi==i] = _cmap[i]
  return pred_image

def plot_imgs(i,img,mask,pred=np.zeros((1024,1024,3)),cmap={},mask_labels={},label_iou={}):  
    fig,(ax1,ax2,ax3, ax4) = plt.subplots(1,4,figsize=(20,4))
    if img.shape[-1]==3:
        ax1.imshow(img)
    else:
        ax1.imshow(img,cmap=plt.get_cmap('gray'))
    ax1.axis('off')
    ax1.title.set_text(f"Input Image {i}") 

    ax2.imshow(mask)
    ax2.axis('off')
    ax2.title.set_text(f"Ground truth {i}")

    ax3.imshow(pred)
    ax3.axis('off')   
    ax3.title.set_text(f"Prediction {i}")  

    dst = cv2.addWeighted(np.asarray(img*255.0,dtype=np.uint8),1,pred,0.5,0)
    ax4.imshow(dst)
    ax4.axis('off')

    patches = [ mpatches.Patch(color=np.array(cmap[i])/255.0, label="{:<15}:{:2.3f} ".format(mask_labels[i],label_iou[i])) for i in label_iou.keys()]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

#     path = "xception/output_result"
    if not os.path.exists(path):
        os.mkdir(path)
    img_name = f"{i}.png"
#     fig.set_size_inches(25, 10)
    plt.savefig(f'{path}/{img_name}', dpi=100)
#     plt.show()
    plt.close(fig)

def IoU(Yi,y_predi,mask_labels={}):
  ## mean Intersection over Union
  ## Mean IoU = TP/(FN + TP + FP)
  
    IoUs = [] 
    precisions = []
    recalls = []
    f1_scores=[]
    f2_scores=[]
    #   dice_scores = []

    labels_iou = {}  
    for c in mask_labels.keys():      
        TP = np.sum( (Yi == c)&(y_predi==c) )
        FP = np.sum( (Yi != c)&(y_predi==c) )
        FN = np.sum( (Yi == c)&(y_predi != c)) 
        TN = np.sum( (Yi != c)&(y_predi != c)) 

        IoU = TP/float(TP + FP + FN)
        precision = TP/float(TP + FP)
        recall = TP/float(TP + FN)

        beta= 1
        f1_score = ((1+beta**2)*precision*recall)/float(beta**2*precision + recall)

        beta= 2
        f2_score  = ((1+beta**2)*precision*recall)/float(beta**2*precision + recall)

        #     dice_score = (2*TP)/float(2*TP + FP + FN)

        if IoU > 0:
#             print("class {:2.0f} {:10}:\t TP= {:6.0f},\t FP= {:6.0f},\t FN= {:6.0f},\t TN= {:6.0f},\t IoU= {:6.3f}".format(c,mask_labels[c],TP,FP,FN,TN,IoU))                    

            labels_iou[c] = IoU
            IoUs.append(IoU)  
            precisions.append(precision) 
            recalls.append(recall)  
            f1_scores.append(f1_score)  
            f2_scores.append(f2_score) 
            #       dice_scores.append(dice_score) 

    mIoU = np.mean(IoUs)
    labels_iou[len(req_mask_labels)-1] = mIoU
#     print("Mean IoU: {:4.6f}".format(mIoU))  

    return labels_iou, [mIoU, np.mean(precisions),np.mean(recalls),np.mean(f1_scores), np.mean(f2_scores)]

IMG_SIZE = 512
def parse_x_y(img_path,mask_path):    
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    
    mask = tf.io.read_file(mask_path)    
    mask = tf.image.decode_png(mask, channels=1)  
    return {'image': image, 'segmentation_mask': mask}

@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE),method='nearest')    
#     if tf.random.uniform(()) > 0.5:
#         input_image = tf.image.flip_left_right(input_image)
#         input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)
    input_mask = tf.one_hot(input_mask, 3)
    input_mask = tf.reshape(input_mask, (IMG_SIZE, IMG_SIZE, 3))
    return input_image, input_mask

columns=["Image_name", "mIoU", "Precision", "Recall", "F1-score(Dice-score)","F2-score"]

def make_predictions(model, csv_path, test_images_folder_path, test_mask_folder_path):    
    test_x = glob(test_images_folder_path)
    test_y = glob(test_mask_folder_path)

    test_x.sort()
    test_y.sort()

    test_dataset = tf.data.Dataset.from_tensor_slices((test_x,test_y))
    test_dataset = test_dataset.map(parse_x_y)

    dataset = {"test": test_dataset}
    dataset['test'] = dataset['test'].map(load_image_train, 
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE
                                         ).batch(BATCH_SIZE)
    input_details = model.get_input_details()
    output_details = model.get_output_details()
        
    df = pd.DataFrame(columns=columns)
    for j in tqdm(range(0,len(test_x),BATCH_SIZE)):
        img, mask = next(iter(dataset['test'] ))
#         interpreter.set_tensor(input_details[0]['index'],[img[0]])               

#         y_pred = model.predict(img) 
#         k.clear_session()
#         gc.collect()
        for i in range(BATCH_SIZE):
#             print("-"*50)
#             print("Image : ", i+j+1)    
            model.set_tensor(input_details[0]['index'],[img[i]])  
#             model.resize_tensor_input(input_details[0]['index'], [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3])
            model.allocate_tensors()
            model.invoke()
            y_pred = model.get_tensor(output_details[0]['index'])

            maski = np.squeeze(np.argmax(mask[i], axis=-1))
            y_predi = np.squeeze(np.argmax(y_pred[0], axis=-1))

            label_iou, eval_=IoU(maski, y_predi, mask_labels=req_mask_labels)

            input_img = img[i]
            ground_truth = np.squeeze(convert_to_rgb(maski, req_cmap))    
            prediction = np.squeeze(convert_to_rgb(y_predi, req_cmap))

            plot_imgs(i+j+1,input_img, ground_truth, prediction,cmap=req_cmap,mask_labels=req_mask_labels,label_iou=label_iou)

            df = df.append(pd.Series([f"Image {i+j+1}",*eval_], index=columns), ignore_index=True)
            
            df.to_csv(csv_path,index=False)
    return df

## mobilenetv2 edge

import tensorflow as tf
import os
import numpy as np
import cv2
from segmentation_models.losses import cce_jaccard_loss, dice_loss, JaccardLoss
from segmentation_models.metrics import iou_score, f1_score, precision, recall
ls = dice_loss + cce_jaccard_loss
metrics = [precision, recall, f1_score, iou_score] 

from tensorflow.keras.models import load_model

# Load TFLite model and allocate tensors.
model = tf.lite.Interpreter(model_path="../../RESULTS/IDD_Dv3p_mobilenetV2_alpha1.0_bs16/model.tflite")
model.allocate_tensors()

path = "../../RESULTS/IDD_Dv3p_mobilenetV2_alpha1.0_bs16/output_result_BDD_lite"
csv_path = "../../RESULTS/IDD_Dv3p_mobilenetV2_alpha1.0_bs16/TestResult_BDD_lite.csv"

BATCH_SIZE = 20
req_cmap = {
        0: (0,0,0), # background                        
        1: (255,0,0),    # road
        2: (0, 0, 255), #obstacle  
        3: (255,255,255)    # miou label
        }

req_mask_labels = {
    0:"Background",    
    1:"Road",
    2:"Obstacle",    
    3:"miou"
}

test_images_folder_path = "../../Dataset/BDD/Test/images/*"
test_mask_folder_path="../../Dataset/BDD/Test/masks/*"
df = make_predictions(model,csv_path, test_images_folder_path, test_mask_folder_path)   