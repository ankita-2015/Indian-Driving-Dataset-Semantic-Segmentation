## A Deep Learning project on Multi-class Semantic Segmentation on [Indian Driving Dataset](https://idd.insaan.iiit.ac.in/dataset/details/)

## About Dataset
I have used dataset which is available at [kaggle IDD dataset](https://www.kaggle.com/abhishekprajapat/idd-20k)

Dataset provides image resolution of (**width**:1920, **height**:1080):
  1. Training Images: 5966
  2. Validation Images: 1016
  
I have used Validation images as Test Set for evaluation purpose and For training i have splitted Training images to train(4950) and Val set(1016)

The label image consists of 26 classes.

For this project, i have customized the number of classes:
1. There are 2 main classes: 

      - **Road** (Drivable)
      - **Moving objects** (Living things and Vehicles category as per this [offical website](https://idd.insaan.iiit.ac.in/dataset/details/))
      - **Background** (Rest of the categories as background which includes Non Drivable, Road Side Objects, Sky, Far Objects)

After mapping it to 3 categories, label image looks like: 

![image](https://user-images.githubusercontent.com/29517840/155579163-ac7d5251-1536-423a-8a7b-5140c9c9e503.png)

Script to customize number of classes for IDD dataset is available at [Scripts/DatasetPreparation/prepare_IDD_dataset.ipynb](https://github.com/ankita-2015/DeepLearning/blob/main/Semantic%20Segmentation/Obstacle%20Segmentation/Scripts/DatasetPreparation/prepare_IDD_dataset.ipynb)

## Training 
 
 - For training i have resized images to 512 X 512
   
   ![image](https://user-images.githubusercontent.com/29517840/155583808-7cf985e4-6c9f-43a4-b434-287a3da4a068.png)

- I have trained various models on this dataset with different techniques. All training scripts are present in [Scripts/TrainingScripts/](https://github.com/ankita-2015/DeepLearning/tree/main/Semantic%20Segmentation/Obstacle%20Segmentation/Scripts/TrainingScripts)
  - Deeplabv3p with MobilenetV2 with pretrained [Cityscapes weights](https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5)
  - Deeplabv3p with Xception with pretrained [Cityscapes weights](https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5) 
  - Unet with MobilenetV2 with pretained encoder
  - Unet with EfficientNetB0 with pretained encoder
  - Unet with EfficientNetB1 with pretained encoder
  - Unet with EfficientNetB6 with pretained encoder
  - Unet with PSPNet with pretained encoder
  - Unet with FPN with pretained encoder
  - Unet with LinkNet with pretained encoder
  - Deeplabv3p with MobilenetV2(alpha 1.0) - custom model with pretrained encoder
  - Deeplabv3p with MobilenetV2(alpha 0.35) - custom model with pretrained encoder and decoder upsampled by 2
  - Deeplabv3p with MobilenetV2(alpha 0.35) - custom model with pretrained encoder and decoder upsampled by 4
  - Deeplabv3p with MobilenetV2(alpha 0.35) - custom model with pretrained encoder and decoder upsampled by 4, applied DepthwiseSeparable Convolution in decoder 
  - Deeplabv3p with MobilenetV2(alpha 0.3) - custom model trained from scratch 

## Evaluation

I have Evaluated models results on datasets:
  - IDD validation set (NOT used in training)
  - [Cityscapes](https://www.kaggle.com/xiaose/cityscapes) test set - 500 images - [prepare cityscape dataset](https://github.com/ankita-2015/DeepLearning/blob/main/Semantic%20Segmentation/Obstacle%20Segmentation/Scripts/DatasetPreparation/prepare_Cityscape_dataset.ipynb)
  - [BDD100k](https://www.kaggle.com/solesensei/solesensei_bdd100k) test set - 1000 images - [prepare_BDD100k dataset](https://github.com/ankita-2015/DeepLearning/blob/main/Semantic%20Segmentation/Obstacle%20Segmentation/Scripts/DatasetPreparation/prepare_BDD_dataset.ipynb)

## Results

| Model |Images Count|mIoU|Precision|Recall|F1-score (Dice-score)|F2-score|
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
|IDD_Unet_efficientnetB6_alpha1_102epoch_IDD|1016|0.935507686|0.966729734|0.964703715|0.964771986|0.964548665|
|IDD_Unet_efficientnetB1_alpha1_169epoch_IDD|1016|0.930382308|0.962567004|0.963652336|0.961783635|0.962565754|
|IDD_Unet_efficientnetB0_alpha1_172epoch_IDD|1016|0.929789305|0.964161625|0.961232964|0.961725158|0.961219339|
|IDD_Unet_mobilenetV2_alpha1_125epoch_IDD|1016|0.924768612|0.96202396|0.957881042|0.958814571|0.958030642|
|**IDD_Dv3p_mobilenetV2_upsampleby2_custom_alpha0_35_168epoch_IDD**|1016|**0.923801275**|0.961042069|0.957990114|0.958356378|0.957920497|
|IDD_Linknet_mobilenetV2_alpha1_100epoch_IDD|1016|0.919378631|0.959779189|0.953338128|0.95566278|0.954095486|
|IDD_Unet_mobilenetV2_alpha0_35_150epoch_IDD|1016|0.917968092|0.961496856|0.950309282|0.954694436|0.951842498|
|IDD_Dv3p_mobilenetV2_upsampleby4_1X1_DSC_custom_alpha0_35_340epoch_IDD|1016|0.912327724|0.955982275|0.948532736|0.951089052|0.949336086|
|IDD_Dv3p_mobilenetV2_upsampleby4_custom_alpha0_35_111epoch_IDD|1016|0.911748393|0.955098572|0.949369409|0.951108694|0.949833539|
|IDD_PSPnet_mobilenetV2_alpha1_125epoch_IDD|1016|0.889653239|0.941406104|0.937448567|0.937932286|0.937336957|  
|**IDD_Unet_efficientnetB1_alpha1_169epoch_CS**|500|**0.736945095**|0.813979215|0.878560476|0.834813002|0.857184156|
|IDD_Unet_efficientnetB6_alpha1_102epoch_CS|500|0.72419415|0.803299049|0.868500158|0.823064889|0.845677884|
|IDD_Linknet_mobilenetV2_alpha1_100epoch_CS|500|0.707557416|0.810975559|0.836996694|0.815088204|0.82578831|
|IDD_mobilenetV2_apha1_725epoch_CS|500|0.704201384|0.820824768|0.834737747|0.817251204|0.824993245|
|IDD_Unet_mobilenetV2_alpha1_125epoch_CS|500|0.703853585|0.792599002|0.847816488|0.80684931|0.826419052|
|IDD_Unet_efficientnetB0_alpha1_172epoch_CS|500|0.702946435|0.775713217|0.84451824|0.796338827|0.81949006|
|IDD_mobilenetV2_apha1_350epoch_CS|484|0.684523423|0.806061196|0.811062906|0.794792795|0.800980755|
|**IDD_mobilenetV2_apha1_350epoch_BDD**|1000|**0.68348494**|0.782480068|0.828049944|0.783499953|0.80064466|
|IDD_Unet_efficientnetB6_alpha1_102epoch_BDD|1000|0.648662205|0.774714055|0.803258877|0.750631798|0.766699096|



