# 1. DATA
## 1.1 Dataset

<div align="justify">
The success of any deep learning model heavily relies on the quality and diversity of the dataset used for training. In our project focusing on traffic violation detection using Deep Learning, meticulous attention was paid to curating and pre-processing a comprehensive dataset. This page elucidates the details of our dataset, encompassing its origins, augmentation techniques employed, and the partitioning strategy adopted for training, validation, and testing.
</div>

## 1.2 Dataset Acquisition and Augmentation

<div align="justify">
Our dataset was initially sourced from Kaggle, a prominent platform for sharing and discovering datasets. Specifically, we leveraged a dataset consisting of images depicting individuals riding bicycles, some with helmets and others without. Given the significance of diversity and volume in training robust deep learning models, we augmented the pre-existing dataset of 1000 images. We employed various augmentation techniques to enhance the dataset's diversity and mitigate overfitting. One such technique involved augmenting the HUE and BRIGHTNESS of the pictures by ±15% and the SATURATION and EXPOSURE by ±10%. Additionally, a minute amount of NOISE was introduced into the dataset. We significantly expanded the dataset through these augmentation strategies, resulting in a new augmented dataset comprising 3000 images.
</div>

## 1.3 Data Augmentation Tools

<div align="justify">
Roboflow, a popular platform for managing and augmenting datasets, 
was instrumental in executing the augmentation process. Its user-friendly interface and extensive 
suite of augmentation options facilitated seamless augmentation while ensuring the integrity of the 
dataset. Furthermore, Roboflow facilitated dataset annotations, a crucial step in supervising the 
training process and enabling the model to learn from labelled data. 
</div>

## 1.4 Dataset Partitioning
The augmented dataset was partitioned into three 
subsets: training, validation, and testing. The partitioning ratio was established as 60:20:20, 
ensuring a balanced data distribution across the subsets. This partitioning strategy is paramount 
in evaluating the model's performance effectively, as it enables rigorous testing on unseen data 
while validating the model's efficacy during training. 

# 2 Model Architecture
## 2.1
![Example Image](./Images/Arch.jpg)
