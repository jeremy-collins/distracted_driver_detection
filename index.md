# Distracted Driver Detection
Jeremy Collins, Alan Hesu, Kin Man Lee, Shruthi Saravanan, Dhrumin Shah


## Introduction
  Distracted driving causes about 920,000 total accidents in the US per year according to the National Highway Transportation and Safety Administration (NHTSA) [1]. Experts in the fields of traffic safety and public health all concur that this is an underestimation of the dangers of distracted driving.  
 
  Prior work in this field includes a data augmentation method for distracted driving detection based on extracting relevant driving operation areas in the image as a preprocessing step using an R-CNN model [2]. The findings from this paper demonstrate the importance of doing operation area extraction in the preprocessing step, which can efficiently reduce redundant information in images and improve classification accuracy. An additional report in this field includes a distracted driving identification algorithm based on deep CNNs [3]. This approach coupled PCA with a multi-layer CNN to further improve performance.  

  In this project, we will use the State Farm Distracted Driver Detection dataset [4] to classify normal driving, texting, phone conversation, radio operation, drinking, reaching back, doing hair and makeup, and talking to passengers.  
  
## Problem Definition
  70% of fatal crashes are caused by unsafe driving behavior. Our project aims to preemptively identify such driving behavior in order to help prevent fatal accidents and adjust a driver’s habits.   

## Methods
  For our supervised learning method, we plan on implementing a convolutional neural network to classify the driver images as depicting normal driving or distracted driving. We chose to use a CNN because of their ability to efficiently extract visual features with higher accuracy than other types of neural networks. We will also be exploring traditional supervised learning methods such as support vector machines and decision trees.  

  
  We plan on exploring several unsupervised learning methods to classify images via clustering. These methods may include k-means, GMM, PCA, or DBSCAN.  
 
  ![img](/distracted_driver_detection/docs/assets/pca_explained_variance.png)


## Potential Results
  For unsupervised clustering methods, we hope to see clusters for each of the defined classes given in the dataset. Since there are 10 classes, the expected amount of clusters should be the same. For resulting clusters that do not match the ground truth labels, we will perform some exploratory analysis to look for other possible similarities that define the clusters. When the number of clusters is limited to two, we expect the model to correlate with the binary classification of normal driving vs. distracted driving.  

  
  For supervised methods, the models we use should be able to accurately classify an unlabeled image to one of the ten classes. Based on prior work in this area, we expect to see around 60-80% accuracy [5] with traditional methods depending on what is used. For a CNN, we expect to see accuracy of 85%+, possibly reaching 95%+ [6] if we find a CNN architecture that works well for our particular use case.  

## Proposed Timeline
A proposed Gantt chart can be viewed <a href="GanttChart - Spring.pdf" target="_blank">here.</a>  

## References
[1] S. Coleman, “Distracted driving statistics 2022,” Bankrate, 07-Sep-2021. [Online]. Available: https://www.bankrate.com/insurance/car/distracted-driving-statistics/. [Accessed: 24-Feb-2022].  
  
[2] J. Wang, Z. Wu, F. Li, and J. Zhang, “A Data Augmentation Approach to Distracted Driving Detection,” Future Internet, vol. 13, no. 1, p. 1, Dec. 2020, doi: 10.3390/fi13010001.  
  
[3] Rao, X., Lin, F., Chen, Z. et al. Distracted driving recognition method based on deep convolutional neural network. J Ambient Intell Human Comput 12, 193–200 (2021). https://doi.org/10.1007/s12652-019-01597-4  
  
[4] State Farm Distracted Driver Detection, Kaggle, 2016. Accessed on: Feb. 24, 2022. [Online]. Available: https://www.kaggle.com/c/state-farm-distracted-driver-detection/overview  
  
[5] D. Feng and Y. Yue, “Machine Learning Techniques for Distracted Driver Detection,” CS 229: Machine Learning, 2019. [Online]. Available: http://cs229.stanford.edu/proj2019spr/report/24.pdf. [Accessed: 23-Feb-2022].  
  
[6] M. H. Alkinani, W. Z. Khan and Q. Arshad, "Detecting Human Driver Inattentive and Aggressive Driving Behavior Using Deep Learning: Recent Advances, Requirements and Open Challenges," in IEEE Access, vol. 8, pp. 105008-105030, 2020, doi: 10.1109/ACCESS.2020.2999829.  



## Video 
<iframe width="560" height="315" src="https://www.youtube.com/embed/X3aVDufNLig" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

