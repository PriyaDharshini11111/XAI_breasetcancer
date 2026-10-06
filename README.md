# XAI_breasetcancer
Enhancing Breast Cancer Diagnosis with Explainable AI
The proposed framework is used in the context of several breast imaging modalities. Its generalization performance was tested by experiments on mammography dataset (CBIS-DDSM and Mini-DDSM) and an ultrasound dataset (BUS). Hence, the study does not concentrate on ultrasound imaging alone, but rather explores the performance of the proposed framework within the scope of various breast imaging modalities that are often used in the clinical setting.  Three publicly released breast cancer image datasets on Kaggle are used in this study to guarantee thorough training and assessment of the suggested model.To avoid data leakage and to have a proper evaluation protocol, first the data sets were randomly split into training sets (70%), testing sets (15%) and validation sets (15%). Then the TVGCF preprocessing, OAST lesion segmentation, and feature extraction were performed independently in each subset. All the data augmentation operations were performed on the training set, excluding the validation and test sets. As such, there was no information from the validation and testing samples used in the model training, hyperparameter optimization and augmentation procedures, which ensures unbiased evaluation of model performance. 
Dataset A- Breast Ultrasound Images Dataset  https://www.kaggle.com/datasets/subhajournal/busi-breast-ultrasound-images-dataset
Datset B - •	Mini-DDSM (Digital Database for Screening Mammography) https://ardisdataset.github.io/MiniDDSM/
Dataset C - •	CBIS-DDSM (Curated Breast Imaging Subset of DDSM  https://www.cancerimagingarchive.net/collection/cbis-ddsm/
TO run the code, 
1. download the dataset from these links 
2. include the dataset in the folder 
3. Modify the dataset path in the code
