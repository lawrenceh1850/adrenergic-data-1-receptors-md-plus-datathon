# adrenergic-data-1-receptors-md-plus-datathon

Lawrence Huang BS, Keyvon Rashidi BSE, Sachin Shankar MS, Dany Alkurdi AB, Felipe Giuste PhD

with adapted code from [Giuste, et al., 2023](https://www.nature.com/articles/s41598-023-36175-4)

<img width="390" alt="image" src="https://github.com/lawrenceh1850/adrenergic-data-1-receptors-md-plus-datathon/assets/26679627/57812536-7056-47e7-9878-f40f60fe3698">

Stages of data processing
1. read in MIMIC-IV dataset with SQL script and join on relevant tables in relational format
  - included tables such as demographics, diagnoses
1. determine CKD status from ICD codes
1. examine most prevalent non-CKD codes associated with CKD patients to determine feature set (prevalence-based feature selection)
1. preprocess data into one-hot encoded ICD codes, split data into train / test sets
1. train 3 different machine learning models on training data to predict CKD status (Logistic Regression, kNN, Random Forest)
1. evaluate model performance on test set with AUROC, select model threshold with best tradeoff of sensitivity / specificity and evaluate confusion matrix of that model
1. best model was RandomForest

<img width="475" alt="image" src="https://github.com/lawrenceh1850/adrenergic-data-1-receptors-md-plus-datathon/assets/26679627/1da08cd2-cb21-4f23-9d10-ae3aad30d6b2">

<img width="627" alt="image" src="https://github.com/lawrenceh1850/adrenergic-data-1-receptors-md-plus-datathon/assets/26679627/4117bde3-d389-4af3-bec0-0c16099a32be">

<img width="643" alt="Screenshot 2023-11-15 at 9 55 28 AM" src="https://github.com/lawrenceh1850/adrenergic-data-1-receptors-md-plus-datathon/assets/26679627/7701fa5f-6b8c-4507-a8a1-3fc61d8a7713">
