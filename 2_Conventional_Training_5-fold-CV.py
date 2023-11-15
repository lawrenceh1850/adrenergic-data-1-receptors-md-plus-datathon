# Train Conventional Models with 5-fold HP tuning #
### Outcome: 'ckd_status'

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from utils.GridSearch_Schema import Classifiers_dict as Results_dict
from utils.HPTune_Classifiers import HPTune

random_state=1234
np.random.seed(random_state)

### User Variables ###
## Preprocessed data directory ##
data_directory = '../Data/'
## Models folder ##
models_folder='../Models/Conventional/'
## Outcome variable ##
outcome_column='ckd_status'
## CV folds ##
num_folds=5
## Results dictionary path (output) ##
model_pkl = models_folder+'HPTuned_'

### Conventional Model Classifiers ###
model_type_list = [
    # 'LogisticRegression',
    # 'XGBoostClassifier',
    'KNeighborsClassifier',
    # 'SVC_radial',
    # 'DecisionTreeClassifier',
    # 'RandomForestClassifier',
    'AdaBoostClassifier',
    # 'GaussianNB',
    # 'Dummy',
]


### Load Training data ###
train_data = pd.read_csv(data_directory+'train_data.csv')

### Feature list ###
feature_list = list()
for i in train_data.columns:
    if i[:2] == 'F_':
        feature_list.append(i)

### Setup features and outcome ###
X_train = train_data[feature_list]
y_train = train_data[outcome_column]



### Iterate across classifiers ###
for model_type in model_type_list:
    print(f"Training model type {model_type}")
    ## Check if results exist ##
    if( os.path.exists( model_pkl+model_type+'.pkl' ) ):
        print('Model Exists: %s'% model_type)
        continue

    ## Hyperparameter tuning ##
    HP = HPTune(model_type=model_type, random_state=random_state)
    ## AUROC ##
    best_roc_auc = HP.fit_CV(X_train, y_train, num_folds=num_folds, foldername=models_folder)
    ## Update Results_dict (imported) ##
    Results_dict[model_type]['aucroc'] = best_roc_auc
    Results_dict[model_type]['hyperparameters'] = HP.return_best_params()
    print()

    ## Save results as pkl ##
    with open(model_pkl+model_type+'.pkl', 'wb') as fh:
        pickle.dump(Results_dict[model_type], fh)
print('Conventional Models: Training Complete')