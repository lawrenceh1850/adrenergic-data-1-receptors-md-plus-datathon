# Returns best classifier fit on full Training data
## X/Y are full Training data
## cv_scheme defines GridSearch Hyperparameter tuning CV scheme
## Results_dict holds results of NestedCV
## Returns Tuned best model trained on full Training dataset

import numpy as np

# Classifiers
#from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier

# Grid Search CV
from sklearn.model_selection import GridSearchCV

# Scores
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

# Kernels
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel, polynomial_kernel, sigmoid_kernel, rbf_kernel, laplacian_kernel, chi2_kernel


def bestModel(X, Y, cv_scheme, Results_dict=None, model=None, score='f1_weighted', n_jobs=-1, random_state=1234):
    
    # If model=None, find model with best mean performance from Results_dict:
    if( model ): pass
    elif( Results_dict ):
        model_list = Results_dict.keys()
        scores_dict = { model:list(Results_dict[model]['scores'].values()) for model in model_list }
        model = sorted(scores_dict.items(), reverse=True, key = lambda kv:( np.mean(kv[1]), kv[0]))[0][0]
        print( "Best Model: %s" % model)
    else:
        raise Exception( 'Specify model or Results_dict' )
    
    ### CLASSIFIERS ###
    if(model=='KNeighborsClassifier'):
        classifier= KNeighborsClassifier()
        # N-Neighbors:
        n_neighbors= [3,5,10,20,50]
        # Weights:
        weights= ['uniform','distance']
        # Algorithm
        algorithm= ['auto', 'ball_tree', 'kd_tree', 'brute']
        # Dictionary with hyper-parameters to tune using cross-validation:
        param_dict = dict(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm )
        

    if(model=='SVC_linear'):
        classifier= SVC()
        # SVR Kernel:
        kernel=['linear']
        # Hyper-parameter domain: 
        hyperparam_C = np.linspace(0.01, 10.01, 11) # L2 Regularization
        # Parameter Dictionary:
        param_dict = dict(kernel=kernel, C=hyperparam_C)
       
    if(model=='SVC_poly'):
        classifier= SVC()
        # SVR Kernel:
        kernel=['poly']
        # Hyper-parameter domain: 
        hyperparam_C = np.linspace(0.01, 10.01, 11) # L2 Regularization
        # Hyper-parameter domain:
        hyperparam_degree = np.linspace(0,15,21)
        # Parameter Dictionary:
        param_dict = dict(kernel=kernel, C=hyperparam_C, gamma=['scale', 'auto'], degree=hyperparam_degree)
        
    if(model=='SVC_radial'):
        classifier= SVC()
        # SVR Kernel:
        kernel=['rbf']
        # Hyper-parameter domain: 
        hyperparam_C = np.linspace(0.01, 10.01, 11) # L2 Regularization
        # Hyper-parameter domain:
        hyperparam_gamma = np.logspace(-10, 5, 25)
        # Parameter Dictionary:
        param_dict = dict(kernel=kernel, C=hyperparam_C, gamma=hyperparam_gamma)
        
    if(model=='SVC_sigmoid'):
        classifier= SVC()
        # SVR Kernel:
        kernel=['sigmoid']
        # Hyper-parameter domain: 
        hyperparam_C = np.linspace(0.01, 10.01, 11) # L2 Regularization
        # Hyper-parameter domain:
        hyperparam_gamma = np.logspace(-10, 5, 30)
        # Parameter Dictionary:
        param_dict = dict(kernel=kernel, C=hyperparam_C)
        
        
    if(model=='DecisionTreeClassifier'):
        classifier=DecisionTreeClassifier()
        # Tree Splitter:
        splitter=['best', 'random']
        # Max Depth:
        max_depth= [None, 2, 10]
        # Min Samples to Split:
        min_samples_split= [2, 10, 25]
        # Max Features:
        max_features= [None, 5,10,25, 'sqrt', 'log2']
        # Parameter Dictionary:
        param_dict = dict(splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, 
                          max_features=max_features, random_state=[random_state] )
        
    if(model=='RandomForestClassifier'):
        classifier=RandomForestClassifier()
        # N-trees:
        n_estimators= [100, 200, 500]
        # Max Depth:
        max_depth= [None, 2, 10]
        # Min Samples to Split:
        min_samples_split= [2, 10, 25]
        # Max Features:
        max_features= [None, 5, 25, 'sqrt', 'log2']
        # Parameter Dictionary:
        param_dict = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, 
                          max_features=max_features, random_state=[random_state] )
        
    if(model=='AdaBoostClassifier'):
        classifier=AdaBoostClassifier()
        # N-trees:
        n_estimators= [50, 100, 500]
        # Learning Rate:
        learning_rate = [ 1., 0.2, 0.5 ]
        # Parameter Dictionary:
        param_dict = dict(n_estimators=n_estimators, learning_rate=learning_rate, random_state=[random_state] )
        
    if(model=='GaussianNB'):
        classifier=GaussianNB()
        # Variance Smoothing:
        var_smoothing = [1e-9, 0.05, 0.10, 0.25]
        # Parameter Dictionary:
        param_dict = dict(var_smoothing=var_smoothing )
        
    if(model=='QuadraticDiscriminantAnalysis'):
        classifier=QuadraticDiscriminantAnalysis()
        # Store Covariances for Classes
        store_covariance=[True]
        # Parameter Dictionary:
        param_dict = dict(store_covariance=store_covariance )
        
    if(model=='Dummy'):
        classifier= DummyClassifier()
        # Stragety:
        strategy=[ 'stratified', 'most_frequent', 'prior', 'uniform', 'constant' ]
        # Constant guess:
        constant=list( set(Y) )
        # Dictionary with hyper-parameters to tune using cross-validation:
        param_dict = dict(strategy=strategy, constant=constant)
    
    # Hyper-parameter Grid Search with Cross-Validation:
    cls = GridSearchCV(classifier, param_dict,
                        scoring=score, cv=cv_scheme, n_jobs=n_jobs)
    
    # Train Tuned models on full Training set:
    cls.fit( X, Y )
    
    # Return tuned model trained on full Training dataset:
    return(cls)