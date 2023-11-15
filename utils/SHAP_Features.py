## XAI
#https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/linear_models/Sentiment%20Analysis%20with%20Logistic%20Regression.html
#https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Census%20income%20classification%20with%20scikit-learn.html

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def model_xai(model, X_train, X_test, model_type=None, n_samples=None, top_n_features=10, random_state=1234, savefigpath=False):
    ## Feature names ##
    feature_names= X_train.columns
    ## Classifiers grouped by Explainer type ##
    tree_classifiers = ['DecisionTreeClassifier', 'XGBoostClassifier', 'AdaBoostClassifier', 'RandomForestClassifier']
    linear_classifiers = ['LogisticRegression']
    kernel_classifiers = ['KNeighborsClassifier', 'GaussianNB', 'SVC_radial']
    ## Probability ##
    predict_proba = lambda x: model.predict_proba(x)[:,1] 
        
    ## Create model Explainer ##
    if(model_type in tree_classifiers):
        try:
            print('Trying Tree Explainer')
            explainer = shap.TreeExplainer(model, X_train, check_additivity=False, seed=random_state)
        except:
            print('Trying Explainer: predict_proba')
            explainer = shap.Explainer(predict_proba, X_train, seed=random_state, model_output="predict_proba")
    elif(model_type in linear_classifiers):
        print('Linear Explainer')
        explainer = shap.LinearExplainer(model, X_train, seed=random_state)
    elif(model_type in kernel_classifiers):
        print('Trying Explainer: predict_proba')
        explainer = shap.Explainer(predict_proba, X_train, seed=random_state, model_output="predict_proba" )
    else:
        print('Trying Explainer: predict_proba')
        explainer = shap.Explainer(predict_proba, X_train, seed=random_state, model_output="predict_proba" )
        
    ## SHAP Values ##
    print('Generating SHAP values')
    if(n_samples):
        rng = np.random.default_rng(seed=random_state)
        rand_inds = rng.choice(range(X_test.shape[0]), size=n_samples, replace=False)
        X_sample = pd.DataFrame( X_test.values[rand_inds], columns=feature_names )
        try:
            shap_values = explainer(X_sample, check_additivity=False)
        except TypeError:
            print('check_additivity not an argument')
            shap_values = explainer(X_sample)
    else:
        try:
            shap_values = explainer(X_test, check_additivity=False)
        except TypeError:
            print('check_additivity not an argument')
            shap_values = explainer(X_test)
    
    ## Correct for multi-column model predictions ##
    if( len(shap_values.values.shape)==3 ):
        print('Correcting multi-column shap_values')
        shap_values = shap_values[:,:,1]
        
    ## Sort Features by importance ##
    f_rankings = list(feature_names[np.argsort(np.abs(shap_values.values).mean(0))[::-1]])
    
    print('Top %i Features'%top_n_features)
    print( f_rankings[:top_n_features] )

    if(savefigpath):
        print('Plotting')
        
        ## Plot: Feature Importance ##
        plt.figure(figsize=(14,7))
        shap.plots.bar(shap_values, max_display=top_n_features+1, show=False )
        plt.tight_layout()
        plt.savefig(savefigpath+'/FeatureRanks_'+model_type+'_bar.pdf')
        plt.close()
        
        ## Plot: Feature Effect (beeswarm) ##
        plt.figure(figsize=(14,7))
        shap.plots.beeswarm(shap_values, max_display=top_n_features+1, show=False )
        plt.tight_layout()
        plt.savefig(savefigpath+'/FeatureRanks_'+model_type+'_swarm.pdf')
        plt.close()
    
    ## Return ##
    return(f_rankings, shap_values.values)