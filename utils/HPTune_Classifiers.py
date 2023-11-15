# Runs Hyperparameter Tuning (Grid Search)

import os

import numpy as np
import pandas as pd

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
import statsmodels.api as sm     # sm.Logit is Logistic Regression
import statsmodels
from xgboost import XGBClassifier

# Grid Search CV
from sklearn.model_selection import ParameterGrid, GridSearchCV

# Scores
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler



class HPTune:
    def __init__(self, model_type='LogisticRegression', random_state=1234, score='roc_auc_score', use_statsmodel=False):
        self.random_state=random_state
        np.random.seed(self.random_state)

        self.model_type = model_type
        print(self.model_type)

        ### CLASSIFIERS ###
        if(model_type=='LogisticRegression' and use_statsmodel):
            self.classifier = None   # placeholder for statsmodels api
            # Hyperparameter Dictionary:
            self.param_dict = dict()
        elif(model_type=='LogisticRegression' and not use_statsmodel):
            self.classifier = LogisticRegression()
            # Regularization
            penalty=['none', 'l2', 'l1']
            solver=['saga'] # works on all chosen penalties
            # Hyperparameter Dictionary:
            self.param_dict = dict(penalty=penalty, max_iter=[1000000], solver=solver, n_jobs=[-1],
                                   random_state=[random_state])
        elif(model_type=='XGBoostClassifier'):
            self.classifier = XGBClassifier()
            # Learning Rate:
            learning_rate = np.linspace(1E-5, 1E-2, 25) #np.logspace(-3, 0, 20)
            # Number of Trees
            n_estimators = [1000000]
            # Parameter Dictionary:
            self.param_dict = dict(learning_rate=learning_rate, n_estimators=n_estimators, eval_metric=['logloss'],
                                   random_state=[random_state], early_stopping_rounds=[10], n_jobs=[-1])

        elif(model_type=='KNeighborsClassifier'):
            self.classifier= KNeighborsClassifier()
            # N-Neighbors:
            n_neighbors = [3, 5, 10, 20, 50, 100, 300, 500, 1000] # [int(round(i)) for i in (np.logspace(0, 3, 15))]
            # Weights:
            weights = ['uniform','distance']
            # Hyperparameter Dictionary:
            self.param_dict = dict(n_neighbors=n_neighbors, weights=weights,
                                   n_jobs=[-1] )

        elif(model_type=='SVC_radial'):
            self.classifier=SVC()
            # SVR Kernel:
            kernel=['rbf']
            # Hyper-parameter domain:
            hyperparam_gamma = np.logspace(-10, 5, 20)
            # Hyperparameter Dictionary:
            self.param_dict = dict(kernel=kernel, gamma=hyperparam_gamma, probability=[True], max_iter=[1000000],
                                   random_state=[random_state])

        elif(model_type=='DecisionTreeClassifier'):
            self.classifier=DecisionTreeClassifier()
            # Hyperparameter Dictionary:
            self.param_dict = dict(random_state=[random_state])

        elif(model_type=='RandomForestClassifier'):
            self.classifier=RandomForestClassifier()
            # N-trees:
            n_estimators=[10, 50, 100, 500, 1000, 5000, 10000] #, 50000, 100000] #[int(round(i)) for i in (np.logspace(0, 3, 15))]
            # Hyperparameter Dictionary:
            self.param_dict = dict(n_estimators=n_estimators, n_jobs=[-1],
                                   random_state=[random_state] )

        elif(model_type=='AdaBoostClassifier'): # Not parallelized
            self.classifier=AdaBoostClassifier()
            # N-trees:
            n_estimators=[10, 50, 100, 500, 1000, 5000, 10000] #[int(round(i)) for i in (np.logspace(0, 3, 15))]
            # Learning Rate:
            learning_rate = np.logspace(-3, 0, 20)
            # Hyperparameter Dictionary:
            self.param_dict = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                                   random_state=[random_state] )

        elif(model_type=='GaussianNB'):
            self.classifier=GaussianNB()
            # Hyperparameter Dictionary:
            self.param_dict = dict()

        elif(model_type=='Dummy'):
            self.classifier = DummyClassifier(strategy='prior')
            # Hyperparameter Dictionary:
            self.param_dict = dict(random_state=[random_state])

        else:
            print('Not a valid model type')


    ### Fit without cross-validation
    ### Returns model with hyperparameters with highest AUROC score
    def fit(self, X_train, y_train, X_valid, y_valid, save_parameter_grid=True, foldername='results/'):
        self.method = 'fit'
        np.random.seed(self.random_state)

        roc_auc_score_list = list()
        ## Logistic Regression: statsmodel api ##
        if self.model_type == 'LogisticRegression':
            ## Add constant term to equation ##
            X_train = statsmodels.tools.tools.add_constant(X_train)
            X_valid = statsmodels.tools.tools.add_constant(X_valid)

            ## Fit ##
            self.classifier = sm.Logit(y_train, X_train).fit();
            ## Validate ##
            roc_auc_score_list.append( roc_auc_score(y_valid, self.classifier.predict(X_valid, which='prob')) )
            self.best_classifier = self.classifier

            self.parameter_grid = pd.DataFrame([max(roc_auc_score_list)], columns=['AUROC'])
            if save_parameter_grid:
                self.parameter_grid.to_csv( os.path.join(foldername, self.model_type + '_hyperparameter_grid.csv'), index=False)

            return max(roc_auc_score_list)

        ## Models with sklearn-style APIs ##
        else:
            ## Dataframe with hyperparameter grid ##
            self.parameter_grid = pd.DataFrame ( list(ParameterGrid(self.param_dict)) ) # All possible combinations of parameters

            ## Iterate across hyperparameters: fit the model and record AUROC ##
            for i in range(len(self.parameter_grid)):

                ## Set model hyperparameters ##
                self.classifier.set_params( **dict(self.parameter_grid.iloc[i]) )

                ## Ensure correct datatypes ##
                try:
                    self.classifier.n_estimators = int(self.classifier.n_estimators)
                    self.classifier.random_state = int(self.classifier.random_state)
                except:
                    pass

                ## XGBoost ##
                if self.model_type == 'XGBoostClassifier':
                    ## Fit ##
                    self.classifier.fit(X=X_train, y=y_train, verbose=0,
                                        eval_set=[(X_train, y_train), (X_valid, y_valid)])
                    ## Predict ##
                    # roc_auc_score_list.append( roc_auc_score(y_valid, MinMaxScaler().fit_transform(
                    #             self.classifier.predict(X_valid, output_margin=True).reshape(-1,1)).ravel()) )
                    roc_auc_score_list.append( roc_auc_score(y_valid, self.classifier.predict_proba(X_valid)[:,1]) )

                    ## Save best model (early stopping) ##
                    self.parameter_grid['n_estimators'][i] = self.classifier.best_iteration

                ## SKLearn Models ##
                else:
                    ## Train ##
                    self.classifier.fit(X_train, y_train)
                    ##
                    roc_auc_score_list.append( roc_auc_score(y_valid, self.classifier.predict_proba(X_valid)[:,1]) )

        ## classifier with hyperparameters optimized, untrained (best mean score across folds) ##
        self.best_classifier = self.classifier.set_params( **dict(self.parameter_grid.iloc[np.argmax(roc_auc_score_list), :]) )

        ## Add AUROC results to parameter grid ##
        self.parameter_grid['AUROC'] = roc_auc_score_list

        ## save parameter grid to csv ##
        if save_parameter_grid:
            self.parameter_grid.to_csv( os.path.join(foldername, self.model_type + '_hyperparameter_grid.csv'), index=False)

        ## return best AUROC score (with optimized hyperparameters) ##
        return max(roc_auc_score_list)




    ### Fit with cross-validation for hyperparameter tuning
    ### Runs on a single training dataset (no separate validation dataset)
    ### Save grid search results and best parameters, does not use sklearn GridSearchCV
    ### Returns the highest mean AUROC score across folds
    def fit_CV(self, X_train, y_train, num_folds=5, save_parameter_grid=True, foldername='cv_results/', use_statsmodel=False):
        self.method = 'fit_CV'
        np.random.seed(self.random_state)

        ## CV folds: Fold==fold where data used for validation ##
        fold_array = np.random.choice(np.arange(1,num_folds+1), size=len(X_train))

        ## XGBoost Classifier ##
        if self.model_type == 'XGBoostClassifier':
            ## Hyperparameter dataframe ##
            self.parameter_grid = pd.DataFrame( list(ParameterGrid(self.param_dict)) )
            #print('Hyperparameters: %s'% self.parameter_grid)

            ## Create empty results dictionary ##
            parameter_set_results = dict()
            for fold in range(1, num_folds+1): # add keys
                parameter_set_results.update( {'split'+str(fold)+'_test_score': [] } )
                parameter_set_results.update( {'best_iteration_'+str(fold): [] } )
            parameter_set_results.update( {'mean_test_score': [] } )

            ### Iterate across hyperparameters ##
            for i in range(len(self.parameter_grid)):
                print('%s:%s'% (i+1,len(self.parameter_grid)), end='\r' )

                ## Set Hyperparameters ##
                self.classifier.set_params( **dict(self.parameter_grid.iloc[i]) )
                self.classifier.n_estimators = int(self.classifier.n_estimators)
                self.classifier.random_state = int(self.classifier.random_state)

                ## Iterate across folds: train->validate, save results ##
                roc_auc_list = list()
                best_iterations_list = list()
                for fold in range(1, num_folds+1):
                    ## Fit ##
                    self.classifier.fit(X=X_train.iloc[fold_array!=fold], y=y_train.iloc[fold_array!=fold], verbose=0,
                                        eval_set=[(X_train.iloc[fold_array==fold], y_train.iloc[fold_array==fold])])

                    ## Validate ##
                    curr_roc_auc_score = roc_auc_score(y_train.iloc[fold_array==fold],
                                                       self.classifier.predict_proba(X_train.iloc[fold_array==fold])[:,1]
                                                      )

                    ## Append score for Fold ##
                    parameter_set_results['split'+str(fold)+'_test_score'].append(curr_roc_auc_score)
                    roc_auc_list.append( curr_roc_auc_score )
                    ## Best n_estimators determined by Early Stopping ##
                    parameter_set_results['best_iteration_'+str(fold)].append( self.classifier.best_iteration )
                    best_iterations_list.append( self.classifier.best_iteration )

                ## Append mean score across folds ##
                parameter_set_results['mean_test_score'].append( np.mean(roc_auc_list) )

            ## Append mean score alongside hyperparameters ##
            self.parameter_grid = pd.concat( (self.parameter_grid, pd.DataFrame(parameter_set_results)), axis=1 )

            ## Save results to csv ##
            if save_parameter_grid:
                self.parameter_grid.to_csv( os.path.join(foldername, self.model_type + '_hyperparameter_grid.csv'), index=False)

            ## Dictionary with best hyperparameters (maximum mean score across folds) ##
            best_param_dict = dict( self.parameter_grid.iloc[np.argmax(self.parameter_grid['mean_test_score'])] )
            ## Set n_estimators to median across folds of best iteration (from early stopping) ##
            best_param_dict['n_estimators'] = int(np.median( best_iterations_list ))

            ## Remove non-hyperparameter values ##
            pop_list = list() # list to hold keys that need to be removed
            for key in best_param_dict.keys():
                if key in ['mean_test_score', 'random_state', 'early_stopping_rounds']:
                    pop_list.append(key)
                elif key[:5] == 'split' and key[6:] == '_test_score':
                    pop_list.append(key)
                elif key[:5] == 'best_': # best_iteration
                    pop_list.append(key)
            for key in pop_list:
                best_param_dict.pop(key)
            ## Set best hyperparameters (highest mean score within each fold) ##
            self.best_params_ = best_param_dict
            ## Set classifier with hyperparameters optimized, untrained (best mean score) ##
            self.best_classifier = self.classifier.set_params( **self.best_params_ )

            ## Return best mean score (optimized hyperparameters) ##
            return np.max(self.parameter_grid['mean_test_score'])

        ## SKLearn Models ##
        else:
            ## Hyperparameter dataframe ##
            self.parameter_grid = pd.DataFrame ( list(ParameterGrid(self.param_dict)) ) # All possible combinations of parameters

            ## Create empty hyperparameter dictionary ##
            parameter_set_results = dict()
            for fold in range(1, num_folds+1):
                parameter_set_results.update( {'split'+str(fold)+'_test_score': [] } )
            parameter_set_results.update( {'mean_test_score': [] } )


            ## Iterate across hyperparameters ##
            for i in range(len(self.parameter_grid)):
                print('%s:%s'% (i+1,len(self.parameter_grid)), end='\r' )

                ## Set hyperparameters ##
                self.classifier.set_params( **dict(self.parameter_grid.iloc[i,:]) )
                ## Ensure correct datatypes ##
                try:
                    self.classifier.n_estimators = int(self.classifier.n_estimators)
                    self.classifier.random_state = int(self.classifier.random_state)
                except:
                    pass

                ## Iterate across folds: train->validate, save mean score across folds
                roc_auc_list = list()
                for fold in range(1, num_folds+1):
                    # print('%s:%s'% (fold+1,num_folds), end='\r' )

                    ## Fit ##
                    self.classifier.fit(X_train.iloc[fold_array!=fold], y_train.iloc[fold_array!=fold])

                    ## Validate ##
                    curr_roc_auc_score = roc_auc_score(y_train.iloc[fold_array==fold],
                                                       self.classifier.predict_proba(X_train.iloc[fold_array==fold])[:,1])

                    ## Append score for Fold ##
                    parameter_set_results['split'+str(fold)+'_test_score'].append( curr_roc_auc_score )
                    roc_auc_list.append( curr_roc_auc_score )

                ## Append mean score across folds for this Hyperparameter set ##
                parameter_set_results['mean_test_score'].append( np.mean(roc_auc_list) )

            ## Append mean score alongside hyperparameters ##
            self.parameter_grid = pd.concat( (self.parameter_grid, pd.DataFrame(parameter_set_results)), axis=1 )

            ## Save results to csv ##
            if save_parameter_grid:
                ## Convert integers to ints ##
                if( 'n_estimators' in self.parameter_grid.columns ):
                    self.parameter_grid['n_estimators'] = self.parameter_grid['n_estimators'].astype(int)
                self.parameter_grid.to_csv( os.path.join(foldername, self.model_type + '_hyperparameter_grid.csv'), index=False)

            ## Dictionary with best hyperparameters (maximum mean score across folds) ##
            best_param_dict = dict( self.parameter_grid.iloc[np.argmax(self.parameter_grid['mean_test_score'])] )

            ## Remove non-hyperparameter values ##
            pop_list = list() # list to hold keys that need to be removed
            for key in best_param_dict.keys():
                if key in ['mean_test_score', 'random_state']:
                    pop_list.append(key)
                elif key[:5] == 'split' and key[6:] == '_test_score':
                    pop_list.append(key)
            for key in pop_list:
                best_param_dict.pop(key)
            ## Set best parameters to object variable ##
            self.best_params_ = best_param_dict
            ## Set best classifier with optimized hyperparameters, untrained (best mean score across folds) ##
            self.best_classifier = self.classifier.set_params( **self.best_params_ )

            ## Return best mean score (optimized hyperparameters) ##
            return np.max(self.parameter_grid['mean_test_score'])



    ### Return best classifier object ###
    def return_best_classifier(self):
        return self.best_classifier


    ### Return best set of hyperparameters ###
    def return_best_params(self):
        if self.method == 'fit':
            param_dict = dict( self.parameter_grid.iloc[np.argmax(self.parameter_grid['AUROC'])] )
            try:
                param_dict.pop('AUROC')
            except:
                pass
            try:
                param_dict.pop('random_state')
            except:
                pass
            try:
                param_dict.pop('max_iter')
            except:
                pass

            return param_dict

        elif self.method == 'fit_CV':
            try:
                return self.best_params_
            except:
                return None

