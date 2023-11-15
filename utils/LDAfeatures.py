# Remap feature array 
# Felipe Giuste
# 4/14/2020

#Input: feature_array (row:observation, column:feature), labels (Observation labels)
#Output: Fit LDA #feature_array with column space reduced

#Parameters:
#-feature_array 
#-n_components (Number of components to find)   
#-whiten (Standardize eigenvector magnitudes)
#-demean (set feature means to zero)

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def LDAfeatures( feature_array, labels, n_components=None, demean=True ):

    ## De-mean Channels ##
    if(demean):
        feature_array = feature_array.copy()
        # Mean = 0 for each feature:
        feature_means = np.mean(feature_array, axis=0)
        feature_array = feature_array - feature_means
    
    ## LDA transform ##
    lda = LDA(n_components=n_components, store_covariance=True )
    lda.fit( feature_array, y=labels )

    ## Return LDA Reduced array ##
    return( lda )