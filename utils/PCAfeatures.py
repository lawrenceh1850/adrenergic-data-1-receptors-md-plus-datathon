# Find Principal Components within feature array 
# Felipe Giuste
# 4/8/2020

#Input: feature_array (row:observation, column:feature)
#Output: Fit PCA #feature_array with column space reduced

#Parameters:
#-feature_array 
#-n_components (Number of components to find)   
#-whiten (Standardize eigenvector magnitudes)
#-demean (set feature means to zero)

import numpy as np
from sklearn.decomposition import PCA

def PCAfeatures( feature_array, n_components=3, whiten=False, demean=True, random_state=1234 ):
    print( 'Number of PCA Components: %s'% (n_components,) )
    print( 'Whiten: %s'% (whiten,) )
    print( 'Demean: %s'% (demean,) )
    
    ## De-mean Channels ##
    if(demean):
        feature_array = feature_array.copy()
        # Mean = 0 for each feature:
        feature_means = np.mean(feature_array, axis=0)
        feature_array = feature_array - feature_means
    
    ## PCA transform ##
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    pca.fit( feature_array )

    ## Return PCA Reduced array ##
    print( 'PCA Component Shape: %s'% (pca.components_.shape,) )
    return( pca )