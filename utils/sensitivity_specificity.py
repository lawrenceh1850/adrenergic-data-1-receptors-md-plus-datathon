import numpy as np
import matplotlib.pyplot as plt

## Metrics ##
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, f1_score, accuracy_score, classification_report

def plot_sensitivity_specificity(y_true, y_pred, model_type='NN', title=None, auroc_score=None):
    threshold_increment=1e-4
    x_buffer=0.0025
    vlines_text_height = 0.03
    vlines_height = 0.3
    vline_fontsize = 12
    
    ### Calculating sensitivity, specificity, MCC at different thresholds
    thresholds = np.arange(0,1+threshold_increment,threshold_increment)
    n_thresholds = len(thresholds)
    sensitivity_array = list()
    specificity_array = list()
    mcc_array = list()

    ### iterate across thresholds
    for threshold in thresholds:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred>threshold).ravel()

        ## sensitivity
        if (tp+fn) == 0:
            # divide by 0 error
            sensitivity_array.append(np.nan)
        else:
            sensitivity_array.append( tp/(tp+fn) )

        ## specificity
        if (tn+fp) == 0:
            # divide by 0 error
            specificity_array.append(np.nan)
        else:
            specificity_array.append( (tn/(tn+fp)) )

        ## MCC
        if np.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)) == 0:
            # mcc will cause divide by 0 error
            mcc_array.append(np.nan)
        else:
            mcc_array.append( matthews_corrcoef(y_true, y_pred>threshold) )

    ## Find maximum MCC and best threshold
    max_mcc = np.max(np.array(mcc_array)[~np.isnan(mcc_array)])
    best_threshold = thresholds[int(np.median(np.where(np.array(mcc_array) == max_mcc)))]
    

    
    ## figure config
    ## Sensitivity Curve ##
    plt.plot(thresholds, sensitivity_array)
    ## Specificity Curve ##
    plt.plot(thresholds, specificity_array)
    
    ## Title ##
    if title is not None:
        plt.title(title, fontsize=20)
    else:
        plt.title('Sensitivity and Specificity vs Predicted Risk', fontsize=20);
        
    ## Axis Labels ##
    plt.xlabel('Predicted Risk', fontsize=20); 
    plt.ylabel('Sensitivity / Specificity', fontsize=20); 
    plt.xlim([0,1]); plt.ylim([0,1])
    
    ## Legend ##
    if model_type == 'XGB':
        plt.legend(['Sensitivity (Sen)', 'Specificity (Spe)'], fontsize=16, loc='upper right')
    else:
        plt.legend(['Sensitivity (Sen)', 'Specificity (Spe)'], fontsize=16, loc='upper right')

    ## Vertical Lines ##
    # sensitivity = 90%
    plt.vlines(np.where(np.array(sensitivity_array)<0.9)[0][0]/n_thresholds, 0,
               vlines_height, colors='lightgrey')
    plt.text((np.where(np.array(sensitivity_array)<0.9)[0][0])/n_thresholds+x_buffer,
             vlines_text_height,
             'Sen = ' + str( round(sensitivity_array[np.where(np.array(sensitivity_array)<0.9)[0][0] - 1]*100, 2))+'%' +
             '\nSpe = ' + str( round(specificity_array[np.where(np.array(sensitivity_array)<0.9)[0][0] - 1]*100, 2))+'%', 
             fontsize=vline_fontsize, rotation='vertical')

    # sensitivity = 95%
    plt.vlines(np.where(np.array(sensitivity_array)<0.95)[0][0]/n_thresholds, 0,
               vlines_height, colors='darkgrey')
    if model_type == 'NN':
        plt.text((np.where(np.array(sensitivity_array)<0.95)[0][0])/n_thresholds+x_buffer, 
                 vlines_text_height,
                 'Sen = ' + str( round(sensitivity_array[np.where(np.array(sensitivity_array)<0.95)[0][0] - 1]*100, 2))+'%' +
                 '\nSpe = ' + str( round(specificity_array[np.where(np.array(sensitivity_array)<0.95)[0][0] - 1]*100, 2))+'%',
                 fontsize=vline_fontsize, rotation='vertical')
    
    # sensitivity = 99%
    plt.vlines(np.where(np.array(sensitivity_array)<0.99)[0][0]/n_thresholds, 0,
               vlines_height, colors='dimgrey')
    if model_type == 'NN':
        plt.text((np.where(np.array(sensitivity_array)<0.99)[0][0])/n_thresholds+x_buffer, 
                 vlines_text_height,
                 'Sen = ' + str( round(sensitivity_array[np.where(np.array(sensitivity_array)<0.99)[0][0] - 1]*100, 2))+'%' +
                 '\nSpe = ' + str( round(specificity_array[np.where(np.array(sensitivity_array)<0.99)[0][0] - 1]*100, 2))+'%',
                fontsize=vline_fontsize, rotation='vertical')

    # specificity = 90%
    plt.vlines(np.where(np.array(specificity_array)>0.9)[0][0]/n_thresholds, 0,
               vlines_height, colors='lightgrey')
    plt.text((np.where(np.array(specificity_array)>0.9)[0][0])/n_thresholds+x_buffer, 
             vlines_text_height,
             'Sen = ' + str( round(sensitivity_array[np.where(np.array(specificity_array)>0.9)[0][0]]*100, 2))+'%' +
             '\nSpe = ' + str( round(specificity_array[np.where(np.array(specificity_array)>0.9)[0][0]]*100, 2))+'%',
             fontsize=vline_fontsize, rotation='vertical')

    # specificity = 95%
    plt.vlines(np.where(np.array(specificity_array)>0.95)[0][0]/n_thresholds, 0,
               vlines_height, 
               colors='darkgrey')
    plt.text((np.where(np.array(specificity_array)>0.95)[0][0])/n_thresholds+x_buffer,
              vlines_text_height,
             'Sen = ' + str( round(sensitivity_array[np.where(np.array(specificity_array)>0.95)[0][0]]*100, 2))+'%' +
             '\nSpe = ' + str( round(specificity_array[np.where(np.array(specificity_array)>0.95)[0][0]]*100, 2))+'%',
            fontsize=vline_fontsize, rotation='vertical')

    # specificity = 99%
    plt.vlines(np.where(np.array(specificity_array)>0.99)[0][0]/n_thresholds, 0,
               vlines_height, colors='dimgrey')
    plt.text((np.where(np.array(specificity_array)>0.99)[0][0])/n_thresholds+x_buffer,
              vlines_text_height,
             'Sen = ' + str( round(sensitivity_array[np.where(np.array(specificity_array)>0.99)[0][0]]*100, 2))+'%' +
             '\nSpe = ' + str( round(specificity_array[np.where(np.array(specificity_array)>0.99)[0][0]]*100, 2))+'%',
             fontsize=vline_fontsize, rotation='vertical')

    # Maximum MCC
    plt.vlines(best_threshold, 0, 0.6, colors='green')
    plt.text(best_threshold+x_buffer, 0.35,
             'Maximum MCC\nSen = ' + str( round(sensitivity_array[int(best_threshold*n_thresholds)]*100, 2))+'%' +
             '\nSpe = ' + str( round(specificity_array[int(best_threshold*n_thresholds)]*100, 2))+'%',
             fontsize=vline_fontsize, rotation='vertical')

    # sensitivity = specificity
    plt.vlines(np.where(np.array(sensitivity_array) < np.array(specificity_array))[0][0]/n_thresholds, 0,
               vlines_height, colors='purple')
    plt.text((np.where(np.array(sensitivity_array) < np.array(specificity_array))[0][0])/n_thresholds+x_buffer, 
             vlines_text_height,
             'Sen = '+str(round(sensitivity_array[np.where(np.array(sensitivity_array)<np.array(specificity_array))[0][0]]*100, 2))+'%'+
             '\nSpe = '+str( round(sensitivity_array[np.where(np.array(sensitivity_array)< np.array(specificity_array))[0][0]]*100, 2))+'%',
             fontsize=vline_fontsize, rotation='vertical');

    ## xtick positions ##
    if model_type == 'XGB':
        xticks_locs = [
            np.where(np.array(sensitivity_array)<0.99)[0][0]/n_thresholds,
            np.where(np.array(sensitivity_array)<0.95)[0][0]/n_thresholds,
            np.where(np.array(sensitivity_array)<0.9)[0][0]/n_thresholds,
            np.where(np.array(sensitivity_array) < np.array(specificity_array))[0][0]/n_thresholds,
            np.where(np.array(specificity_array)>0.9)[0][0]/n_thresholds,
            best_threshold,
            np.where(np.array(specificity_array)>0.95)[0][0]/n_thresholds,
            np.where(np.array(specificity_array)>0.99)[0][0]/n_thresholds,
            0,0.5,1,1.1
        ]
    else: 
        xticks_locs = [
            np.where(np.array(sensitivity_array)<0.99)[0][0]/n_thresholds,
            np.where(np.array(sensitivity_array)<0.95)[0][0]/n_thresholds,
            np.where(np.array(sensitivity_array)<0.9)[0][0]/n_thresholds,
            np.where(np.array(sensitivity_array) < np.array(specificity_array))[0][0]/n_thresholds,
            np.where(np.array(specificity_array)>0.9)[0][0]/n_thresholds,
            best_threshold,
            np.where(np.array(specificity_array)>0.95)[0][0]/n_thresholds,
            np.where(np.array(specificity_array)>0.99)[0][0]/n_thresholds,
            0,0.5,1,1.1
        ]
        xticks_locs.sort()
    
    ## Prevent tick overlap ##
    xticks_locs = [s for s,t in zip(xticks_locs, xticks_locs[1:]) if abs(t-s)>0.01]
    ## xticks ##
    plt.xticks(ticks=xticks_locs, rotation=45, ha='right', fontsize=12);
    
    ## AUROC ##
    auroc_score = roc_auc_score(y_true=y_true, y_score=y_pred)
    # plt.text(0.85, 0.95, f'AUROC: {auroc_score*100:.2f}%', fontsize=16)
    # plt.tight_layout()
    # plt.savefig('temporary.png', dpi=300)