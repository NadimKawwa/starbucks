import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, confusion_matrix, f1_score, classification_report
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import seaborn as sns




def plot_confusion_matrix(y_true, y_pred, classes=np.arange(0,7),
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues, saveloc=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Inputs:
    - y_true(numpy): array of ground truth
    - y_pred(nump): array of predictions
    - classes(numpy): array of possible classes
    - normalize(bool): normalize output
    - title(string): plot title
    - cmap(pyplot): color scheme
    - saveloc(string): location to save plot
    
    Output:
    - confusion matrix
    - pyplot axis
    
    
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    print("Confusion Matrix: ")
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    if saveloc:
        plt.savefig(saveloc)
        
    
    print("Plotted confusion matrix")
    plt.show()
    
    return ax



def model_evaluation(y_true, y_pred, average='weighted',classes=np.arange(0,7),
                     normalize=False,
                     title=None,
                     cmap=plt.cm.Blues, saveloc=None):
    
    
    """
    Prints out accuracy
    Prints out classification report
    Prints out confusion matrix
    Plots confusion matrix
    
    Inputs:
        - y_true(numpy): array of ground truth
        - y_pred(nump): array of predictions
        - classes(numpy): array of possible classes
        - normalize(bool): normalize output
        - title(string): plot title
        - cmap(pyplot): color scheme
        - saveloc(string): location to save plot

    Output:
        - confusion matrix
        - pyplot axis
    
    
    """
    
    #print accuracy
    accuracy = np.mean(y_true == y_pred)
    print("Accuracy = {:.3f}\n".format(accuracy))
    
    #get classification report
    print("classification report:\n")
    print(classification_report(y_true,y_pred, labels=classes))
    print("\n")
    #generate confusion matrix
    return plot_confusion_matrix(y_true, y_pred, classes,
                                 normalize,
                                 title, cmap, saveloc)
    
    
    
    
def make_svd_heatmap_plot(k, lr, actual_v_pred, save=True):
    sns.heatmap(actual_v_pred, vmin=0, vmax=10);
    plt.xticks(np.arange(10), np.arange(1,11));
    plt.yticks(np.arange(10), np.arange(1,11));
    plt.xlabel("Predicted Values");
    plt.ylabel("Actual Values");
    
    plt.title("Actual vs. Predicted Values, k={} lr={}".format(k, lr))
    save_path = "plots/svd_heatmap_"+"lr"+str(lr)+"_k"+str(k)+".png"
    
    if save:
        plt.savefig(save_path)
    plt.show()



def make_svd_hist_plot(k, lr, acts, preds, save=True):
    
    plt.figure(figsize=(8,8))
    plt.hist(acts, density=True, alpha=.5, label='actual');
    plt.hist(preds, density=True, alpha=.5, label='predicted');
    plt.legend(loc=2, prop={'size': 15});
    plt.xlabel('Times Seen')
    
    plt.title('Predicted vs. Actual, k={}, lr={}'.format(k, lr))
    save_path = "plots/svd_hist_"+"lr"+str(lr)+"_k"+str(k)+".png"
    if save:
        plt.savefig(save_path)
    plt.show()
    
    


