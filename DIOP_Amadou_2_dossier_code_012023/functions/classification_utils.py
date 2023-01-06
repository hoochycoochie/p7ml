from sklearn.metrics import roc_curve, auc, adjusted_rand_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%matplotlib inline
import seaborn as sns
from sklearn.metrics import confusion_matrix


def grid_search_cv_multiclass(
    model, param_grid, scoring, cv, label_encoder, X_train, X_test, y_train, y_test
):
    gr = GridSearchCV(model, cv=cv, param_grid=param_grid, scoring=scoring)
    gr.fit(X_train, y_train)
    classes = gr.best_estimator_.classes_

    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))

    pred_prob = gr.best_estimator_.predict_proba(X_test)

    labels = gr.best_estimator_.predict(X_test)
    # print('label1',labels)
    labels_inverse = np.array(
        list(map(lambda label: label_encoder.inverse_transform([label])[0], labels))
    )
    # print('label2',labels)
    fpr = {}
    tpr = {}
    thresh = {}
    roc_auc = dict()

    n_class = classes.shape[0]
    fig = go.Figure()
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:, i], pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        name = "%s vs Rest (AUC=%0.2f)" % (
            label_encoder.inverse_transform([classes[i]])[0],
            roc_auc[i],
        )
        fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], name=name, mode="lines"))

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain="domain"),
        width=1000,
        height=800,
    )
    fig.show()
    mean_auc = 0
    for val in roc_auc.values():
        mean_auc += val
    mean_auc = mean_auc / len(roc_auc)
    ## Ajout graphique de clusterization
    ARI = np.round(adjusted_rand_score(y_test, labels), 4)

    # text_utils.TSNE_visu_fct(X_test,y_test,all_labels,labels,ARI,title1,title2,title3,labels_inverse)
    y_true = y_test
    y_pred = gr.best_estimator_.predict(X_test)
    y_true_label = np.array(
        list(map(lambda label: label_encoder.inverse_transform([label])[0], y_true))
    )

    y_pred_label = np.array(
        list(map(lambda label: label_encoder.inverse_transform([label])[0], y_pred))
    )
    # plot_cm(y_true, y_pred,labels_inverse)

    # skplt.metrics.plot_confusion_matrix(
    # y_true_label,
    # y_pred_label,
    # figsize=(12,12))

    plot_cm(y_true_label, y_pred_label)
    print("ARI", ARI)
    print("mean_auc", mean_auc)
    return mean_auc, ARI, gr


def grid_search_cv_binaryclass(
    model, param_grid, scoring, cv, X_train, X_test, y_train, y_test
):
    gr = GridSearchCV(model, cv=cv, param_grid=param_grid, scoring=scoring)
    gr.fit(X_train, y_train)

    labels = gr.best_estimator_.predict(X_test)

    y_pred_proba = gr.best_estimator_.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    # create ROC curve
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc=4)
    plt.show()

    ## Ajout graphique de clusterization
    ARI = np.round(adjusted_rand_score(y_test, labels), 4)

    # text_utils.TSNE_visu_fct(X_test,y_test,all_labels,labels,ARI,title1,title2,title3,labels_inverse)
    y_true = y_test
    y_pred = gr.best_estimator_.predict(X_test)

    # plot_cm(y_true, y_pred,labels_inverse)

    # skplt.metrics.plot_confusion_matrix(
    # y_true_label,
    # y_pred_label,
    # figsize=(12,12))

    plot_cm(y_true, y_pred)
    print("ARI", ARI)

    return ARI, gr


def plot_cm(y_true, y_pred, figsize=(10, 10)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    # cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = "%.1f%%\n%d/%d" % (p, c, s)
            elif c == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = "%.1f%%\n%d" % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    # cm = pd.DataFrame(cm, index=labels, columns=np.unique(y_true))
    cm.head()
    cm.index.name = "Actual"
    cm.columns.name = "Predicted"
    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt="", ax=ax)
