import numpy as np
from sklearn import metrics

def get_best_f1(label, score):
    precision, recall, ths = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    best_th = ths[np.argmax(f1)]
    return best_f1, best_p, best_r, best_th

def get_metrics(label, score):
    best_f1, best_p, best_r, _ = get_best_f1(label, score)

    return best_f1, best_p, best_r