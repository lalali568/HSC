from util.spot import SPOT
import numpy as np
from sklearn.metrics import *
def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict



def pot_eval(config,init_score, score, label, q=1e-5, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t
    Returns:
        dict: pot result dict
    """

    if config['model']=='TranAD':
        if config['dataset']=="MSL":
            lm=[0.999,1.04]
            lms = lm[0]
        if config['dataset']=="penism":
            lm = [0.999, 1.04]
            lms = lm[0]
    if config['model']=='AE_basic':
        if config['dataset']=="penism":
            lm = [0.999, 1.04]
            lms = lm[0]
    if config['model']=='GRELEN':
        if config['dataset']=="SWAT":
            lm=[0.999,1.04]
            lms = lm[0]
        if config['dataset']=="penism":
            lm = [0.999, 1.04]
            lms = lm[0]
    if config['model']=='COUTA':
        lm= [0.999, 1.04]
        lms = lm[0]
    if config['model']=="HSR":
        lm=[0.999,1.04]
        lms = lm[0]
    while True:
        try:
            s = SPOT(q)  # SPOT object
            s.fit(init_score, score)  # data import
            s.initialize(level=lms, min_extrema=False, verbose=False)  # initialization step
        except: lms = lms * 0.999
        else: break
    ret = s.run(dynamic=False)  # run
    pot_th = np.mean(ret['thresholds']) * lm[1]


    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    return pred,label,pot_th
    p_t = calc_point2point(pred, label)
    # print('POT result: ', p_t, pot_th, p_latency)
    return {
        'f1': p_t[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
        'ROC/AUC': p_t[7],
        'threshold': pot_th,
        # 'pot-latency': p_latency
    }, np.array(pred)