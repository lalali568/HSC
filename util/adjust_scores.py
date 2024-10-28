import numpy as np
def adjust_scores(label, score):
    """
    这段代码，我是从COUTA这个模型里面拿过来的，我觉得这个代码的意思是，对于每一个ground-truth的异常段，使用最大的分数作为该段中所有点的分数。
    adjust the score for segment detection. i.e., for each ground-truth anomaly segment,
    use the maximum score as the score of all points in that segment. This corresponds to point-adjust f1-score.
    ** This function is copied/modified from the source code in [Zhihan Li et al. KDD21]

    Parameters
    ----------
        score: np.ndarray
            anomaly score, higher score indicates higher likelihoods to be anomaly
        label: np.ndarray
            ground-truth label

    Return
    ----------
        score: np.ndarray
            adjusted score

    """

    score = score.copy()
    assert len(score) == len(label)
    splits = np.where(label[1:] != label[:-1])[0] + 1#这里的意思是，找到label中，从第二个元素开始，如果当前元素和前一个元素不相等，那么就返回当前元素的索引+1，也就是说，返回的是label中，从第二个元素开始，如果当前元素和前一个元素不相等的元素的索引+1
    is_anomaly = label[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])
    return score
