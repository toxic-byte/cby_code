import numpy as np
from sklearn.metrics import precision_recall_fscore_support

#evaluation metrics
def calACC(hypo, gold):
    _correct = 0
    for i in range(0,len(hypo)):
        _hypo = np.array(hypo[i].cpu())
        _gold = np.array(gold[i].cpu())
        _hypo[_hypo>=0.5] = 1
        _hypo[_hypo<0.5] = 0
        correct = (_hypo == _gold).sum()
        total = _gold.shape[0]
        correct = correct/total
        _correct += correct
    total = len(gold)
    return _correct/total


def calF(hypo, gold, return_all=False):
    b_f1 = ma_f1 = mi_f1 = 0
    b_p = b_r= 0
    for i in range(0,len(hypo)):
        _hypo = np.array(hypo[i].cpu())
        _gold = np.array(gold[i].cpu())
        _hypo[_hypo>=0.5] = 1
        _hypo[_hypo<0.5] = 0
        p,r,f,_ = precision_recall_fscore_support(_gold, _hypo, average='weighted', zero_division=1)
        b_f1 += f
        b_p += p
        b_r += r
        p,r,f,_ = precision_recall_fscore_support(_gold, _hypo, average='macro', zero_division=1)
        ma_f1 += f
        p,r,f,_ = precision_recall_fscore_support(_gold, _hypo, average='micro', zero_division=1)
        mi_f1 += f
    total = len(hypo)
    return b_f1/total, b_p/total, b_r/total, ma_f1/total, mi_f1/total

def evaluate_annotations(gold, hypo):
    """
    Computes Fmax
    """
    total = 0
    p = 0.0
    r = 0.0
    p_total= 0
    prec_list=[0]
    rec_list=[0]
    for i in range(0,len(hypo)):
        _hypo = np.array(hypo[i].cpu())
        _gold = np.array(gold[i].cpu())
        real_num = np.sum(_gold == 1)
        _hypo[_hypo>=0.5] = 1
        _hypo[_hypo<0.5] = 0
        pred_num = np.sum(_hypo == 1)
        if real_num == 0 or pred_num == 0:
            continue
        tpn = np.sum((_gold == 1) & (_hypo == 1))
        fpn = np.sum((_gold == 0) & (_hypo == 1))
        fnn = np.sum((_gold == 1) & (_hypo == 0))
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if pred_num > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
        if i % 100 == 0 or i == len(hypo)-1:
            if p_total > 0 and total > 0:
                prec_list.append(p/ p_total)
                rec_list.append(r/ total)
    if total != 0:
        r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    prec_list = np.array(prec_list)
    rec_list = np.array(rec_list)
    sorted_index = np.argsort(rec_list)
    rec_list = rec_list[sorted_index]
    prec_list = prec_list[sorted_index]
    aupr = np.trapz(prec_list, rec_list)
    return f, p, r, aupr