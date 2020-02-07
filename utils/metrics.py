from tqdm import tqdm
import numpy as np
import os


def accuracy(tn, fp, fn, tp):
    return np.nan_to_num((tp+tn)/(tn+fp+fn+tp))


def recall(tn, fp, fn, tp):
    return np.nan_to_num(tp/(tp+fn))


def specifity(tn, fp, fn, tp):
    return np.nan_to_num(tn/(tn+fp))


def precision(tn, fp, fn, tp):
    return np.nan_to_num(tp/(tp+fp))


def f1_score(tn, fp, fn, tp):
    prc = precision(tn, fp, fn, tp)
    rec = recall(tn, fp, fn, tp)
    return np.nan_to_num(2*(prc*rec)/(prc+rec))


def balanced_accuracy(tn, fp, fn, tp):
    spc = specifity(tn, fp, fn, tp)
    rec = recall(tn, fp, fn, tp)
    return np.nan_to_num((1/2)*(spc+rec))


def g_mean(tn, fp, fn, tp):
    spc = specifity(tn, fp, fn, tp)
    rec = recall(tn, fp, fn, tp)
    return np.nan_to_num(np.sqrt(spc*rec))


def mcc(tn, fp, fn, tp):
    return np.nan_to_num(((tp*tn)-(fp*fn))/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))


def calculate_metrics(methods, streams, metrics, experiment_name, recount=False):
    data = {}
    metrics_func = {
                    "accuracy": accuracy,
                    "recall": recall,
                    "specifity": specifity,
                    "precision": precision,
                    "f1_score": f1_score,
                    "balanced_accuracy": balanced_accuracy,
                    "g_mean": g_mean,
                    "mcc": mcc

    }

    for stream_name in streams:
        for clf_name in methods:
            try:
                filename = "results/raw_conf/%s/%s/%s.csv" % (experiment_name, stream_name, clf_name)
                data[stream_name, clf_name] = np.genfromtxt(filename, delimiter=',', dtype=np.int16)
            except Exception:
                data[stream_name, clf_name] = None
                print("Error in loading data", stream_name, clf_name)

    for stream_name in tqdm(streams, "Metrics"):
        for clf_name in methods:
            if data[stream_name, clf_name] is None:
                continue
            for metric in metrics:
                if os.path.exists("results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, stream_name, metric, clf_name)) and not recount:
                    continue
                tn = data[stream_name, clf_name][:, 1]
                fp = data[stream_name, clf_name][:, 2]
                fn = data[stream_name, clf_name][:, 3]
                tp = data[stream_name, clf_name][:, 4]

                result = metrics_func[metric](tn, fp, fn, tp)

                filename = "results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, stream_name, metric, clf_name)

                if not os.path.exists("results/raw_metrics/%s/%s/%s/" % (experiment_name, stream_name, metric)):
                    os.makedirs("results/raw_metrics/%s/%s/%s/" % (experiment_name, stream_name, metric))

                np.savetxt(fname=filename, fmt="%f", X=result)
