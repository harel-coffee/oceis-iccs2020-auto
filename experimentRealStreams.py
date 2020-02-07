from ensembles import KMeanClustering
from ensembles import LearnppCDS
from ensembles import LearnppNIE
from ensembles import REA
from ensembles import OUSE
from ensembles import DeterministicSamplingEnsemble
from ensembles import OCEIS
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from skmultiflow.trees import HoeffdingTree
from sklearn.svm import SVC
from utils import evaluation
from scipy.io import arff

from joblib import Parallel, delayed
import time
import logging
import warnings
import traceback
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.simplefilter("ignore")


def load_data(stream_name):
    logging.basicConfig(filename='realStreams.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')

    try:
        data, meta = arff.loadarff("streams/%s.arff" % stream_name)
        classes = meta[meta.names()[-1]][1]
        return data, classes
    except Exception as ex:
        print("Exception")
        logging.exception("Load data exception - %s", stream_name)


def evaluate_method(classifier, stream_name, method_name, initial_size, step_size):

    logging.basicConfig(filename='realStreams.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')

    try:
        logging.info("Start %s %s", stream_name, method_name)
        print(stream_name, method_name)
        start = time.time()

        data, meta = arff.loadarff("streams/%s.arff" % stream_name)
        if data is None:
            print("Empty data")
            raise Exception

        classes = meta[meta.names()[-1]][1]
        evl = evaluation.Evaluation(classifier=classifier, stream_name="%s" % stream_name, method_name=method_name, experiment_name=experiment_name, tqdm=False)
        evl.test_and_train(data=data, classes=classes, initial_size=initial_size, step_size=step_size)
        evl.save_to_csv_confmat()
        logging.info("End %s %s %f", stream_name, method_name, time.time()-start)
        print("End", stream_name, method_name, time.time()-start)

    except Exception as ex:
        logging.exception("Exception in %s %s", stream_name, method_name)
        print(str(ex))
        traceback.print_exc()
        print("Exception in %s %s" % (stream_name, method_name))


base_classifiers = [SVC(probability=True)]
experiment_names = ["svm"]

for base_classifier, experiment_name in zip(base_classifiers, experiment_names):

    methods = [
               KMeanClustering(base_classifier=base_classifier),
               LearnppCDS(base_classifier=base_classifier),
               LearnppNIE(base_classifier=base_classifier),
               REA(base_classifier=base_classifier),
               OUSE(base_classifier=base_classifier),
               OCEIS(),
               MLPClassifier(hidden_layer_sizes=(25,)),
               ]

    names = [
             "KMeanClustering",
             "LearnppCDS",
             "LearnppNIE",
             "REA",
             "OUSE",
             "OCEIS",
             "MLPClassifier",
            ]


    step_sizes = [
                  250
                  500,
                  100,
                  100,
                  50,
                  250,
                  250,
                  500,
                  100,
                  100,
                  100,
                  150,
                  150,
                  50,
                  50,
                  50,
                  52,
                  30,
                  30,
                  30,
                  100,
                  100,
                  100,
                  70,
                  60,
                  60,
                  1000,
                  1000,
                  1000,
                  1000,
                  ]

    initial_sizes = [
                     500
                     1000,
                     200,
                     200,
                     100,
                     500,
                     500,
                     1000,
                     200,
                     200,
                     200,
                     300,
                     300,
                     100,
                     100,
                     100,
                     52,
                     60,
                     60,
                     60,
                     200,
                     200,
                     200,
                     140,
                     120,
                     120,
                     2000,
                     2000,
                     2000,
                     2000,
                    ]

    streams = []                                                # step init
    streams += ["real/abalone-17_vs_7-8-9-10"]                  # 250 500
    streams += ["real/elecNormNew"]                             # 500 1000
    streams += ["real/jm1"]                                     # 100 200
    streams += ["real/kc1"]                                     # 100 200
    streams += ["real/kc2"]                                     # 50 100
    streams += ["real/kr-vs-k-three_vs_eleven"]                 # 250 500
    streams += ["real/kr-vs-k-zero-one_vs_draw"]                # 250 500
    streams += ["real/page-blocks0"]                            # 500 1000
    streams += ["real/segment0"]                                # 100 200
    streams += ["real/shuttle-1vs4"]                            # 100 200
    streams += ["real/vehicle0"]                                # 100 200
    streams += ["real/yeast1"]                                  # 150 300
    streams += ["real/yeast3"]                                  # 150 300
    streams += ["real/wisconsin"]                               # 50 100
    streams += ["real/australian"]                              # 50 100
    streams += ["real/pima"]                                    # 50 100
    streams += ["real/heart"]                                   # 52 52
    streams += ["real/glass0"]                                  # 30 60
    streams += ["real/glass-0-1-2-3_vs_4-5-6"]                  # 30 60
    streams += ["real/glass1"]                                  # 30 60
    streams += ["real/yeast-0-2-5-7-9_vs_3-6-8"]                # 100 200
    streams += ["real/vowel0"]                                  # 100 200
    streams += ["real/yeast-0-2-5-6_vs_3-7-8-9"]                # 100 200
    streams += ["real/yeast-0-3-5-9_vs_7-8"]                    # 70 140
    streams += ["real/yeast-2_vs_4"]                            # 60 120
    streams += ["real/yeast-0-5-6-7-9_vs_4"]                    # 60 120
    streams += ["real/shuttle-5vsA"]                            # 1000 2000
    streams += ["real/shuttle-1vsA"]                            # 1000 2000
    streams += ["real/shuttle-4-5vsA"]                          # 1000 2000
    streams += ["real/shuttle-4vsA"]                            # 1000 2000



    print("Start")
    start = time.time()


    Parallel(n_jobs=-1)(
        delayed(evaluate_method)(clone(classifier), stream_name, name, initial_size, step_size)
            for classifier, name in zip(methods,names) for stream_name,initial_size,step_size in zip(streams,initial_sizes,step_sizes))


    end = time.time()
    print("End %f" % (end-start))
