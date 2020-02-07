from sklearn.base import BaseEstimator
import numpy as np
from utils import minority_majority_split, minority_majority_name
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score
from sklearn.base import clone


class OCEIS(BaseEstimator):

    def __init__(self, base_classifier=OneClassSVM(nu=0.1), number_of_classifiers=10, cluster_method=KMeans):
        self.base_classifier = base_classifier
        self.number_of_classifiers = number_of_classifiers
        self.cluster_method = cluster_method

        self.drift_detector = None

        self.metrics_array = []
        self.classifier_array_maj = []
        self.classifier_array_min = []
        self.weights_array_min = []
        self.weights_array_maj = []
        self.number_of_features = None

        self.minority_name = None
        self.majority_name = None
        self.classes = None
        self.label_encoder = None
        self.iteration = 0

    def partial_fit(self, X, y, classes=None):

        # Initial preperation
        if classes is None and self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
            self.classes = self.label_encoder.classes
        elif self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(classes)
            self.classes = classes

        if classes[0] is "positive":
            self.minority_name = self.label_encoder.transform(classes[0])
            self.majority_name = self.label_encoder.transform(classes[1])
        elif classes[1] is "positive":
            self.minority_name = self.label_encoder.transform(classes[1])
            self.majority_name = self.label_encodr.transform(classes[0])

        y = self.label_encoder.transform(y)

        if self.minority_name is None or self.majority_name is None:
            self.minority_name, self.majority_name = minority_majority_name(y)
            self.number_of_features = len(X[0])

        # Split data
        minority, majority = minority_majority_split(X, y, self.minority_name, self.majority_name)

        samples, n_of_clust = self._best_number_of_clusters(minority, 10)
        clfs = []
        for i in range(n_of_clust):
            clfs.append(clone(self.base_classifier).fit(samples[i]))
        self.classifier_array_min.append(clfs)

        samples, n_of_clust = self._best_number_of_clusters(majority, 10)
        clfs = []
        for i in range(n_of_clust):
            clfs.append(clone(self.base_classifier).fit(samples[i]))
        self.classifier_array_maj.append(clfs)

        if len(self.classifier_array_maj) > self.number_of_classifiers:
            del self.classifier_array_maj[-1]
        if len(self.classifier_array_min) > self.number_of_classifiers:
            del self.classifier_array_min[-1]

    def _best_number_of_clusters(self, data, kmax=10):

        sil_values = []
        clusters = []

        for k in range(2, kmax+1):
            try:
                kmeans = KMeans(n_clusters=k)
                labels = kmeans.fit_predict(data)
                clusters.append(labels)
                sil_values.append(silhouette_score(data, labels, metric='euclidean'))
            except Exception:
                break

        best_number = np.argmax(sil_values)
        n_of_clust = best_number+2
        samples = [[] for i in range(n_of_clust)]

        for i, x in enumerate(clusters[best_number]):
            samples[x].append(data[i].tolist())

        return samples, n_of_clust

    def predict(self, X):
        predictions = []
        for clfs_min, clfs_maj in zip(self.classifier_array_min, self.classifier_array_maj):
            probas_min = np.max([clf.decision_function(X) for clf in clfs_min], axis=0)
            probas_maj = np.max([clf.decision_function(X) for clf in clfs_maj], axis=0)
            probas_ = np.stack((probas_maj, probas_min), axis=1)
            predictions.append(np.argmax(probas_, axis=1))

        predict = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=np.asarray(predictions).T)
        return predict
