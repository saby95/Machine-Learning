import numpy as np
import operator as op

class EuclideanDistance(object):

    def __call__(self, p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return np.sqrt(np.sum(np.power((p-q),2)))


class NearestNeighbour(object):

    def __init__(self, dist_metic=EuclideanDistance(),k=1):
        self.k = k
        self.dist_metric = dist_metic
        self.X = []
        self.Y = np.array([],dtype=np.int32)

    def update(self, X, Y):
        self.X.append(X)
        self.Y = np.append(self.Y, Y)

    def compute(self, X, Y):
        self.X = X
        self.Y = np.asarray(Y)

    def predict(self,q):
        distances = []
        for xi in self.X:
            xi = xi.reshape(-1,1)
            d = self.dist_metric(xi, q)
            distances.append(d)
        if len(distances) > len(self.Y):
            raise Exception('Distance Exception')
        distances = np.asarray(distances)
        idx = np.argsort(distances)
        sorted_Y = self.Y[idx]
        sorted_distances = distances[idx]
        sorted_Y = sorted_Y[0:self.k]
        sorted_distances = sorted_distances[0:self.k]
        hist = dict((key,val) for key, val in enumerate(np.bincount(sorted_Y)) if val)
        predicted_label = max(hist.items(), key=op.itemgetter(1))[0]
        return [predicted_label, {'labels' : sorted_Y, 'distances' : sorted_distances}]
