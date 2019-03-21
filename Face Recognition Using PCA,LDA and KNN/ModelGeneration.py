import _pickle as cPickle


class PredictableModel(object):

    def __init__(self, feature, classifier):
        self.feature = feature
        self.classifier = classifier

    def compute(self, X, Y):
        featues = self.feature.train(X, Y)
        return self.classifier.compute(featues, Y)

    def predict(self, X):
        q = self.feature.extract(X)
        return self.classifier.predict(q)


def save_model(filename, model):
    output = open(filename, 'wb')
    cPickle.dump(model, output)
    output.close()


def load_model(filename):
    pkl_file = open(filename, 'rb')
    res = cPickle.load(pkl_file)
    pkl_file.close()
    return res
