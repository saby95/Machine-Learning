import numpy as np


def flatten_to_columns(X):

    if len(X) == 0:
        return np.array([])
    num_el = 1
    for i in range(0, np.ndim(X[0])):
        num_el = num_el * X[0].shape[i]
    ret_matrix = np.empty([num_el,0],dtype=X[0].dtype)
    for row in X:
        ret_matrix = np.append(ret_matrix, row.reshape(-1,1), axis=1)
    return np.asmatrix(ret_matrix)


class PCA:

    def __init__(self, num_components = 0):
        object.__init__(self)
        self.num_components = num_components

    def compute(self, X, Y):
        X_flat = flatten_to_columns(X)
        Y = np.asarray(Y)

        if self.num_components<=0 or (self.num_components > X_flat.shape[1]-1):
            self.num_components = X_flat.shape[1]-1

        self.mean = X_flat.mean(axis=1).reshape(-1,1)

        X_flat = X_flat - self.mean

        self.eigenvectors, self.eigenvalues, variances = np.linalg.svd(X_flat, full_matrices=False)

        srt = np.argsort(-self.eigenvalues)
        self.eigenvalues, self.eigenvectors = self.eigenvalues[srt], self.eigenvectors[:,srt]

        self.eigenvectors = self.eigenvectors[0:,0:self.num_components].copy()
        self.eigenvalues = self.eigenvalues[0:self.num_components].copy()

        self.eigenvalues = np.power(self.eigenvalues,2) / X_flat.shape[1]

        features = []
        for x in X:
            projected = self.project(x.reshape(-1,1))
            features.append(projected)
        return features

    def project(self, X):
        X = X - self.mean
        return np.dot(self.eigenvectors.T, X)

    def extract(self, X):
        X = np.asarray(X).reshape(-1,1)
        return self.project(X)

    def reconstruct(self, X):
        X = np.dot(self.eigenvectors, X)
        return X + self.mean


class LDA:

    def __init__(self, num_components = 0):
        object.__init__(self)
        self.num_components = num_components

    def compute(self, X, Y):
        X_flattened = flatten_to_columns(X)
        Y = np.asarray(Y)

        d = X_flattened.shape[0]
        c = len(np.unique(Y))

        if self.num_components <= 0 or (self.num_components > (c-1)):
            self.num_components = c-1

        mean_total = X_flattened.mean(axis=1).reshape(-1,1)
        Sw = np.zeros((d,d), dtype=np.float32)
        Sb = np.zeros((d,d), dtype=np.float32)

        for i in range(1,c+1):
            Xi = []
            Xi = X_flattened[:,np.where(Y==i)[0]]
            meanClass = np.mean(Xi, axis=1)
            Sw = Sw + np.dot((Xi-meanClass), (Xi-meanClass).T)
            Sb = Sb + Xi.shape[1] * np.dot((meanClass - mean_total), (meanClass - mean_total).T)

        self.eigenvalues, self.eigenvectors = np.linalg.eig(np.linalg.inv(Sw)*Sb)
        srt = np.argsort(-self.eigenvalues.real)
        self.eigenvalues, self.eigenvectors = self.eigenvectors[srt], self.eigenvectors[:,srt]
        features = []
        for x in X:
            xp = self.project(x.reshape(-1,1))
            features.append(xp)
        return features

    def project(self, X):
        return np.dot(self.eigenvectors.T, X)

    def reconstruct(self, X):
        return np.dot(self.eigenvectors, X)


class ChainOperator:

    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def compute(self, X, Y):
        X = self.model1.compute(X, Y)
        return self.model2.compute(X, Y)

    def extract(self, X):
        X = self.model1.extract(X)
        return self.model2.extract(X)


class Fisherfaces:

    def __init__(self, num_components = 0):
        object.__init__(self)
        self.num_components = num_components

    def train(self, X, Y):
        X_flattened = flatten_to_columns(X)
        Y = np.asarray(Y)

        n = len(Y)
        c = len(np.unique(Y))

        pca = PCA(num_components=(n-c))
        lda = LDA(num_components=self.num_components)

        model = ChainOperator(pca,lda)
        model.compute(X,Y)

        self.eigenvalues = lda.eigenvalues
        self.num_components = lda.num_components

        self.eigenvectors = np.dot(pca.eigenvectors,lda.eigenvectors)

        features = []
        for x in X:
            xp = self.project(x.reshape(-1,1))
            features.append(xp)
        return features

    def extract(self,X):
        X = np.asarray(X).reshape(-1,1)
        return self.project(X)

    def project(self,X):
        return np.dot(self.eigenvectors.T, X)

    def reconstruct(self,X):
        return np.dot(self.eigenvectors, X)
