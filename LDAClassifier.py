import numpy as np
import scipy
import scipy.linalg
from sklearn.utils.extmath import safe_sparse_dot

np.set_printoptions(precision=5)
np.seterr(all='raise')

class LDAClassifier:
    def __init__ (self, type="offline", regularization_coefficient=0.01, max_num_of_features = 50):
        if type not in ["offline", "online"]:
            raise Exception("Non-valid type")

        # attributes, that do not change during add_feature
        if regularization_coefficient is not None:
            if not str(regularization_coefficient).replace('.','',1).isdigit():
                raise Exception("Non-valid regularization coefficient (not numeric): ", regularization_coefficient)
            if regularization_coefficient <= 0 or regularization_coefficient >= 1:
                raise Exception("Non-valid regularization coefficient")
        self.regularization_coefficient = regularization_coefficient
        self.type = type
        # for preallocating
        self.max_num_of_features = max_num_of_features
        # classes labels
        self.class_labels = []
        # number of classes, len(np.unique(self.y))
        self.num_classes = 0
        # training target variable
        self.y = []
        # overall number of observations
        self.overall_num_obs = 0
        # number of observations by class
        self.num_obs_by_class = []
        # posterior probabilities by class
        self.prob_classes = []
        # natural logarithm of posterior probabilities by class
        self.log_prob_classes = []

        # attributes, that do change during add_feature
        # mean vectors by class
        self.mean_vecs_by_class = []
        # total covariance matrix
        self.total_cov_mat = []
        # training data: rows = observations, columns = features
        self.X = []
        # number of features, self.X.shape[1]
        self.num_features = 0
        # lower triangular matrix (Cholesky decomposition)
        self.L = []
        # (inverse of lower triangular matrix) * (mean vectors by class)
        self.L_inv_mv_by_class = []
        # (inverse of total covariance matrix) * (mean vectors by class)
        self.total_cov_mat_inv_mv_by_class = []
        # -0.5 * (mean vectors by class)^T * (inverse of total covariance matrix) * (mean vectors by class) + natural logarithm of posterior probabilities by class
        self.mv_total_cov_mat_inv_mv_by_class = []
        # results of discriminant function
        self.scores = []

    # training the model
    def fit(self, X, y):
        if X.ndim == 1:
            X = X[np.newaxis, :]
        if y.shape[0] != X.shape[0]:
            X = X.T
        self.class_labels = np.unique(y)
        self.num_classes = self.class_labels.shape[0]
        self.y = y
        self.overall_num_obs = X.shape[0]
        self.num_features = X.shape[1]
        self.num_obs_by_class = []

        if self.type == "offline":
            self.max_num_of_features = self.num_features
        # preallocating
        self.total_cov_mat = np.zeros((self.max_num_of_features, self.max_num_of_features))
        self.L = np.zeros((self.max_num_of_features, self.max_num_of_features))
        self.num_obs_by_class = np.zeros(self.num_classes)
        self.mean_vecs_by_class = np.zeros((self.num_classes, self.max_num_of_features))
        self.X = np.zeros((self.overall_num_obs, self.max_num_of_features))
        self.total_cov_mat_inv_mv_by_class = np.zeros((self.num_classes, self.max_num_of_features))
        self.L_inv_mv_by_class = np.zeros((self.num_classes, self.max_num_of_features))
        self.mv_total_cov_mat_inv_mv_by_class = np.zeros(self.num_classes)

        for cl, n_cl in zip(self.class_labels, range(0, self.num_classes)):
            mean = np.mean(X[self.y == cl], axis=0)
            self.mean_vecs_by_class[n_cl, :self.num_features] = mean
            self.num_obs_by_class[n_cl] = X[self.y == cl].shape[0]

        self.X[:, :self.num_features] += X
        self.prob_classes = self.prob()
        self.log_prob_classes = np.log(self.prob_classes)
        self.total_covariance_matrix()
        self.L[:self.num_features, :self.num_features] = self.cholesky_decomposition()

        for n_cl in range(self.num_classes):
            self.L_inv_mv_by_class[n_cl, :self.num_features] = self.forward_substitution(self.L[:self.num_features, :self.num_features], self.mean_vecs_by_class[n_cl, :self.num_features])
            self.total_cov_mat_inv_mv_by_class[n_cl, :self.num_features] = self.backward_substitution(self.L[:self.num_features, :self.num_features].T, self.L_inv_mv_by_class[n_cl, :self.num_features])
            #self.mv_total_cov_mat_inv_mv_by_class[n_cl] = -0.5*(self.mean_vecs_by_class[n_cl, :self.num_features].T@self.total_cov_mat_inv_mv_by_class[n_cl, :self.num_features])+self.log_prob_classes[n_cl]
            #self.mv_total_cov_mat_inv_mv_by_class[n_cl] = -(np.sum(self.mean_vecs_by_class[n_cl, :self.num_features]*self.total_cov_mat_inv_mv_by_class[n_cl, :self.num_features]))/2+self.log_prob_classes[n_cl]
            #self.mv_total_cov_mat_inv_mv_by_class[n_cl] = -0.5*scipy.linalg.blas.ddot(x=self.mean_vecs_by_class[n_cl, :self.num_features].T, y=self.total_cov_mat_inv_mv_by_class[n_cl, :self.num_features])+self.log_prob_classes[n_cl]
            self.mv_total_cov_mat_inv_mv_by_class[n_cl] = -0.5 * (np.dot(self.mean_vecs_by_class[n_cl, :self.num_features], self.total_cov_mat_inv_mv_by_class[n_cl, :self.num_features].T)) + self.log_prob_classes[n_cl]

        return self

    # add a new training feature and update the internal structure
    def add_feature(self, new_feature):
        self.num_features = self.num_features + 1
        self.X[:, self.num_features-1] = new_feature
        new_row_total = np.zeros(self.num_features)

        for cl, n_cl in zip(self.class_labels, range(0, self.num_classes)):
            new_feature_class = new_feature[self.y == cl]

            new_mean = np.mean(new_feature_class, axis=0)
            self.mean_vecs_by_class[n_cl, self.num_features-1] = new_mean

            X = self.X[:, :self.num_features][self.y == cl] - self.mean_vecs_by_class[n_cl, :self.num_features]
            new_row = scipy.linalg.blas.dgemv(alpha=1/self.num_obs_by_class[n_cl], a=X.T, x=(new_feature_class-new_mean))
            if self.regularization_coefficient is not None:
                new_row_total += self.prob_classes[n_cl] * (self.regularization_update(new_row))
            else:
                new_row_total += self.prob_classes[n_cl] * (new_row)
        self.total_cov_mat[self.num_features-1, :self.num_features] = new_row_total
        self.cholesky_decomposition_update()
        for n_cl in range(self.num_classes):
            self.L_inv_mv_by_class[n_cl, self.num_features-1] = (self.mean_vecs_by_class[n_cl, self.num_features-1] - np.sum(self.L[self.num_features-1][:self.num_features-1] * self.L_inv_mv_by_class[n_cl, :self.num_features-1])) / (self.L[self.num_features-1, self.num_features-1])
            self.total_cov_mat_inv_mv_by_class[n_cl, :self.num_features] = self.backward_substitution(self.L[:self.num_features, :self.num_features].T, self.L_inv_mv_by_class[n_cl, :self.num_features])
            #self.mv_total_cov_mat_inv_mv_by_class[n_cl] = -0.5*(self.mean_vecs_by_class[n_cl, :self.num_features].T@self.total_cov_mat_inv_mv_by_class[n_cl, :self.num_features])+self.log_prob_classes[n_cl]
            #self.mv_total_cov_mat_inv_mv_by_class[n_cl] = -0.5 * (np.sum(self.mean_vecs_by_class[n_cl, :self.num_features] * self.total_cov_mat_inv_mv_by_class[n_cl,:self.num_features])) + self.log_prob_classes[n_cl]
            #self.mv_total_cov_mat_inv_mv_by_class[n_cl] = -0.5*scipy.linalg.blas.ddot(x=self.mean_vecs_by_class[n_cl, :self.num_features].T, y=self.total_cov_mat_inv_mv_by_class[n_cl, :self.num_features])+self.log_prob_classes[n_cl]
            self.mv_total_cov_mat_inv_mv_by_class[n_cl] = -0.5 * (np.dot(self.mean_vecs_by_class[n_cl, :self.num_features], self.total_cov_mat_inv_mv_by_class[n_cl, :self.num_features].T)) + self.log_prob_classes[n_cl]




        return self

    # pooled total covariance matrix
    def total_covariance_matrix(self):
        for i in range(self.num_classes):
            X = self.X[:, :self.num_features][self.y == self.class_labels[i]] - np.array(self.mean_vecs_by_class[i][:self.num_features])
            if self.regularization_coefficient is not None:
                self.total_cov_mat[:self.num_features, :self.num_features] += self.prob_classes[i]*self.regularization(scipy.linalg.blas.dsyrk(1/self.num_obs_by_class[i], X.T, lower=True))
            else:
                self.total_cov_mat[:self.num_features, :self.num_features] += scipy.linalg.blas.dsyrk(self.prob_classes[i] / self.num_obs_by_class[i], X.T, lower=True)
        return self.total_cov_mat

    def regularization(self, cov_mat):
        if self.regularization_coefficient is not None:
            return (1-self.regularization_coefficient)*cov_mat+self.regularization_coefficient*np.eye(self.num_features)

    def regularization_update(self, new_row):
        if self.regularization_coefficient is not None:
            new_row = (1-self.regularization_coefficient)*new_row
            new_row[self.num_features-1] += self.regularization_coefficient
        return new_row

    def cholesky_decomposition(self):
        return scipy.linalg.cholesky(self.total_cov_mat[:self.num_features, :self.num_features], lower=True)

    def cholesky_decomposition_update(self):
        new_row = scipy.linalg.blas.dtrsv(a=self.L[:self.num_features-1, :self.num_features-1], x=self.total_cov_mat[self.num_features - 1,:self.num_features], trans=0, lower=1, diag=0)
        self.L[self.num_features-1, :self.num_features-1] = new_row[:self.num_features-1]

        sumk = np.sum(self.L[self.num_features - 1, :self.num_features-1] ** 2)
        new_row[self.num_features-1] = np.sqrt(self.total_cov_mat[self.num_features-1, self.num_features-1] - sumk)
        self.L[self.num_features-1, :self.num_features] = new_row

    def forward_substitution(self, L, y):
        x = scipy.linalg.blas.dtrsv(a=L[:self.num_features, :self.num_features], x=y, trans=0, lower=1, diag=0)
        return x

    def backward_substitution(self, U, y):
        x = scipy.linalg.blas.dtrsv(a=U, x=y, trans=0, lower=0, diag=0)
        return x

    def prob(self):
        return np.array(self.num_obs_by_class) / self.overall_num_obs

    # for comparing with sklearn.covariance_
    def total_covariance(self):
        total_cov_mat = np.zeros((self.num_features, self.num_features))
        total_cov_mat += self.total_cov_mat[:self.num_features, :self.num_features]
        for i in range(self.num_features):
            total_cov_mat[i,i] = 0
        return total_cov_mat.T+self.total_cov_mat[:self.num_features, :self.num_features]

    # for comparing with sklearn.covariance_
    def means_by_class(self):
        return self.mean_vecs_by_class[:, :self.num_features]

    # for comparing with sklearn.covariance_
    def coef(self):
        return self.total_cov_mat_inv_mv_by_class[:, :self.num_features]

    # for comparing with sklearn.covariance_
    def intercept(self):
        return self.mv_total_cov_mat_inv_mv_by_class

    # prediction = class with a highest prior probability/result of the discriminant function
    # discriminant function(x, k) = log of proir probability of class k \
    # − 0.5 * (mean of class k)^T * (inverse of total covariance matrix) * (mean of class k)
    # + x^T * (inverse of total covariance matrix) * (mean of class k)
    # based on Bayes decision rule
    def predict(self, data_to_predict):
        #self.scores = np.full((len(data_to_predict), self.num_classes), np.nan)  # pre-allocated
        self.scores = scipy.linalg.blas.dgemm(alpha=1, a=data_to_predict, b=self.total_cov_mat_inv_mv_by_class[:, :self.num_features].T)+self.mv_total_cov_mat_inv_mv_by_class

        return self.class_labels[np.argmax(self.scores, axis=1)]
