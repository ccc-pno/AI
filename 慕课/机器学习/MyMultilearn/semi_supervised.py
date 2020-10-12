from validation import validation
import numpy as np 
from metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors


class LabelPropagations(validation):
    
    def __init__(self, kernel='rbf', *, gamma=0.1, n_neighbors=7,
             alpha=1, max_iter=30, tol=1e-3, n_jobs=None):
        """初始化LabelPropagations"""
        self.max_iter = max_iter
        self.tol = tol

        # kernel parameters
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors

        # clamping factor
        self.alpha = alpha

        self.n_jobs = n_jobs
        
        
#         self.classes_ = None            #标签种类,不重复
#         self.label_distributions_ = None    #标签概率矩阵Pij
#         self.transduction_ = None         #等同于labels
#         self.n_iter_ = None

        
    def __repr__(self):
        return "LabelPropagations()"
    
    def predict(self,X):
        """利用已经有的划分完的社区和相似矩阵，预测点属于哪个社区"""
        graph_matrix = rbf_kernel(X,self.X_,self.gamma)
        pro_matrix = np.dot(graph_matrix,self.label_distributions_)
        return np.argmax(pro_matrix,axis=1)
    
    def fit(self, X, y):
        """根据已知数据用LabelPropagations训练数据"""
        
        X,y = self._validate_data(X,y)
        self.X_ = X
        
        # construct actual graph
        graph_matrix = self._build_graph()
        
        # label constuction
        # construct a categorical distribution for classification only
        classes = np.unique(y)
        classes = np.sort(classes[classes!=-1])
        self.classes_ = classes
        
        n_samples,n_classes = len(X),len(self.classes_)
        
        # initialize distributions
        self.label_distributions_ = np.zeros((n_samples,n_classes))
        unlabeled = y == -1
        labeled = y != -1
        for label in classes:
            self.label_distributions_[y == label, classes == label] = 1
#         self.label_distributions_[unlabeled,:] = -1
        y_static = np.copy(self.label_distributions_)
        l_previous = np.zeros((n_samples,n_classes))
        
        
        for i in range(self.max_iter):
            if np.abs(self.label_distributions_ - l_previous).sum() < self.tol:
                break
            print(graph_matrix)
            l_previous = self.label_distributions_
            self.label_distributions_ = np.dot(graph_matrix,self.label_distributions_)
            print(self.label_distributions_)
            print()
            self.label_distributions_ = np.where(unlabeled[:,np.newaxis],self.label_distributions_, y_static)
        
        return self
        
    
    def _get_kernel(self, X, y=None):
        if self.kernel == "rbf":
            return rbf_kernel(X, X, gamma=self.gamma)
        elif self.kernel == "knn":
            if self.nn_fit is None:
                self.nn_fit = NearestNeighbors(n_neighbors=self.n_neighbors,
                                               n_jobs=self.n_jobs).fit(X)
            if y is None:
                return self.nn_fit.kneighbors_graph(self.nn_fit._fit_X,
                                                    self.n_neighbors,
                                                    mode='connectivity')
            
    
        
    def _build_graph(self):
        if self.kernel == 'knn':k
            self.nn_fit = None
        affinity_matrix = self._get_kernel(self.X_)
        normalizer = affinity_matrix.sum(axis=0)
        affinity_matrix /= normalizer[:, np.newaxis]
        return affinity_matrix
