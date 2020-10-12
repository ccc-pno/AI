from collections import Counter #集合

class KNeighborsClassifier():
    def __init__(self, n_neighbors=3):
        assert n_neighbors>0, \
            "n_neighbors must be larger than zero"
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
    
    
    def __repr__(self):
        print("KNeighborsClassifier()")
    
    
    def fit(self, X_train, y_train):
        """根据训练集X_train，y_train训练的过程"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of X_test"
        self.X_train = X_train
        self.y_train = y_train
        
    
    def predict(self, X_test):
        """根据预测数据X_test预测的过程"""
        assert self.X_train is not None and self.y_train is not None,\
            "you must train before predict!"
        res=[]
        
        def distance(a,b,p=2):
            return np.sum((a-b)**p)
        
        def Knn_classfy(x_test, X_train=self.X_train, k=self.n_neighbors):
            distances=[]
            for i in range(self.X_train.shape[0]-1):
                d = distance(a=x_test,b=self.X_train[i])
                distances.append(d)
            nearest = np.argsort(distances)
            top_k = self.y_train[nearest[:k],0]
            votes = Counter(top_k)
            return votes.most_common(1)[0][0] #最多前 1 个元素
        
        for point in X_test:
            res.append(Knn_classfy(point))
        
        return np.array(res)