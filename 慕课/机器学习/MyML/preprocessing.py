# 均值方差归一化
class StandardScaler():
    def __init__(self):
        self.mean_ = None
        self.scale_= None #标准差
        self.var_  = None #方差
    
    
    def __repr__(self):
        print("StandardScaler()")
        
    
    def fit(self, X_train):
        """根据训练数据集X_train获得数据的均值和方差"""
        self.mean_ = np.mean(X_train,axis=0)       #每列均值
        self.scale_= np.std(X_train,axis=0,ddof=0) #总体方差，分母n而不是n-1
        
        
    def transform(self, X_test):
        """按照训练得到的均值、标准差将X_test归一化"""
        res = np.empty(X_test.shape,dtype=X_test.dtype)
        for i in range(X_test.shape[1]):
            res[:,i] = (X_test[:,i]-self.mean_[i])/self.scale_[i]
        return res
    
    
    
# 最值归一化
class MinMaxScaler():
    def __init__(self):
        self.min_ = None
        self.max_ = None 
    
    
    def __repr__(self):
        print("MinMaxScaler()")
        
    
    def fit(self, X_train):
        """按照最值方差归一化"""
        self.min_ = np.min(X_train,axis=0)       #每列均值
        self.max_ = np.max(X_train,axis=0) #总体方差，分母n而不是n-1
        
        
    def transform(self, X_test):
        """按照训练得到的最值将X_test归一化"""
        res = np.empty(X_test.shape,dtype=X_test.dtype)
        for i in range(X_test.shape[1]):
            res[:,i] = (X_test[:,i]-self.min_[i])/(self.max_[i]-self.min_[i])
        return res