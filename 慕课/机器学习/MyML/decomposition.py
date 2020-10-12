import numpy as np




# PCA 与sklearn中的PCA计算结果不一致，还差反映射和n_components<1的函数
class PCA():
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ratio_ = None
        
    def __repr__(self):
        print("PCA()")
    
    def fit(self,X_train,learning_rate=0.01,n_iters=1e4):
        """获得数据集X的前n个主成分"""
        assert self.n_components <= X.shape[1], \
            "n_components must not be greater than the feature number of X"
        n = X_train.shape[1]
        if self.n_components > 1:
            components = np.ones((self.n_components,X_train.shape[1]))
        else:
            pass
        self.components_ = np.zeros_like(components)
        X_train=X_train-np.mean(X_train,axis=0) #中心化:demean
       
        def J(w,X):
            return np.sum((X.dot(w))**2)/len(X)
        
        def dJ(w,X):
            return X.T.dot(X.dot(w))*2/len(X)
        
        def dJ_debug(w,X,epsilon=1e-4):
            res=np.zeros_like(w)
            for i in range(len(w)):
                w_1=w.copy()
                w_1[i]+=epsilon
                w_2=w.copy()
                w_2[i]-=epsilon
                res[i]=( J(w_1,X) - J(w_2,X) ) / ( 2 * epsilon )
            return res
        
        def direction(w):
            """转换为方向向量"""
            return w / np.linalg.norm(w)
        
        def gredient_ascent(dJ,w,X):
            i_iter=0
            while(i_iter < n_iters):
                last_J=J(w,X)
                w=w+learning_rate*dJ(w,X)
                w=direction(w)
                if(J(w,X)-last_J<1e-4):
                    break
                i_iter+=1
            return w
        
        for i in range(components.shape[0]):
            w=direction(components[i].reshape(-1,1))
            w=gredient_ascent(dJ,w,X_train) #w一直是列向量
            
            X_train=X_train-X_train.dot(w)*(w.reshape(1,-1)) #更新X,n个特征对应的模长均乘以方向向量
            
            self.components_[i,:]=w.reshape(1,-1) #用行存
            
        return self
    
    def transform(self,X):
        """将给定的X映射到各主成分"""
        assert self.components_ is not None,\
            "you must be fit before transform"
        return X.dot(self.components_.T)
    
#     def inverse_trainsform(self,X_pca):
#         """将给定的X_pca映射回原坐标系"""
#         assert self.components_ is not None,\
#             "you must be fit before transform"
#         return X_pca.dot
            