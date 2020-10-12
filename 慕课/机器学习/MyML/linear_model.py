from sklearn.metrics import r2_score
import numpy as np




# 线性回归
class LinearRegression():
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None           #theta 0-n

    def __repr__(self):
        return "LinearRegression()"
    
    def fit_normal(self,X_train,y_train):
        """根据训练数据集X_train, y_train和常规解法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        
        X_b = np.hstack((np.ones(X_train.shape[0]).reshape(-1,1), X_train))
        self._theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
    
    
    def fit_gd(self,X_train, y_train, learning_rate=0.01, 
                     epsilon=1e-6, n_iters = 1e4):
        """根据训练数据集X_train, y_train和梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        
        X_b = np.hstack((np.ones(X_train.shape[0]).reshape(-1,1), X_train))
        self._theta = np.zeros(X_b.shape[1]).reshape(-1,1)
        y_train = y_train.reshape(-1,1)
        i_iters = 0

        def J(theta=self._theta, X=X_b, y=y_train, m=X_train.shape[0]):
            return np.sum(1/(2*m) * (X.dot(theta) - y)**2)
        
#         def dJ(theta=self._theta, X=X_b, y=y_train, m=X_train.shape[0]):
#             dj = np.empty(len(self._theta))
#             dj[0] = np.sum(X.dot(theta)-y)
#             for i in range(1,len(theta)):
#                 dj[i] = (X.dot(theta)-y).T.dot(X[:,i])
# #             print("dj:{}".format(dj))
#             return np.array(dj).reshape(-1,1)/m
        def dJ(theta=self._theta, X=X_b, y=y_train, m=X_train.shape[0]):
            return 1/m*(X.T.dot(X.dot(theta)-y))
    
        while(i_iters<n_iters):
            last_J = J(theta=self._theta)
            self._theta = self._theta - learning_rate*dJ(theta=self._theta)
            
            i_iters+=1
            if(abs(J(theta=self._theta)-last_J)<1e-8):
                break

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
            
    def predict(self,X_test):
        """给定待预测数据集X_test, 返回X_test的结果向量"""
        assert self.coef_ is not None and self.intercept_ is not None \
            and self._theta is not None,\
            "You must be train before predict!"
        assert X_test.shape[1] == self.coef_.shape[0], \
            "The feature number of X_predict must be equal to X_test"
        
        X_b = np.hstack((np.ones(X_test.shape[0]).reshape(-1,1), X_test))
        return X_b.dot(self._theta)
    
    def score(self, X_test, y_test):
        """根据测试数据集X_test 和 y_test确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)
    
    
# 岭回归
class Ridge():
    def __init__(self,alpha=1):
        self._theta = None
        self.coef_ = None
        self.intercept_ = None
        self.alpha_ = alpha
    
    def __repr__(self):
        print("Ridge()")
        
    def fit(self, X_train, y_train, learning_rate=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, alpha训练Ridge模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to y_train."
        X_b = np.hstack((np.ones(X_train.shape[0]).reshape(-1,1),X_train))
        theta = np.zeros(X_b.shape[1]).reshape(-1,1)
        alpha = self.alpha_
        i_iter = 0
        
        def J(X=X_b, y=y_train, theta=theta, alpha=alpha, m=X_b.shape[0]):
            return 1/m*np.sum((X.dot(theta)-y)**2) + alpha/2 * np.sum(theta**2)
        
        def dJ(X=X_b, y=y_train, theta=theta, alpha=alpha, m=X_b.shape[0]):
            return 2/m*(X.T.dot(X.dot(theta)-y)) + alpha*theta
        
        while(i_iter < n_iters):
            last_J = J(theta=theta)
            theta = theta - learning_rate*dJ(theta=theta)
            if(abs(J(theta=theta)-last_J)<1e-6):
                break
            i_iter+=1
        
        self._theta = theta
        self.coef_  = theta[1:]
        self.intercept_ = theta[0]
        
    def predict(self, X_test):
        """给定待预测数据集X_test, 返回X_test的结果向量"""
        assert self.coef_ is not None and self.intercept_ is not None \
            and self._theta is not None,\
            "You must be train before predict!"
        assert X_test.shape[1] == self.coef_.shape[0], \
            "The feature number of X_predict must be equal to X_test"
        X_b = np.hstack((np.ones(X_test.shape[0]).reshape(-1,1),X_test))
        return X_b.dot(self._theta)
    
    def score(self, X_test, y_test):
        """根据测试数据集X_test 和 y_test确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)
    


# Lasso回归，|theta|当theta=0时怎么求导？？？
class Lasso():
    def __init__(self,alpha=1):
        self._theta = None
        self.coef_ = None
        self.intercept_ = None
        self.alpha_ = alpha
    
    def __repr__(self):
        print("Lasso()")
        
    def fit(self, X_train, y_train, learning_rate=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, alpha训练Ridge模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to y_train."
        X_b = np.hstack((np.ones(X_train.shape[0]).reshape(-1,1),X_train))
        theta = np.ones(X_b.shape[1]).reshape(-1,1)
        alpha = self.alpha_
        i_iter = 0
        
        def J(X=X_b, y=y_train, theta=theta, alpha=alpha, m=X_b.shape[0]):
            return 1/m*np.sum((X.dot(theta)-y)**2) + alpha * np.sum(abs(theta))
        
        def dJ(X=X_b, y=y_train, theta=theta, alpha=alpha, m=X_b.shape[0]):   
            return 2/m*(X.T.dot(X.dot(theta)-y)) + alpha*(theta/abs(theta))
        
        while(i_iter < n_iters):
            last_J = J(theta=theta)
            theta = theta - learning_rate*dJ(theta=theta)
            if(abs(J(theta=theta)-last_J)<1e-6):
                break
            i_iter+=1
        
        self._theta = theta
        self.coef_  = theta[1:]
        self.intercept_ = theta[0]
        
    def predict(self, X_test):
        """给定待预测数据集X_test, 返回X_test的结果向量"""
        assert self.coef_ is not None and self.intercept_ is not None \
            and self._theta is not None,\
            "You must be train before predict!"
        assert X_test.shape[1] == self.coef_.shape[0], \
            "The feature number of X_predict must be equal to X_test"
        X_b = np.hstack((np.ones(X_test.shape[0]).reshape(-1,1),X_test))
        return X_b.dot(self._theta)
    
    def score(self, X_test, y_test):
        """根据测试数据集X_test 和 y_test确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)
    
    
    
# 逻辑回归
class LogisticRegression():
    def __init__(self):
        self.coef_=None
        self.intercept_=None
        self._theta=None
        self.multi_class=None
        self.C=1.
        self.penalty='l2'
        self.max_iter=1e4
    
    def __repr__(self):
        print("LogisticRegression()")
    
    def fit(self,X_train,y_train,learning_rate=0.01):
        """根据训练集X_train,y_train训练逻辑回归模型"""
        assert X_train.shape[0]==y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
            
        def sigmoid(t):
            return 1/(1+np.exp(-t))
        
        def J(theta,X):
            return np.sum(y*sigmoid(X.dot(theta))+(1-y)*sigmoid(1-X.dot(theta)))/(-len(X))
        
        def dJ(theta,X):
            return X.T.dot(sigmoid(X.dot(theta))-y)/len(X)
        
        def dJ_debug(w,X,epsilon=1e-4):
            res=np.zeros_like(w)
            for i in range(len(w)):
                w_1=w.copy()
                w_1[i]+=epsilon
                w_2=w.copy()
                w_2[i]-=epsilon
                res[i]=( J(w_1,X) - J(w_2,X) ) / ( 2 * epsilon )
            return res
        
        def gredient(dJ_func,w,X):
            i_iter=0
            while(i_iter < self.max_iter):
                last_J=J(w,X)
                w=w-learning_rate*dJ_func(w,X)
                if(J(w,X)-last_J<1e-4):
                    break
                i_iter+=1
            return w
        
        X_b=np.hstack((np.ones(X_train.shape[0]).reshape(-1,1),X_train))
        theta=np.zeros((X_b.shape[1],1))
        gredient=gredient(dJ,theta,X_b)
        self._theta=gredient
        self.coef_=self._theta[1:]
        self.intercept_=self._theta[0]
            
        return self
    
    def predict(self,X_test):
        """根据测试集X_test预测y_predict"""
        assert self._theta is not None,\
            "you must fit before predict"
        def sigmoid(t):
            return 1/(1+np.exp(-t))
        X_b=np.hstack((np.ones(X_test.shape[0]).reshape(-1,1),X_test))
        y_predict=np.array(sigmoid(X_b.dot(self._theta))>0.5,dtype=int)
        return y_predict
    
    def score(self,X_test,y_test):
        """根据测试集X_test,y_test计算准确度"""
        assert X_test.shape[0]==y_test.shape[0], \
            "the size of X_test must be equal to the size of y_test"
        def sigmoid(t):
            return 1/(1+np.exp(-t))
        X_b=np.hstack((np.ones(X_test.shape[0]).reshape(-1,1),X_test))
        y_predict=np.array(sigmoid(X_b.dot(self._theta))>0.5,dtype=int)
        return np.sum(y_predict==y_test)/len(y_test)