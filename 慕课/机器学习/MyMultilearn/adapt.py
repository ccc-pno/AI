import numpy as np



class MLkNN():
    def __init__(self, k=10, s=1):
        self.k = k
        self.s = s

    def __repr__(self):
        print("MLKNN()")
        
    def knn(self,X,x_predict):
        distances = np.zeros(self.train_data_num)
        for i in range(self.train_data_num):
            distances[i] = np.sum((X[i]-x_predict)**2)
        nearest = np.argsort(distances)

        return nearest[:self.k]

    def fit(self, X, y):
        """用MLKNN，根据训练集X进行训练"""
        self.X_ = X
        self.label_num = y.shape[1]
        self.train_data_num = X.shape[0]
        self.ph1 = np.zeros(self.label_num)
        self.ph0 = np.zeros(self.label_num)
        self.peh1 = np.zeros((self.label_num,self.k+1))
        self.peh0 = np.zeros((self.label_num,self.k+1))
        
        for l in range(self.label_num):
            self.ph1[l] = (self.s + np.sum(y[:,l])) / (2*self.s + self.train_data_num)
            self.ph0[l] = 1 - self.ph1[l]
        
            c1 = np.zeros(self.k+1)
            c0 = np.zeros(self.k+1)
            for i in range(self.train_data_num):
                neighbor_l_sum = int(np.sum(y[self.knn(X,X[i]),l]))
                if y[i,l] == 1:
                    c1[neighbor_l_sum] += 1
                else:
                    c0[neighbor_l_sum] += 1
                    
            for j in range(self.k+1):       
                self.peh1[l][j] = (self.s + c1[j]) / (self.s * (self.k+1) + np.sum(c1))
                self.peh0[l][j] = (self.s + c0[j]) / (self.s * (self.k+1) + np.sum(c0))
            
    def predict(self,X):
        test_data_num = X.shape[0]
        res = np.zeros((self.label_num,self.k+1))
        predict = np.zeros((test_data_num,self.label_num))
        
#         for l in range(self.label):
#             for j in range(self.k+1):
#                 res[l][j] = np.argmax(self.ph0[l]*self.peh0[l][j],self.ph1[l]*self.peh1[l][j])
        for l in range(self.label_num):
            for i in range(test_data_num):
                neighbor_l_sum = int(np.sum(y[self.knn(self.X_,X[i]),l]))
                a = self.ph0[l]*self.peh0[l][neighbor_l_sum]
                b = self.ph1[l]*self.peh1[l][neighbor_l_sum]
                predict[i][l] = np.argmax(np.array([a,b]))
        
        return np.array(predict)
