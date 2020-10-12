import numpy as np
import warnings

class validation():
    def _validate_data(self,X,y):
        """检查矩阵是否缺失，X的长度是否与y相等"""
        if np.isnan(X).sum() or X is None:
            raise ValueError("X is invalid: it must not be nan or none")
        if np.isnan(y).sum() or y is None:
            raise ValueError("y is invalid: it must not be nan or none")
        if len(X) != len(y):
            raise ValueError("the length of X must be equal to y")
            
        return X,y