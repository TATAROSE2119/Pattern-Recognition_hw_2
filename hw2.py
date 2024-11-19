import numpy as np
from matplotlib import pyplot as plt

#样本数据
x1_class=np.array([[0,0],[2,1],[1,0]])
x2_class=np.array([[-1,1],[-2,0],[-2,-1]])
x3_class=np.array([[0,-2],[0,-1],[1,-2]])

#计算均值
mean_class1=np.mean(x1_class,axis=1)
mean_class2=np.mean(x2_class,axis=1)
mean_class3=np.mean(x3_class,axis=1)
#计算协方差矩阵
cov_class1=np.cov(x1_class)
cov_class2=np.cov(x2_class)
cov_class3=np.cov(x3_class)
#样本数量
n1=x1_class.shape[1]
n2=x2_class.shape[1]
n3=x3_class.shape[1]
#共享协方差矩阵
shared_cov=(n1*cov_class1+n2*cov_class2+n3*cov_class3)/(n1+n2+n3)

# 判别函数 (协方差不等)
def discriminant_unequal(x, mean, cov):
    cov_inv = np.linalg.inv(cov)
    term1 = -0.5 * np.log(np.linalg.det(cov))
    term2 = -0.5 * np.dot(np.dot((x - mean).T, cov_inv), (x - mean))
    return term1 + term2

# 计算点 (-2, 2) 的判别值
test_point = np.array([-2, 2])
g1 = discriminant_unequal(test_point, mean_class1, cov_class1)
g2 = discriminant_unequal(test_point, mean_class2, cov_class2)
g3 = discriminant_unequal(test_point, mean_class3, cov_class3)

# 分类结果
predicted_class = np.argmax([g1, g2, g3]) + 1