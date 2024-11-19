import numpy as np

# 样本数据
x_class1 = np.array([[0, 2, 1], [0, 1, 0]])  # ω1
x_class2 = np.array([[-1, -2, -2], [1, 0, -1]])  # ω2
x_class3 = np.array([[0, 0, 1], [-2, -1, -2]])  # ω3

# 样本均值
mean_class1 = np.mean(x_class1, axis=1)
mean_class2 = np.mean(x_class2, axis=1)
mean_class3 = np.mean(x_class3, axis=1)

# 样本协方差矩阵
cov_class1 = np.cov(x_class1)
cov_class2 = np.cov(x_class2)
cov_class3 = np.cov(x_class3)

# 正则化协方差矩阵 如果协方差矩阵较为稳定，则不会再报 Singular matrix 错误。正则化或伪逆会轻微影响分类结果，但总体精度不受影响。
def regularize_covariance(cov, lambda_val=1e-6):
    return cov + lambda_val * np.eye(cov.shape[0])

cov_class1 = regularize_covariance(cov_class1)
cov_class2 = regularize_covariance(cov_class2)
cov_class3 = regularize_covariance(cov_class3)

# 判别函数
def discriminant_unequal(x, mean, cov):
    cov_inv = np.linalg.inv(cov)
    term1 = -0.5 * np.log(np.linalg.det(cov))
    term2 = -0.5 * np.dot(np.dot((x - mean).T, cov_inv), (x - mean))
    return term1 + term2

# 测试点
test_point = np.array([-2, 2])

# 判别值
g1 = discriminant_unequal(test_point, mean_class1, cov_class1)
g2 = discriminant_unequal(test_point, mean_class2, cov_class2)
g3 = discriminant_unequal(test_point, mean_class3, cov_class3)

# 分类结果
predicted_class = np.argmax([g1, g2, g3]) + 1
print("The predicted class is:", predicted_class)





# 样本数量
n1, n2, n3 = x_class1.shape[1], x_class2.shape[1], x_class3.shape[1]
# 计算共享协方差矩阵
shared_cov = (n1 * cov_class1 + n2 * cov_class2 + n3 * cov_class3) / (n1 + n2 + n3)

# 判别函数 (协方差相等)
def discriminant_equal(x, mean, shared_cov):
    cov_inv = np.linalg.inv(shared_cov)
    term1 = np.dot(np.dot(mean.T, cov_inv), x)
    term2 = -0.5 * np.dot(np.dot(mean.T, cov_inv), mean)
    return term1 + term2

# 计算点 (-2, 2) 的判别值
g1 = discriminant_equal(test_point, mean_class1, shared_cov)
g2 = discriminant_equal(test_point, mean_class2, shared_cov)
g3 = discriminant_equal(test_point, mean_class3, shared_cov)

# 分类结果
predicted_class = np.argmax([g1, g2, g3]) + 1
print("The predicted class is:", predicted_class)

import matplotlib.pyplot as plt

def plot_decision_boundary(mean1, mean2, mean3, cov1, cov2, cov3, shared_cov=None, equal_cov=False):
    x1_vals = np.linspace(-3, 3, 200)
    x2_vals = np.linspace(-3, 3, 200)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = np.zeros((X1.shape[0], X1.shape[1], 3))

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = np.array([X1[i, j], X2[i, j]])
            if equal_cov:
                Z[i, j, 0] = discriminant_equal(x, mean1, shared_cov)
                Z[i, j, 1] = discriminant_equal(x, mean2, shared_cov)
                Z[i, j, 2] = discriminant_equal(x, mean3, shared_cov)
            else:
                Z[i, j, 0] = discriminant_unequal(x, mean1, cov1)
                Z[i, j, 1] = discriminant_unequal(x, mean2, cov2)
                Z[i, j, 2] = discriminant_unequal(x, mean3, cov3)

    decision = np.argmax(Z, axis=2)
    plt.contourf(X1, X2, decision, alpha=0.5, cmap='coolwarm')
    plt.scatter(x_class1[0, :], x_class1[1, :], label='ω1', color='blue')
    plt.scatter(x_class2[0, :], x_class2[1, :], label='ω2', color='red')
    plt.scatter(x_class3[0, :], x_class3[1, :], label='ω3', color='green')
    plt.scatter(test_point[0], test_point[1], color='black', label='Test Point', s=100)
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary")
    plt.show()

# 绘制决策边界
plot_decision_boundary(mean_class1, mean_class2, mean_class3, cov_class1, cov_class2, cov_class3, shared_cov=None, equal_cov=False)
plot_decision_boundary(mean_class1, mean_class2, mean_class3, cov_class1, cov_class2, cov_class3, shared_cov=shared_cov, equal_cov=True)