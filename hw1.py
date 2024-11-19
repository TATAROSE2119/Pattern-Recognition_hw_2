import numpy as np
from matplotlib import pyplot as plt

# 样本数据
# ω1类
x1_class1 = np.array([[1, 1], [1, 0], [2, -1]]).T  # (2行, 3列)
# ω2类
x1_class2 = np.array([[-1, 1], [-1, 0], [-2, -1]]).T  # (2行, 3列)

# 均值计算
mean_class1 = np.mean(x1_class1, axis=1)  # ω1均值
mean_class2 = np.mean(x1_class2, axis=1)  # ω2均值

# 协方差矩阵计算
cov_class1 = np.cov(x1_class1)
cov_class2 = np.cov(x1_class2)

# 样本数量
n1 = x1_class1.shape[1]
n2 = x1_class2.shape[1]

# 共享协方差矩阵
shared_cov = (n1 * cov_class1 + n2 * cov_class2) / (n1 + n2)

# 判别函数
def discriminant(x, mean1, mean2, shared_cov):
    cov_inv = np.linalg.inv(shared_cov)
    w = np.dot(cov_inv, (mean1 - mean2))
    w0 = -0.5 * np.dot(np.dot((mean1 + mean2).T, cov_inv), (mean1 - mean2))
    return np.dot(w.T, x) + w0

# 绘图
def plot_decision_boundary_with_test_point(mean1, mean2, shared_cov, test_point):
    x1_vals = np.linspace(-3, 3, 200)
    x2_vals = np.linspace(-3, 3, 200)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = np.zeros_like(X1)

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = np.array([X1[i, j], X2[i, j]])
            Z[i, j] = discriminant(x, mean1, mean2, shared_cov)

    plt.contourf(X1, X2, Z, levels=0, cmap='coolwarm', alpha=0.5)
    plt.scatter(x1_class1[0, :], x1_class1[1, :], color='red', label='ω1')
    plt.scatter(x1_class2[0, :], x1_class2[1, :], color='blue', label='ω2')
    plt.scatter(test_point[0], test_point[1], color='green', label='Test Point (2,0)', edgecolors='black', s=100)
    plt.title("Decision Boundary with Test Point")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()

# 点 (2, 0)
test_point = np.array([2, 0])

# 计算判别函数值
g_value = discriminant(test_point, mean_class1, mean_class2, shared_cov)

# 判断类别
predicted_class = "ω1" if g_value > 0 else "ω2"
print(f"The point ({test_point[0]}, {test_point[1]}) is classified as {predicted_class}.")

plot_decision_boundary_with_test_point(mean_class1, mean_class2, shared_cov, test_point)
plot_decision_boundary_with_test_point(mean_class1, mean_class2, cov_class1, test_point)
plot_decision_boundary_with_test_point(mean_class1, mean_class2, cov_class2, test_point)
