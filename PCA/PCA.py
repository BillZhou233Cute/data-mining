import numpy as np


def PCA(filename, dimension):
    # 从 CSV 读数据，然后去掉“ID”字段得到 X
    data = np.loadtxt(filename, dtype=np.uint32, delimiter=',', encoding='utf-8')
    X = data[:, 1:]

    # 将 X 转置后，每行为一个字段，每列为一条记录
    # 将各个字段的值做零均值化处理，得到 A
    # 为了写起来方便起见这里先零均值化处理再转置了，效果一样
    A = (X - X.mean(axis=0)).T

    # 协方差矩阵
    C = np.corrcoef(A)

    # 特征值和特征向量
    P = np.linalg.eig(C)

    # 排序
    P2 = sorted(zip(P[0].tolist(), list(P[1])), reverse=True)
    TZZ = [el[0] for el in P2]
    TZXL = [el[1] for el in P2]

    # 计算累计贡献率
    LJGXL = [sum(TZZ[:i + 1]) / sum(TZZ) for i in range(len(TZZ))]

    # 需要前多少维作为新的主成分，就取排序后的前多少个特征向量
    Z = np.array(TZXL[:dimension])
    Y = np.matmul(Z, A)

    return Y, LJGXL


if __name__ == '__main__':
    Y, LJGXL = PCA('./1.csv', 2)
    print(Y)
