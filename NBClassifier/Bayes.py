import numpy as np


class BayesClassifier:
    '''
    这个类实现一个最最简单的朴素贝叶斯分类器。
    适用于输入的各个属性都是离散值、输出为二分类（0/1）的，最最简单的情况。
    '''

    def __init__(self, trainX, trainY):
        '''
        使用训练集初始化分类器。
        @param trainX: 训练集的属性部分，二维列表每行为一组属性
        @param trainY: 训练集的标签部分，一维列表每个数据（0/1）与 trainX 的每组属性一一对应
        '''

        # 取属性的维数
        self.__dimensions = len(trainX[0])

        # 分离正样本和负样本
        sample = list(zip(trainY, trainX))
        positiveSample = list(filter(lambda x: x[0] == 1, sample))
        negativeSample = list(filter(lambda x: x[0] == 0, sample))

        # 计算正负样本的先验概率
        self.__positiveP = len(positiveSample) / len(sample)
        self.__negativeP = len(negativeSample) / len(sample)

        # 计算正样本似然概率
        pCounts = [{} for i in range(self.__dimensions)]
        for el in positiveSample:
            for i in range(self.__dimensions):
                pCounts[i][el[1][i]] = pCounts[i].get(el[1][i], 0) + 1
        self.__positiveLikelihood = [{key: el[key] / sum(el.values()) for key in el} for el in pCounts]

        # 计算负样本似然概率
        nCounts = [{} for i in range(self.__dimensions)]
        for el in negativeSample:
            for i in range(self.__dimensions):
                nCounts[i][el[1][i]] = nCounts[i].get(el[1][i], 0) + 1
        self.__negativeLikelihood = [{key: el[key] / sum(el.values()) for key in el} for el in nCounts]


    def predict(self, testX):
        '''
        对测试集进行二分类操作。
        @param testX: 测试集的属性部分，二维列表每行为一组属性
        return value: 测试集的分类输出，一维列表每个数据（0/1）为 testX 每组属性的分类结果
        '''

        ans = []

        # 对测试集的每组数据，分别计算并比较其分类为正、负样本的后验概率大小
        for el in testX:
            if len(el) != self.__dimensions:
                raise ValueError('incorrect dimension')
            else:
                positivePredictP = self.__positiveP
                negativePredictP = self.__negativeP
                for i in range(self.__dimensions):
                    positivePredictP *= self.__positiveLikelihood[i].get(el[i], 0.0)
                    negativePredictP *= self.__negativeLikelihood[i].get(el[i], 0.0)
                ans.append(1 if positivePredictP > negativePredictP else 0)
        return ans


if __name__ == '__main__':
    train = np.loadtxt('./1.csv', dtype=np.uint32, delimiter=',', encoding='utf-8')
    trainX, trainY = train[:, :-1].tolist(), train[:, -1].tolist()
    classifier = BayesClassifier(trainX, trainY)
    testX = np.loadtxt('./2.csv', dtype=np.uint32, delimiter=',', encoding='utf-8').tolist()
    ans = classifier.predict(testX)
    print(ans)
