'''
矩阵分解推荐算法
假设有m*n的矩阵，m行表示用户，n列表示商品，第i行j列的值表示用户i对商品j的评分。
评分采用5分制，5分表示最好，1分表示最差，0分表示用户未购买商品。
目的：现在要给用户i推荐h个未购买的商品，在该用户所有未购买的商品中，购买这k个商品的打分可能最高
矩阵分解原理：Rmn = Pmk × Qkn。其中k理解为用户的k维特征和商品的k维特征
源码参考：http://blog.csdn.net/google19890102/article/details/51124556#comments
'''

import numpy as np
import random


# 生成Rmn矩阵模型
def makeMatrixMode():
    data = np.random.randint(0, 6, size=[20, 20])
    np.savetxt("matrix.txt", data, fmt='%d')
    return data


# 批量梯度下降算法
def BatchGradientDescent(data, k):
    # 将数组转换为矩阵
    dataMatrix = np.mat(data)
    # 获取矩阵的行和列
    m, n = np.shape(dataMatrix)
    # 随机生成预测矩阵Pmk
    p = np.mat(np.random.random((m, k)))
    # 随机生成预测矩阵Qkn
    q = np.mat(np.random.random((k, n)))

    # 梯度步长
    alpha = 0.001
    # 正则参数
    beta = 0.02
    # 最大循环次数
    maxCycles = 1000

    for step in range(maxCycles):
        for i in range(m):
            for j in range(n):
                # 只需要拟合购买的商品，未购买的商品通过最后的P × Q生成
                if dataMatrix[i, j] > 0:
                    # error用来计算残差
                    error = dataMatrix[i, j]
                    for a in range(k):
                        error = error - p[i, a] * q[a, j]
                    for a in range(k):
                        '''
                        2 * error * q[a, j]为损失函数的偏导数（损失函数是复合导数）
                        beta * p[i, a]是正则化项
                        p更新一个步长的梯度后，会更接近真实值
                        
                        q同理
                        '''
                        p[i, a] = p[i, a] + alpha * (2 * error * q[a, j] - beta * p[i, a])
                        q[a, j] = q[a, j] + alpha * (2 * error * p[i, a] - beta * q[a, j])
        if step % 10 == 0:
            print(step)
    return p, q


# 随机梯度下降算法
def randGradientDescent(data, k):
    # 将数组转换为矩阵
    dataMatrix = np.mat(data)
    # 获取矩阵的行和列
    m, n = np.shape(dataMatrix)
    # 随机生成预测矩阵Pmk
    p = np.mat(np.random.random((m, k)))
    # 随机生成预测矩阵Qkn
    q = np.mat(np.random.random((k, n)))

    # 梯度步长
    alpha = 0.001
    # 正则参数
    beta = 0.02
    # 最大循环次数
    maxCycles = 1000

    for step in range(maxCycles):
        i = np.random.randint(0, m)
        j = np.random.randint(0, n)
        # 只需要拟合购买的商品，未购买的商品通过最后的P × Q生成
        if dataMatrix[i, j] > 0:
            # error用来计算残差
            error = dataMatrix[i, j]
            for a in range(k):
                error = error - p[i, a] * q[a, j]
            for a in range(k):
                '''
                2 * error * q[a, j]为损失函数的偏导数（损失函数是复合导数）
                beta * p[i, a]是正则化项
                p更新一个步长的梯度后，会更接近真实值

                q同理
                '''
                p[i, a] = p[i, a] + alpha * (2 * error * q[a, j] - beta * p[i, a])
                q[a, j] = q[a, j] + alpha * (2 * error * p[i, a] - beta * q[a, j])
        if step % 10 == 0:
            print(step)
    return p, q


# 小批量随机梯度下降算法
def LittleBatchGradientDescent(data, k):
    # 将数组转换为矩阵
    dataMatrix = np.mat(data)
    # 获取矩阵的行和列
    m, n = np.shape(dataMatrix)
    # 随机生成预测矩阵Pmk
    p = np.mat(np.random.random((m, k)))
    # 随机生成预测矩阵Qkn
    q = np.mat(np.random.random((k, n)))

    # 梯度步长
    alpha = 0.001
    # 正则参数
    beta = 0.02
    # 最大循环次数
    maxCycles = 1000

    for step in range(maxCycles):
        # 随机取不重复值
        m1 = random.sample(range(m), 20)
        n1 = random.sample(range(n), 20)
        for i in m1:
            for j in n1:
                # 只需要拟合购买的商品，未购买的商品通过最后的P × Q生成
                if dataMatrix[i, j] > 0:
                    # error用来计算残差
                    error = dataMatrix[i, j]
                    for a in range(k):
                        error = error - p[i, a] * q[a, j]
                    for a in range(k):
                        '''
                        2 * error * q[a, j]为损失函数的偏导数（损失函数是复合导数）
                        beta * p[i, a]是正则化项
                        p更新一个步长的梯度后，会更接近真实值

                        q同理
                        '''
                        p[i, a] = p[i, a] + alpha * (2 * error * q[a, j] - beta * p[i, a])
                        q[a, j] = q[a, j] + alpha * (2 * error * p[i, a] - beta * q[a, j])
        if step % 10 == 0:
            print(step)
    return p, q


if __name__ == '__main__':
    data = makeMatrixMode()

    # 设置特征维度为5

    # print('randGradientDescent start')
    # p, q = randGradientDescent(data, 5)
    # result = p * q
    # print('randGradientDescent complete')
    # np.savetxt("rand.txt", np.round(result), fmt='%d')

    # print('LittleBatchGradientDescent start')
    # p, q = LittleBatchGradientDescent(data, 5)
    # result = p * q
    # print('LittleBatchGradientDescent complete')
    # np.savetxt("little.txt", np.round(result), fmt='%d')

    print('BatchGradientDescent start')
    p, q = BatchGradientDescent(data, 5)
    result = p * q
    print('BatchGradientDescent start')
    np.savetxt("batch.txt", np.round(result), fmt='%d')
