'''
皮尔逊值：在统计学中，皮尔逊积矩相关系数用于度量两个变量X和Y之间的相关（线性相关），其值介于-1与1之间。
在自然科学领域中，该系数广泛用于度量两个变量之间的相关程度
'''
import math


def ml_ppmcc(x, y):
    n = len(x)
    sum1 = sum([x1 for x1 in x])
    sum2 = sum([y1 for y1 in y])

    sum1Sq = sum([pow(x1, 2) for x1 in x])
    sum2Sq = sum([pow(y1, 2) for y1 in y])

    pSum = sum(x1*y1 for (x1, y1) in zip(x, y))

    num = pSum-(sum1*sum2)/n
    den = math.sqrt((sum1Sq-pow(sum1, 2)/n)*(sum2Sq-pow(sum2, 2)/n))
    ret = num/den
    return ret


def main():
    x = [1, 3, 5, 7, 9]
    y = [0.11, 0.13, 0.15, 0.17, 0.19]
    print(ml_ppmcc(x, y))

if __name__ == '__main__':
    main()
