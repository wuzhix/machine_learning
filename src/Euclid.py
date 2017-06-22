'''
欧几里得距离：在数学中，欧几里得距离或欧几里得度量是欧几里得空间中两点间“普通”（即直线）距离。
'''
import math


def ml_euclid(x, y):
    sum_of_squares = sum([pow(x1-y1, 2) for (x1, y1) in zip(x, y)])
    ret = math.sqrt(sum_of_squares)
    return ret


def main():
    x = [1, 3, 5, 7, 9]
    y = [2, 4, 6, 8, 10]
    print(ml_euclid(x, y))

if __name__ == '__main__':
    main()
