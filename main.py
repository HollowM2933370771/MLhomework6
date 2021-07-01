import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os

def generate(main, support, coeff):
    ret = main.copy()
    for i in main.columns[1:]:
        res = []
        lm, ls = [], []
        lm = main[i].tolist()
        ls = support[i].tolist()

        for j in range(len(main)):
            res.append((lm[j] * coeff) + (ls[j] * (1. - coeff)))
        ret[i] = res

    return ret


def improve(sub1, sub2, sub3, sub4, sub5, sub6, sub_ens, majority, m_majority):
    sub1v = sub1.values
    sub2v = sub2.values
    sub3v = sub3.values
    sub4v = sub4.values
    sub5v = sub5.values
    sub6v = sub6.values

    imp = sub_ens.copy()
    impv = imp.values
    NCLASS = 9
    number = 0

    for i in range(len(sub_ens)):
        c_count = 0
        row = impv[i, 1:]
        row_sort = np.sort(row)

        row1 = sub1v[i, 1:]
        row2 = sub2v[i, 1:]
        row3 = sub3v[i, 1:]
        row4 = sub4v[i, 1:]
        row5 = sub5v[i, 1:]
        row6 = sub6v[i, 1:]
        row1_sort = np.sort(row1)
        row2_sort = np.sort(row2)
        row3_sort = np.sort(row3)
        row4_sort = np.sort(row4)
        row5_sort = np.sort(row5)
        row6_sort = np.sort(row6)

        for j in range(NCLASS):
            count = 0
            for k in range(NCLASS):
                if (row6[j] == row6_sort[k]):
                    if (row1[j] == row1_sort[k]):
                        count = count + 1
                    if (row2[j] == row2_sort[k]):
                        count = count + 1
                    if (row3[j] == row3_sort[k]):
                        count = count + 1
                    if (row4[j] == row4_sort[k]):
                        count = count + 1
                    if (row5[j] == row5_sort[k]):
                        count = count + 1
            if (count >= majority):
                c_count = c_count + 1
        if ((c_count >= m_majority) and (row6_sort[8] >= row_sort[8])):
            impv[i, 1:] = row6
            number = number + 1

    imp.iloc[:, 1:] = impv[:, 1:]
    p_number = round(((number / 100000) * 100), 2)
    print('>>>  R  E  T  U  R  N  S  <<<')
    print(30 * '——')
    print(f'Number of changes: {number}\n')
    print(f'Percentage of changes: {p_number} %')
    print(30 * '——')
    return imp


if __name__ == '__main__':

    for dirname, _, filenames in os.walk('./input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    df1 = pd.read_csv('./input/train.csv')
    df2 = pd.read_csv('./input/test.csv')
    sam = pd.read_csv('./input/sample_submission.csv')

    sub1 = pd.read_csv('./input/histgradient-catboost-nn/submission1.csv')
    sub2 = pd.read_csv('./input/histgradient-catboost-nn/submission2.csv')
    sub3 = pd.read_csv('./input/histgradient-catboost-nn/submission3.csv')
    sub4 = pd.read_csv('./input/histgradient-catboost-nn/submission4.csv')
    sub5 = pd.read_csv('./input/histgradient-catboost-nn/submission5.csv')
    sub6 = pd.read_csv('./input/histgradient-catboost-nn/submission6.csv')

    sub = generate(sub2, sub1, 0.80)
    sub = generate(sub3, sub , 0.85)
    sub = generate(sub4, sub , 0.85)
    sub = generate(sub5, sub , 0.85)
    sub = generate(sub6, sub , 0.55)

    sub_ens = sub
    sub_ens.to_csv("result_tmp.csv", index=False)

    sub_imp = improve(sub1, sub2, sub3, sub4, sub5, sub6, sub_ens, 5, 7)
    sub_imp.to_csv("result_final.csv", index=False)
