import argparse
import logging

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


def load_data_MNTest(fl="./MCTestData.csv"):
    """
    Loads data stored in McNemarTest.csv
    :param fl: filename of csv file
    :return: labels, prediction1, prediction2
    """
    data = pd.read_csv(fl, header=None).to_numpy()
    labels = data[:, 0]
    prediction_1 = data[:, 1]
    prediction_2 = data[:, 2]
    return labels, prediction_1, prediction_2


def load_data_TMStTest(fl="./TMStTestData.csv"):
    """
    Loads data stored in fl
    :param fl: filename of csv file
    :return: y1, y2
    """
    data = np.loadtxt(fl, delimiter=",")
    y1 = data[:, 0]
    y2 = data[:, 1]
    return y1, y2


def load_data_FTest(fl="./FTestData.csv"):
    """
    Loads data stored in fl
    :param fl: filename of csv file
    :return: evaluations
    """
    errors = np.loadtxt(fl, delimiter=",")
    return errors


def McNemar_test(labels, prediction_1, prediction_2):
    """
    TODO
    :param labels: the ground truth labels
    :param prediction_1: the prediction results from model 1
    :param prediction_2:  the prediction results from model 2
    :return: the test statistic chi2_Mc
    """
    # chi2_Mc = np.random.uniform(0, 1)
    A, B, C, D = 0, 0, 0, 0
    for truth, model1, model2 in zip(labels, prediction_1, prediction_2):
        if model1 == truth and model2 == truth:
            A += 1
        elif model1 == truth:
            B += 1
        elif model2 == truth:
            C += 1
        else:
            D += 1
    chi2_Mc = (abs(B-C) - 1) ** 2 / (B + C)
    return chi2_Mc

import math

def TwoMatchedSamplest_test(y1, y2):
    """
    TODO
    :param y1: runs of algorithm 1
    :param y2: runs of algorithm 2
    :return: the test statistic t-value
    """
    # t_value = np.random.uniform(0, 1)
    d = y1 - y2
    dhat = sum(d) / len(d)
    sigma = math.sqrt(1 / (len(y1) - 1) * sum([(di - dhat) ** 2 for di in d]))
    t_value = math.sqrt(len(y1)) * dhat / sigma
    return t_value


def Friedman_test(errors):
    """
    TODO
    :param errors: the error values of different algorithms on different datasets
    :return: chi2_F: the test statistic chi2_F value
    :return: FData_stats: the statistical data of the Friedan test data, you can add anything needed to facilitate
    solving the following post hoc problems
    """
    chi2_F = np.random.uniform(0, 1)
    FData_stats = {'errors': errors}

    n, k = len(errors), len(errors[0])
    # print(n, k)
    R = np.argsort(np.argsort(errors))
    Rhat = sum(sum(R)) / (n * k)
    
    SStotal = 0
    Rdot_list = []
    for j in range(k):
        Rdot = 0
        for i in range(n):
            Rdot += R[i][j] / n
        Rdot_list.append(Rdot)
        SStotal += n * ((Rdot - Rhat) ** 2)
    
    SSerror = 0
    for i in range(n):
        for j in range(k):
            SSerror += (R[i][j] - Rhat) ** 2 / (n * (k - 1))
    
    # print(SStotal, SSerror)
    chi2_F = SStotal / SSerror

    FData_stats['Rank_dot'] = Rdot_list

    return chi2_F, FData_stats


def Nemenyi_test(FData_stats):
    """
    TODO
    :param FData_stats: the statistical data of the Friedan test data to be utilized in the post hoc problems
    :return: the test statisic Q value
    """
    Q_value = np.empty_like([1])
    errors = FData_stats['errors']
    Rdot_list = FData_stats['Rank_dot']

    n, k = len(errors), len(errors[0])
    Q_value = []
    for j1 in range(k):
        Q_j1 = []
        for j2 in range(k):
            if j2 <= j1:
                Q_j1.append(0)
                continue
            Q_j1.append((Rdot_list[j1] - Rdot_list[j2]) / math.sqrt(k * (k + 1) / (6 * n)))
        # print(Q_j1)
        Q_value.append(Q_j1)

    return Q_value


def box_plot(FData_stats):
    """
    TODO
    :param FData_stats: the statistical data of the Friedan test data to be utilized in the post hoc problems
    """
    print(FData_stats)
    """
    TODO
    refer to: https://blog.csdn.net/hustqb/article/details/77717026
    """
    import matplotlib.pyplot as plt
    import numpy as np

    np.random.seed(19680801)
    all_data = [np.random.normal(0, std, size=100) for std in range(1, 4)]
    labels = ['x1', 'x2', 'x3']

    bplot = plt.boxplot(all_data, patch_artist=True, labels=labels)  # 设置箱型图可填充
    plt.title('Rectangular box plot')

    colors = ['pink', 'lightblue', 'lightgreen']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)  # 为不同的箱型图填充不同的颜色

    # plt.yaxis.grid(True)
    plt.xlabel('Three separate samples')
    plt.ylabel('Observed values')
    plt.savefig('image.jpg')
    pass


def main(args):
    # (a)
    labels, prediction_A, prediction_B = load_data_MNTest()
    chi2_Mc = McNemar_test(labels, prediction_A, prediction_B)

    # (b)
    y1, y2 = load_data_TMStTest()
    t_value = TwoMatchedSamplest_test(y1, y2)

    # (c)
    errors = load_data_FTest()
    chi2_F, FData_stats = Friedman_test(errors)

    # (d)
    Q_value = Nemenyi_test(FData_stats)

    # (e)
    box_plot(FData_stats)


if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('ex03')

    cmdline_parser.add_argument('-v', '--verbose', default='INFO', choices=['INFO', 'DEBUG'], help='verbosity')
    cmdline_parser.add_argument('--seed', default=12345, help='Which seed to use', required=False, type=int)
    args, unknowns = cmdline_parser.parse_known_args()
    np.random.seed(args.seed)
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')
    main(args)
