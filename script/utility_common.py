# Created on 25 Jan 2023 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Functions used commonly

import os
import sys
import csv
import time
import numpy as np


class Logger(object):
    def __init__(self, file_log, stream=sys.stdout):
        self.terminal = stream
        self.file_log = file_log
        with open(file_log, "w") as f:
            f.write("")

    def write(self, massage):
        self.terminal.write(massage)
        with open(self.file_log, "a") as f:
            f.write(massage)

    def flush(self):
        pass


def timestamp():
    print(f" --- DATETIME {time.ctime(time.time())} --- ")


def create_directory(path):
    try:
        os.makedirs(path)
    finally:
        return path


def logit(p):
    tol = 1e-10
    p = np.array(p, dtype="float")
    p[p <= tol] = tol
    p[p >= 1 - tol] = 1 - tol
    p = p / (1 - p)
    return np.log(p)


def logit_inverse(y):
    return 1 / (1 + np.exp(-y))


def unnormalize(scaler, yhat, yhat_std):
    mean, std = scaler.mean_[0], scaler.scale_[0]
    yhat = std * yhat + mean
    yhat_std = std * yhat_std
    return yhat, yhat_std


def euclidean_distance(vec, weights=None):
    if weights is None:
        dist = np.linalg.norm(vec)
    else:
        dist = np.sqrt(np.sum([x ** 2 * weight for (x, weight) in zip(vec, weights)]))
    return dist


def write_csv(path, raw, method="w"):
    csv_file = open(path, method, newline="")
    writer = csv.writer(csv_file)
    writer.writerow(raw)
    csv_file.close()


def list_str2value(alist, symbol=None, seperator=","):
    if symbol is None:
        symbol = ["[", "]", " "]
    for sym in symbol:
        alist = alist.replace(sym, "")
    alist = alist.split(seperator)
    return list(map(float, alist))


def tuple_str2value(alist):
    alist = alist.replace("(", "").replace(")", "").replace(" ", "")
    alist = alist.split(",")
    return tuple(map(float, alist))
