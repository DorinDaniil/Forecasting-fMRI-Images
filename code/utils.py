import numpy as np
import pandas as pd
import os
import pickle
import re


def save_object(obj, filename):
    # Overwrites any existing file.
    with open(os.path.join(os.path.dirname(os.getcwd()), "src", filename), 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(os.path.join(os.path.dirname(os.getcwd()), "src", filename), 'rb') as inp:
        obj = pickle.load(inp)
        return obj


def preprocess(v):
    return (v - v.min()) / (v.max() - v.min())


def MSE(A):
    m, n = A.shape
    return (np.linalg.norm(A, "fro") ** 2) / (m * n)