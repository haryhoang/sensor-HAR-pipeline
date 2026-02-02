import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import re
import os
from scipy.stats import kurtosis, skew
from preprocessing import load_all_data

def run_pipeline(folder_path):
    return load_all_data(folder_path)

