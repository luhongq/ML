# code by zheng_saber
import pandas as pd
import numpy as np
import pickle, time, os

from ult import load_dataset,pictocsv

if __name__ == '__main__':
    dataset_path = "F:/wireless/huawei/upload_dataset/"
    pictocsv(dataset_path)
    p_data = load_dataset(dataset_path)

    res = pd.DataFrame([p_data.min(),p_data.max(),p_data.mean(),p_data.std()])
    res = res.round(2)
    print("len:",len(p_data), res)


