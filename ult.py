# code by zheng_saber
import pandas as pd

import time, os
from tqdm import tqdm

def get_filename(root_dir):
    filenames = []

    for root, dirs, files in os.walk(root_dir, topdown=False):

        for name in files:

            # print("files: ",os.path.join(root, name),name)
            file_name = name
            file_content = os.path.join(root, name)
            filenames.append([file_name,file_content])
    return filenames

def load_dataset1(dataset_path, debug=False):
    t = time.time()
    sample_cnt = 0


    # 获取所有文件的路径
    path = get_filename(dataset_path)
    all_data = pd.DataFrame(columns=['D3','HR','TILT','FQ','OBS','RSRP' ])
    # 遍历每个文件
    for file_name, file_content in tqdm(path):
        sample_cnt += 1


        with open(file_content, 'rb') as test:
            pb_data = pd.read_csv(test)


        # 只选择需要的列
        pb_data = pb_data[['D3','HR','TILT','FQ','OBS','RSRP' ]]


        # 将该文件的数据追加到list中
        all_data = pd.concat([all_data, pb_data], ignore_index=True)

        if debug:
            if sample_cnt == 1:  # 如果debug模式开启，只处理一个文件
                break



    return all_data

def load_dataset(dataset_path, debug=False):
    t = time.time()
    sample_cnt = 0

    all_data = pd.DataFrame(columns=['CI', 'FB', 'RSP', 'RSRP',
                                     'betaV', 'CCI', 'Hb', 'Husr', 'Hm',
                                     'deltaH', 'deltaHv', 'L', 'D',
                                     'UCI', 'cosA', 'cosB', 'cosC'
                                     ])
    path = get_filename(dataset_path)

    for file_name, file_content in tqdm(path):
        sample_cnt += 1



        with open(file_content, 'rb') as test:
            pb_data = pd.read_pickle(test)


        pb_data = pb_data[['CI', 'FB', 'RSP', 'RSRP',
                         'betaV', 'CCI', 'Hb', 'Husr', 'Hm',
                         'deltaH', 'deltaHv', 'L', 'D',
                         'UCI', 'cosA', 'cosB', 'cosC'
                         ]]

        all_data = pd.concat([all_data, pb_data], ignore_index=True)
        if debug:
            if sample_cnt == 1:
                break

    print("dataset loaded in %.1f s" % (time.time() - t))
    return all_data


def pictocsv(rootdir):
    path = get_filename(rootdir)
    index=0
    for file_name, file_content in path:
        with open(file_content, 'rb') as test:
            index+=1
            pb_data = pickle.load(test)
            df = pd.DataFrame(pb_data)
            df.to_csv(rootdir+'part'+str(index)+'.csv', index=False, encoding='utf-8')


import numpy as np
import os, time, pickle




def train_data(name, model, x_train,x_test, y_train,y_test):
    t = time.time()
    print("Now doing:", name)

    model.fit(x_train,y_train)
    score = model.score(x_train, y_train)
    result = model.predict(x_train)
    if name in ['KNN', 'SVR_LR', 'GBDT', 'Lightbgm', 'RFR', 'GBR', 'ETR','STACK','VOTE','BR']:
        y_train=np.array(y_train).reshape(-1, 1)
    print(y_train.shape)
    result = np.reshape(result,[len(result),1])
    Residual = (result - y_train)
    ResidualSquare = Residual**2
    num_regress = len(result)

    MAPE = np.sum(np.divide(Residual,(y_train)))/num_regress*100
    RMSE = np.sqrt(np.mean(ResidualSquare) )

    print('n={%.2f}'%num_regress)
    print('R^2={%.2f}'%score)
    print('RMSE={%.2f}'%RMSE)
    print('MAPE={%.2f}'%MAPE)
    print("regression_method in %.1f s" % (time.time() - t))

    with open('./model/ML/' + name + '.pickle', 'wb') as fw:
        pickle.dump(model, fw)



    print("testing")
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    result = np.reshape(result, [len(result), 1])

    print(y_test.shape)
    Residual = (result - y_test)
    ResidualSquare = Residual ** 2
    num_regress = len(result)

    MAPE = np.sum(np.divide(Residual, (y_test))) / num_regress * 100
    RMSE = np.sqrt(np.mean(ResidualSquare))

    print('n={%.2f}' % num_regress)
    print('R^2={%.2f}' % score)
    print('RMSE={%.2f}' % RMSE)
    print('MAPE={%.2f}' % MAPE)
    print("regression_method in %.1f s" % (time.time() - t))

