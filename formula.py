import math

import numpy as np

from  tqdm import  tqdm
import warnings

# 屏蔽 FutureWarning 警告
warnings.filterwarnings("ignore", category=FutureWarning)
colun=['FB', 'RSP','RSRP',
                     'betaV', 'CCI', 'Hb', 'Husr', 'Hm',
                     'deltaH', 'deltaHv', 'L', 'D',
                     'UCI','ekparam','x','y','LOS']
colunyuanshi=['CI', 'FB', 'RSP', 'RSRP',
                                     'betaV', 'CCI', 'Hb', 'Husr', 'Hm',
                                     'deltaH', 'deltaHv', 'L', 'D',
                                     'UCI', 'cosA', 'cosB', 'cosC'
                                     ]
import os
import pandas as pd
import time
import ast
from ult import get_filename
# def get_filename(root_dir):
#     filenames = []
#
#     # 遍历所有子目录和文件
#     for root, dirs, files in os.walk(root_dir, topdown=False):
#         for name in files:
#             if name.endswith('.csv'):  # 只选择CSV文件
#
#                 file_name = name
#                 file_content = os.path.join(root, name)
#                 filenames.append([file_name, file_content])
#     return filenames

def load_dataset(dataset_path, debug=False):
    t = time.time()
    sample_cnt = 0

    # 使用list保存每个文件的数据
    all_data_list = []

    # 获取所有文件的路径
    path = get_filename(dataset_path)
    all_data = pd.DataFrame(columns=colun)
    # 遍历每个文件
    for file_name, file_content in tqdm(path):
        sample_cnt += 1

        # print(f"Processing {sample_cnt}: {file_content}")
        with open(file_content, 'rb') as test:
            pb_data = pd.read_pickle(test)
        # 使用pandas加载CSV文件
        # pb_data = pd.read_csv(file_content)

        # 打印每个文件的头部数据以验证
        # print(pb_data.head())

        # 只选择需要的列
        pb_data = pb_data[colun]


        # 将该文件的数据追加到list中
        all_data = pd.concat([all_data, pb_data], ignore_index=True)

        if debug:
            if sample_cnt == 1:  # 如果debug模式开启，只处理一个文件
                break


    print(type(all_data))
    # print("Dataset loaded in %.1f s" % (time.time() - t))
    return all_data





class FormulaModel:

    def __init__(self, formula, selected_features,model_name):
        """
        初始化模型，设定数学公式和所选特征
        :param formula: 一个函数，定义了预测的数学公式
        :param selected_features: 需要用于公式计算的特征名称列表
        """
        self.formula = formula
        self.selected_features = selected_features
        print(f'现在使用{model_name}模型预测')

    def fit(self, X_train, y_train):
        """
        模拟模型的训练过程
        :param X_train: 训练数据（DataFrame）
        :param y_train: 训练标签
        """
        start_time = time.time()
        # 只选择指定的特征用于训练
        X_train_selected = X_train[self.selected_features]
        # 构建设计矩阵（包含偏置项1的列，便于拟合常数项）
        A = np.column_stack([np.ones(X_train_selected.shape[0]), X_train_selected])

        # 最小二乘法求解
        self.coefficients_, residuals, rank, s = np.linalg.lstsq(A, y_train, rcond=None)
        print(f"拟合的系数: {self.coefficients_}")
        print(f"训练中，使用公式 {self.formula.__name__} 并选择特征 {self.selected_features}")
        end_time = time.time()
        print(f"训练完成，耗时 {end_time - start_time:.2f} 秒")

    def predict(self, X):
        """
        使用公式进行预测
        :param X: 输入特征数据（DataFrame）
        :return: 预测值
        """
        # 只选择指定的特征用于预测

        X_selected = X[self.selected_features]
        return self.formula(X_selected)

    def score(self, X_test, y_test):
        """
        测试模型的效果
        :param X_test: 测试数据特征（DataFrame）
        :param y_test: 测试数据标签
        :return: 模型的平均绝对误差
        """
        start_time = time.time()
        X_test=X_test[self.selected_features]
        y_pred = self.predict(X_test).reshape(-1, 1)
        y_test=np.array(y_test.values)
        # print(y_pred,y_test)
        error = np.mean(np.abs(y_pred - y_test))
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        mape = np.mean(np.abs((y_pred - y_test) / y_test)) * 100

        end_time = time.time()
        print(f"测试完成，耗时 {end_time - start_time:.2f} 秒")
        print(f"平均绝对误差: {error:.4f}")
        print(f"均方根误差: {rmse:.4f}")
        print(f"平均绝对百分比误差: {mape:.4f}")
        return error



def cost231(X):

    # 计算 ah
    # ah = (1.1 * np.log10(X['FB'].values) - 0.7) * (1.5 + X['Hm'].values) - (1.56 * np.log10(X['FB'].values) - 0.8)
    ah=3.2*(np.log10(11.75*(X['Hm'].values+1.5)))**2-4.97
    # 初始化 cell 和 cm 列
    cell = np.zeros(X.shape[0])
    cm = np.zeros(X.shape[0])

    # 对 UCI 列应用条件判断
    uci_conditions = X['UCI'].isin([4, 19])
    cell[uci_conditions] = -4.78 * (np.log10(X['FB'].values[uci_conditions]) ** 2) - 18.33 * np.log10(X['FB'].values[uci_conditions]) - 40.98
    cm[uci_conditions] = 0

    uci_conditions = X['UCI'].isin([5, 10, 11, 12, 13, 14, 15, 16, 20])
    cell[uci_conditions] = 0
    cm[uci_conditions] = 3

    uci_conditions = ~X['UCI'].isin([4, 19, 5, 10, 11, 12, 13, 14, 15, 16, 20])
    cell[uci_conditions] = -2 * (np.log10(X['FB'].values[uci_conditions] / 28) )** 2 - 5.4
    cm[uci_conditions] = 0

    # 计算 Lp
    Lp = 46.3 + 33.9 * np.log10(X['FB'].values) +(44.96 - 6.55 * np.log10(X['Hb'].values)) * np.log10(X['D'].values / 1000) -13.82 * np.log10(X['Hb'].values) - ah + cm + cell
    print(Lp)
    # 返回 RSP 和 Lp 的差值
    return np.array((X['RSP'].values - Lp).ravel())

def spm(X):
    # Ldif=0
    # Lp =K1+K2*np.log10(X['D'].values)+K3*np.log10(X['deltaH'].values-1.5-X['Hm'].values)+K4*Ldif+K5*np.log10(X['deltaH'].values)*np.log10(X['D'].values)+K6*(1.5 + X['Hm'].values)+K7*
    clutter = np.zeros(X.shape[0])

    clutter_condition=X['UCI'].isin([ 10, 11, 12, 13, 14, 15, 16])
    clutter[clutter_condition]=3.37


    Lp=23.5+44.9*np.log10(X['D'].values)-6.55*np.log10(X['Hb'].values)*np.log10(X['D'].values)+5.83*np.log10(np.abs(X['Hb'].values-(X['Husr'].values+ X['Hm'].values)-1.5))
    print(Lp)
    return np.array((X['RSP'].values - Lp).ravel())

def ekloss(X):
    Gt=0
    Gr=0
    f,hteff,hr,data1=X['FB'].values,X['deltaH'].values,X['Hm'].values+1.5,X['ekparam'].values

    a0=np.where(X['UCI'].isin([4, 17, 18, 19]), 1.5,
               np.where(X['UCI'].isin([1, 2, 3, 7, 8, 9]), 0.5, 2.5))
    X['y'] = X['y'].astype(float)
    X['x'] = X['x'].astype(float)
    h_string =X['deltaH'].values-X['Hm'].values-1.5
    a = np.where((X['x'] == 0) & (X['y'] == 0), 0,
                 np.where(X['x'] == 0, 90,
                 np.degrees(np.arctan(np.divide(X['y'].values, X['x'].values,
                               out=np.zeros_like(X['y'].values), where=X['x'].values != 0)))))
    d = X['D'].values/ 1000
    dmod = [min(max(5, value), 100) for value in d]
    amod =  [min(50, value) for value in a]

    r1 = 4.49 - 0.655 * np.log10(hteff)
    lamda = 300 / f
    Ad = []
    Adif = []
    e1 = []
    for data,r2,d1,lamda1,hr1 in tqdm(zip(data1,r1,d,lamda,hr)):  # 每行数据是一个包含多个字典的列表
        data=ast.literal_eval(data)
        r = [r2]
        dis = []
        h_bui = []


        for i in range(0, len(data), 2):
            dis.append((data[i]['distance'] + data[i + 1]['distance']) / 2000)
            h_bui.append((data[i]['elevation'] + data[i + 1]['elevation']) / 2)

        dis.append(d1)

        for i in range(len(dis) - 1):
            d_mid = (dis[i] + dis[i + 1]) / 2

            ci = -(h_string - h_bui[i])
            vi = ci * np.sqrt(2 * d1 / (lamda1 * d_mid * (d1 - d_mid)))
            r.append(2 + (r2 - 2) * (1 + np.tanh(2 * (vi + 1)) / 2))

        if len(data) / 2 == 1:
            Adif.append(5)
        elif len(data) / 2 == 2:
            Adif.append(9)
        elif len(data) / 2 == 3:
            Adif.append(12)
        elif len(data) / 2 == 4:
            Adif.append(14)
        elif len(data) / 2 >= 5:
            Adif.append(15)
        else:
            Adif.append(0)

        if hr1 <= 5:
            e1.append(10)
        elif 5 < hr1 < 10:
            e1.append(2 * hr1)
        else:
            e1.append(20)
        Ad.append(calculateExpression(r, dis)-(5*np.log10(5*d1+ 1) + 2))



    At = -13.82 * np.log10(hteff)

    Ar = -3 - e1 * np.log10(hr * e1 / 3)

    L0 = 69.6 + 26.2 * np.log10(f) + Ad + Ar + At + Gt + Gr
    print(L0, Ad, Ar, At)

    Alu = 1



    Aor = a0 * (amod - 35) * (1 + np.log10(10 / dmod)) / 25
    print(L0,Adif,Aor)
    return np.array(X['RSP'].values-(L0 + Adif + Alu + Aor))


def ekloss1(X):
    Gt = 0
    Gr = 0

    # 初始化列表
    Adif = []
    L0 = []

    for index, row in tqdm(X.iterrows()):
        # 逐行提取数据
        f = row['FB']  # 频率
        hteff = row['deltaH']  # 有效高度
        hr = row['Hm'] + 1.5  # 参考高度
        data1 = row['ekparam']  # 其他参数

        # 计算 a0
        a0 = np.where(row['UCI'] in [4, 17, 18, 19], 1.5,
                      np.where(row['UCI'] in [1, 2, 3, 7, 8, 9], 0.5, 2.5))

        # 计算 Adif
        data = ast.literal_eval(data1)
        Adif_val = 0
        if len(data) / 2 == 1:
            Adif_val = -5
        elif len(data) / 2 == 2:
            Adif_val = -9
        elif len(data) / 2 == 3:
            Adif_val = -12
        elif len(data) / 2 == 4:
            Adif_val = -14
        elif len(data) / 2 >= 5:
            Adif_val = -15
        else:
            Adif_val=0

        Adif.append(Adif_val)

        # 计算其他参数
        r2 = 4.49 - 0.655 * np.log10(hteff)
        d1 = row['D'] / 1000
        hr1 = hr

        lamda = 300 / f
        dis = []
        h_bui = []

        for i in range(0, len(data), 2):
            dis.append((data[i]['distance'] + data[i + 1]['distance']) / 2000)
            h_bui.append((data[i]['elevation'] + data[i + 1]['elevation']) / 2)

        dis.append(d1)

        # 计算信号衰减
        r = [r2]
        for i in range(len(dis) - 1):
            d_mid = (dis[i] + dis[i + 1]) / 2
            ci = -(hteff - h_bui[i])
            vi = ci * np.sqrt(2 * d1 / (lamda * d_mid * (d1 - d_mid)))
            r.append(2 + (r2 - 2) * (1 + np.tanh(2 * (vi + 1)) / 2))

        # 计算 L0
        At = -13.82 * np.log10(hteff)
        Ar = -3 - (2 if hr1 <= 5 else (2 * hr1 if hr1 < 10 else 20)) * np.log10(
            hr1 * (2 if hr1 <= 5 else (2 * hr1 if hr1 < 10 else 20)) / 3)

        L0_value = 69.6 + 26.2 * np.log10(f) + calculateExpression(r, dis) + Ar + At + Gt + Gr
        L0.append(L0_value)

    # 计算最终的 RSP 值
    Alu = 1
    Aor = a0 * (np.minimum(50, np.maximum(5, (X['D'].values / 1000) - 35))) * (
                1 + np.log10(10 / np.minimum(50, (X['D'].values / 1000)))) / 25
    print((np.array(L0) + np.array(Adif) + Alu + Aor))
    return np.array(X['RSP'].values - (np.array(L0) + np.array(Adif) + Alu + Aor))

def calculateExpression(r, d) :
    r = np.array(r)
    d = np.array(d)
    log_d = np.log10(d)

    # 初始项
    result = 10 * r[0] * log_d[0]

    # 剩余项
    result += np.sum(10 * r[1:] * (log_d[1:] - log_d[:-1]))


    return result

def tr38901(X):
    dBP=4*(X['deltaH'].values-1)*(X['Hm'].values+1.5)*X['FB'].values/300
    pl = np.where(
        X['L'].values <= 10,
        28 + 22 * np.log10(X['D'].values) + 20 * np.log10(X['FB'].values / 1000),
        np.where(
            (X['L'].values > 10) & (X['L'].values < dBP),
            28 + 22 * np.log10(X['D'].values) + 20 * np.log10(X['FB'].values / 1000),
            28 + 40 * np.log10(X['D'].values) + 20 * np.log10(X['FB'].values / 1000) -
            9 * np.log10(dBP ** 2 + (X['Hb'].values -X['Husr'].values- X['Hm'].values - 1.5) ** 2)
        )
    )

    # 计算非 LOS 情况下的 pl1
    # pl1_non_los = 13.54 + 39.08 * np.log10(X['D'].values) + 20 * np.log10(X['FB'].values / 1000) - 0.6 *(X['Hm'].values)
    pl1_non_los = 32.4 + 30 * np.log10(X['D'].values) + 20 * np.log10(X['FB'].values / 1000)
    a=np.where(X['LOS'].values,np.random.normal(0, 4),np.random.normal(0, 6))
    # 根据 LOS 状态选择 pl 或 pl1
    loss = np.where(X['LOS'].values, pl, pl1_non_los)+a
    print(loss)
    return np.array(X['RSP'].values-loss)
if __name__ == '__main__':

    dataset_path = 'F:/wireless/huawei/filter_dataset/'
    data = load_dataset(dataset_path, debug=False)
    print(dataset_path)
    data = data.dropna(axis=0, how='any')

    print("len(data):", len(data))

    data = data.iloc[list(data["deltaH"] > 1)]

    # colunfloat = ['FB', 'RSP', 'betaV', 'deltaHv', 'L', 'D', 'RSRP']
    # colunint = ['CCI', 'Hb', 'Husr', 'Hm', 'deltaH', 'UCI',]

    colunfloat=['FB', 'RSP','betaV', 'deltaHv', 'L', 'D','RSRP']
    colunint=['CCI', 'Hb', 'Husr', 'Hm','deltaH','UCI','x','y']
    colunlist=['ekparam']
    data[colunfloat]=data[colunfloat].astype("float")
    data[colunint]=data[colunint].astype('int')

    # print("step1, log...")
    # p_data["FB"] = np.log10(data["FB"].astype("float"))
    # p_data["Hb"] = np.log10(data["Hb"].astype("float"))
    # p_data["D"] = np.log10(data["D"].astype("float"))
    # p_data["Husr"] = np.log10(data["Husr"].astype("float"))
    # # p_data["deltaHv"] = np.log10(data["deltaHv"].astype("float"))
    # p_data["deltaH"] = np.log10(data["deltaH"].astype("float"))
    # p_data["L"] = np.log10(data["L"].astype("float"))
    #
    # print("step2, z-score...")
    # p_data = (p_data - p_data.mean()) / data.std()
    #
    # p_data["RSRP"] = data["RSRP"]

    inputs = data[colun]

    label = data[['RSRP']]
    # 打印输入数据的形状和第一个数据行
    print(inputs.shape, "inputs length:", len(inputs), "inputs first row:", inputs.iloc[0])

    # 打印标签数据的形状和第一个数据行
    print(label.shape, "label length:", len(label), "label first row:", label.iloc[0])

    model_name='cost231'
    # 创建模型实例，传入数学公式和选定的特征
    model = FormulaModel(cost231, selected_features=['FB','deltaH','UCI','D','RSP','Hm','Husr','Hb'],model_name=model_name)
    model.score(inputs, label)

    model_name='spm'
    model = FormulaModel(spm, selected_features=['deltaH', 'FB', 'D', 'RSP', 'Hm','UCI','Husr','Hb'],model_name=model_name)
    model.score(inputs, label)

    model_name = 'tr38901'
    model = FormulaModel(tr38901, selected_features=['deltaH', 'FB', 'D', 'L', 'RSP', 'Hm', 'LOS','Husr','Hb'],
                         model_name=model_name)
    model.score(inputs, label)
    # model_name='ek'
    # model = FormulaModel(ekloss1, selected_features=['deltaH', 'FB', 'D', 'RSP', 'Hm','ekparam','UCI','x','y','Husr','Hb'],model_name=model_name)
    # model.score(inputs, label)

