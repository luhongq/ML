# code by zheng_saber

import numpy as np
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import tqdm
from ult import get_filename
from sklearn.model_selection import KFold
import numpy as np
import pickle
import time


def train_data1(name, model, x,xtest ,y,ytest):
    t = time.time()
    print("Now doing:", name)

    # 设置10折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    total_score = 0
    total_rmse = 0
    total_mape = 0
    num_regress_total = 0

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 训练模型
        model.fit(x_train, y_train)

        # 计算训练集指标
        score = model.score(x_train, y_train)
        result = model.predict(x_train)

        # 如果需要将目标重整形
        if name in ['KNN', 'SVR_LR', 'GBDT', 'Lightbgm', 'RFR', 'GBR', 'ETR', 'STACK', 'VOTE', 'BR']:
            y_train = np.array(y_train).reshape(-1, 1)

        result = np.reshape(result, [len(result), 1])
        Residual = (result - y_train)
        ResidualSquare = Residual ** 2
        num_regress = len(result)

        MAPE = np.sum(np.abs(np.divide(Residual, y_train))) / num_regress * 100
        RMSE = np.sqrt(np.mean(ResidualSquare))

        print(f"Fold Training Results - R^2: {score:.2f}, RMSE: {RMSE:.2f}, MAPE: {MAPE:.2f}")

        # 计算测试集指标
        score = model.score(x_test, y_test)
        result = model.predict(x_test)
        result = np.reshape(result, [len(result), 1])

        Residual = (result - y_test)
        ResidualSquare = Residual ** 2
        num_regress_test = len(result)

        MAPE_test = np.sum(np.abs(np.divide(Residual, y_test))) / num_regress_test * 100
        RMSE_test = np.sqrt(np.mean(ResidualSquare))

        print(f"Fold Testing Results - R^2: {score:.2f}, RMSE: {RMSE_test:.2f}, MAPE: {MAPE_test:.2f}")

        # 累加各项指标
        total_score += score
        total_rmse += RMSE_test
        total_mape += MAPE_test
        num_regress_total += num_regress_test

    # 计算平均值
    avg_score = total_score / kf.get_n_splits()
    avg_rmse = total_rmse / kf.get_n_splits()
    avg_mape = total_mape / kf.get_n_splits()

    print(f"\nAverage Results over 10-fold Cross-Validation:")
    print(f"R^2: {avg_score:.2f}")
    print(f"RMSE: {avg_rmse:.2f}")
    print(f"MAPE: {avg_mape:.2f}")
    print(f"Regression completed in {time.time() - t:.1f} s")

    # 保存模型
    with open(f'./model/ML1/{name}.pickle', 'wb') as fw:
        pickle.dump(model, fw)
    print("testing")
    score = model.score(xtest, ytest)
    result = model.predict(xtest)
    result = np.reshape(result, [len(result), 1])

    print(ytest.shape)
    Residual = (result - ytest)
    ResidualSquare = Residual ** 2
    num_regress = len(result)

    MAPE = np.sum(np.divide(Residual, (ytest))) / num_regress * 100
    RMSE = np.sqrt(np.mean(ResidualSquare))

    print('n={%.2f}' % num_regress)
    print('R^2={%.2f}' % score)
    print('RMSE={%.2f}' % RMSE)
    print('MAPE={%.2f}' % MAPE)
    print("regression_method in %.1f s" % (time.time() - t))

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
import pandas as pd

dataset_path = 'F:/wireless/huawei/4Gtrain/'
data = load_dataset1(dataset_path, debug=False)

data = data.dropna(axis=0,how='any')



print("len(data):", len(data))


p_data = data.astype("float")



p_data["RSRP"] = data["RSRP"]


inputs = p_data[['D3','HR','TILT','FQ','OBS' ]].values

label = p_data[['RSRP']].values



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =\
      train_test_split(inputs, label, test_size=0.3, shuffle=True)

print( "train:",np.shape(X_train) )
print( "test:",np.shape(X_test) )





dir_path = './model/ML1/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


#
# from sklearn.ensemble import ExtraTreesRegressor
# print('极端随机树')
# etr = ExtraTreesRegressor(n_estimators=100)
# train_data("ETR", etr, X_train, X_test, Y_train.ravel(), Y_test)
#
# from sklearn.neighbors import KNeighborsRegressor
# print('KNN回归')
# knn = KNeighborsRegressor(weights="uniform")
# train_data("KNN", knn,
#                   X_train, X_test, Y_train.ravel(), Y_test)

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
print('线性回归')
lr = LinearRegression()
train_data1("LR", lr,
                  X_train, X_test, Y_train, Y_test)

with open('./model/ML1/' + 'LR' + '.pickle', 'rb') as f:
    model = pd.read_pickle(f)
model.predict(inputs)
# 进行预测
predictions = model.predict(inputs)

# 绘制预测值与真实值的对比图
plt.figure(figsize=(10, 6))

# 绘制真实值
plt.plot(label, label='True Labels', color='b', marker='o', linestyle='-', linewidth=1)

# 绘制预测值
plt.plot(predictions, label='Predicted Values', color='r', marker='x', linestyle='-', linewidth=1)

# 添加图例和标题
plt.legend()
plt.title('Comparison of Predicted Values and True Labels')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.show()

# from sklearn.linear_model import ElasticNet
# print(' ElasticNet 回归')
# en = ElasticNet(alpha=0.1, l1_ratio=0.7)
# train_data("EN", en, X_train, X_test, Y_train, Y_test)

# from sklearn.linear_model import Ridge
# print('岭回归')
# ridge = Ridge(alpha=1.0)
# train_data("RR", ridge, X_train, X_test, Y_train, Y_test)
#
# from sklearn.linear_model import Lasso
# print('Lasso 回归 ')
# lasso = Lasso(alpha=0.1)
# train_data("Lasso", lasso, X_train, X_test, Y_train, Y_test)
#
# from sklearn import tree
# print('决策树回归')
# dtr = tree.DecisionTreeRegressor()
# train_data("DTR", dtr,
#                   X_train, X_test, Y_train, Y_test)
# #
# from sklearn.ensemble import RandomForestRegressor
# #
# print('随机森林回归')
# rfr = RandomForestRegressor(n_estimators=100)
# train_data("RFR", rfr, X_train, X_test, Y_train.ravel(), Y_test)
#
# from sklearn.ensemble import GradientBoostingRegressor
# print('梯度提升回归')
# gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
# train_data("GBR", gbr, X_train, X_test, Y_train.ravel(), Y_test)
# #
#



# from xgboost import XGBRegressor
# print('XGBoost 回归')
# xgbr = XGBRegressor(n_estimators=100, learning_rate=0.1)
# train_data("XGBR", xgbr, X_train, X_test, Y_train, Y_test)






# from sklearn.ensemble import GradientBoostingRegressor
# print('gbdt回归')
# gbdt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.2, max_depth=4, random_state=42, subsample=0.85,max_features=0.9)
# train_data("GBDT", gbdt,
#                   X_train, X_test, Y_train.ravel(), Y_test)
#
#
# from lightgbm import  LGBMRegressor
# print(' LightGBM 回归')
# # 初始化模型
# lgbm_model = LGBMRegressor(
#     boosting_type='gbdt',         # 基于梯度提升决策树
#     objective='regression',       # 回归任务
#     n_estimators=100,             # 总迭代次数（树的数量）
#     learning_rate=0.2,           # 学习步长
#
#     feature_fraction=0.9,         # 特征采样比例
#     random_state=42,
#     bagging_fraction=0.85,  # 数据采样比例，85%的数据用于训练
#     bagging_freq= 5,  # 每5次迭代进行一次 bagging
# # 随机种子
#     max_depth=4
# )
# train_data("Lightbgm", lgbm_model,
#                   X_train, X_test, Y_train.ravel(), Y_test)

# from sklearn.ensemble import BaggingRegressor
# from sklearn.tree import DecisionTreeRegressor
# print('袋装法')
# # 使用决策树作为基模型的袋装法
# bagging = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=10, random_state=0)
# train_data("BR", bagging,
#                   X_train, X_test, Y_train.ravel(), Y_test)


# from sklearn.ensemble import VotingRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVR
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# # 初始化不同的模型
# model1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
# model2 = RandomForestRegressor(n_estimators=100)
# model3 = SVR(kernel='linear')
# print('投票法')
# # 使用投票法
# voting = VotingRegressor([('gb', model1), ('rf', model2), ('svr', model3)])
# train_data("VOTE", voting,
#                   X_train, X_test, Y_train.ravel(), Y_test)
#
# from sklearn.ensemble import StackingRegressor
# from sklearn.linear_model import Ridge
# from sklearn.svm import SVR
# print('堆叠法')
# #基模型
# base_models = [
#     ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)),
#     ('rf', RandomForestRegressor(n_estimators=100)),
# ]
#
# # 元模型
# meta_model = Ridge()
#
# # 使用堆叠法
# stacking = StackingRegressor(estimators=base_models, final_estimator=meta_model)
#
# train_data("STACK", stacking,
#                   X_train, X_test, Y_train.ravel(), Y_test)
#
#
# from sklearn.svm import SVR
# print('线性支持向量回归')
# l_svr = SVR(kernel='linear')
# train_data("SVR_LR"+tailname, l_svr,
#                   X_train, X_test, Y_train.ravel(), Y_test)