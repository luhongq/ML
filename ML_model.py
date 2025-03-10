# code by zheng_saber

import numpy as np
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings("ignore", category=FutureWarning)
from ult import load_dataset, train_data
import pickle

dataset_path = 'F:/wireless/huawei/filter_dataset/'
data = load_dataset(dataset_path, debug=False)

data = data.dropna(axis=0,how='any')



print("len(data):", len(data))
print((data['deltaH'] < 0).sum())

p_data = data.astype("float")


print("step1, log...")
p_data["FB"] = data["FB"].astype("float")
p_data["Hb"] = data["Hb"].astype("float")
p_data["D"] = np.log10(data["D"].astype("float"))
p_data["Husr"] =np.log10(data["Husr"].astype("float"))

p_data["deltaH"] = np.sign(data["deltaH"]) * np.log10(np.abs(data["deltaH"].astype("float")) + 1)
p_data["L"] = np.log10(data["L"].astype("float"))
print('data_std:',data.std)
print("step2, z-score...")

mean_values = p_data.mean()
std_values = data.std()
p_data = (p_data - mean_values)/std_values
# 保存均值和标准差
with open('./model/mean_values.pkl', 'wb') as fw:
    pickle.dump(mean_values, fw)
with open('./model/std_values.pkl', 'wb') as fw:
    pickle.dump(std_values, fw)
print('ok')
p_data["RSRP"] = data["RSRP"]


inputs = p_data[['FB','RSP',
             'betaV', 'CCI', 'Hb', 'Husr', 'Hm',
             'deltaH', 'deltaHv', 'L', 'D',
             'UCI', 'cosA', 'cosB', 'cosC' ]].values

label = p_data[['RSRP']].values



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =\
      train_test_split(inputs, label, test_size=0.2, shuffle=True)

print( "train:",np.shape(X_train) )
print( "test:",np.shape(X_test) )





dir_path = './model/ML/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)



from sklearn.ensemble import ExtraTreesRegressor
print('极端随机树')
etr = ExtraTreesRegressor(n_estimators=100)
train_data("ETR", etr, X_train, X_test, Y_train.ravel(), Y_test)

from sklearn.neighbors import KNeighborsRegressor
print('KNN回归')
knn = KNeighborsRegressor(weights="uniform")
train_data("KNN", knn,
                  X_train, X_test, Y_train.ravel(), Y_test)


from sklearn.linear_model import LinearRegression
print('线性回归')
lr = LinearRegression()
train_data("LR", lr,
                  X_train, X_test, Y_train, Y_test)


# from sklearn.linear_model import ElasticNet
# print(' ElasticNet 回归')
# en = ElasticNet(alpha=0.1, l1_ratio=0.7)
# train_data("EN", en, X_train, X_test, Y_train, Y_test)

from sklearn.linear_model import Ridge
print('岭回归')
ridge = Ridge(alpha=1.0)
train_data("RR", ridge, X_train, X_test, Y_train, Y_test)

from sklearn.linear_model import Lasso
print('Lasso 回归 ')
lasso = Lasso(alpha=0.1)
train_data("Lasso", lasso, X_train, X_test, Y_train, Y_test)

from sklearn import tree
print('决策树回归')
dtr = tree.DecisionTreeRegressor()
train_data("DTR", dtr,
                  X_train, X_test, Y_train, Y_test)
#
from sklearn.ensemble import RandomForestRegressor
#
print('随机森林回归')
rfr = RandomForestRegressor(n_estimators=100)
train_data("RFR", rfr, X_train, X_test, Y_train.ravel(), Y_test)

from sklearn.ensemble import GradientBoostingRegressor
print('梯度提升回归')
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
train_data("GBR", gbr, X_train, X_test, Y_train.ravel(), Y_test)
#
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

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
print('袋装法')
# 使用决策树作为基模型的袋装法
bagging = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=10, random_state=0)
train_data("BR", bagging,
                  X_train, X_test, Y_train.ravel(), Y_test)


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