import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# 假设我们有一个 DataFrame 'data'，RSRP 数据和一些输入特征
# 输入特征是 'feature1', 'feature2', ..., 'featureN'
from ult import load_dataset
import os
import pickle
from tensorflow.keras import backend as K

# 自定义 RMSE 指标
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
# 标准化数据，确保各个特征在相同尺度上
scaler = StandardScaler()

dataset_path = 'F:/wireless/huawei/upload_dataset/'
data = load_dataset(dataset_path, debug=False)

data = data.dropna(axis=0,how='any')
print("len(data):", len(data))

data = data.iloc[list(data["deltaH"]>1)]

p_data = data.astype("float")


print("step1, log...")
p_data["FB"] = np.log10(data["FB"].astype("float"))
p_data["Hb"] = np.log10(data["Hb"].astype("float"))
p_data["D"] = np.log10(data["D"].astype("float"))
p_data["Husr"] = np.log10(data["Husr"].astype("float"))
# p_data["deltaHv"] = np.log10(data["deltaHv"].astype("float"))
p_data["deltaH"] = np.log10(data["deltaH"].astype("float"))
p_data["L"] = np.log10(data["L"].astype("float"))

print("step2, z-score...")
p_data = (p_data - p_data.mean())/data.std()

p_data["RSRP"] = data["RSRP"]


inputs = p_data[['FB', 'RSP',
             'betaV', 'CCI', 'Hb', 'Husr', 'Hm',
             'deltaH', 'deltaHv', 'L', 'D',
             'UCI', 'cosA', 'cosB', 'cosC' ]].values
X = scaler.fit_transform(inputs)
with open('./model/annscaler.pickle', 'wb') as f_scaler:
    pickle.dump(scaler, f_scaler)
label = p_data[['RSRP']].values
# 将数据集划分为训练集、验证集和测试集 (8:1:1比例)
X_train, X_temp, y_train, y_temp = train_test_split(X, label, test_size=0.8, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(X_train.shape[1])
# 构建 MLP模型
model = Sequential()

# 输入层和第一个隐藏层
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # 假设我们有 N 个输入特征

# 第二个隐藏层
model.add(Dense(32, activation='relu'))

# 输出层：1 个神经元，用于预测 RSRP
model.add(Dense(1, activation='linear'))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error',metrics=[rmse])
# 训练模型，并在验证集上进行评估
history = model.fit(X_train, y_train,
                    epochs=10,            # 训练的迭代次数
                    batch_size=32,         # 每批次使用的样本数
                    validation_data=(X_val, y_val),  # 验证集
                    verbose=1)            # 输出训练信息

# 在测试集上评估模型
y_pred = model.predict(X_test)

# 计算测试集的均方误差（MSE）
mse_test = mean_squared_error(y_test, y_pred)
print(f'Test MSE: {mse_test}')
# 计算 RMSE（均方根误差）
rmse_test = np.sqrt(mse_test)
print(f'Test RMSE: {rmse_test}')

# 计算 MAPE（平均绝对百分比误差）
mape_test = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f'Test MAPE: {mape_test}%')
# 如果你想获得其他评价指标，比如 R^2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2}')
# 保存模型
model.save('./model/mlp.h5')


# 计算每个 epoch 的 MSE 和 RMSE
train_rmse = np.sqrt(np.array(history.history['loss']))  # 训练损失的 RMSE
val_rmse = np.sqrt(np.array(history.history['val_loss']))  # 验证损失的 RMSE
# 创建 DataFrame
rmse_df = pd.DataFrame({
    'Epoch': np.arange(1, len(train_rmse) + 1),
    'Training RMSE': train_rmse,
    'Validation RMSE': val_rmse
})

# 保存为 CSV 文件
rmse_df.to_csv('./model/rmse_results.csv', index=False)
print("RMSE results have been saved to './model/rmse_results.csv'")