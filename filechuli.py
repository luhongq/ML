import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
colun=['FB', 'RSP','RSRP',
                     'betaV', 'CCI', 'Hb', 'Husr', 'Hm',
                     'deltaH', 'deltaHv', 'L', 'D',
                     'UCI','cosA', 'cosB', 'cosC','ekparam','x','y','LOS','COST231', 'SPM', 'TR38901', 'LR', 'KNN', 'DTR', 'RR', 'Lasso', 'GBR', 'RFR', 'ETR', 'BR']
coulunyuashi=['FB','RSP',
             'betaV', 'CCI', 'Hb', 'Husr', 'Hm',
             'deltaH', 'deltaHv', 'L', 'D',
             'UCI', 'cosA', 'cosB', 'cosC' ]
#文件处理
# 读取原始pickle文件，并按小区编号分为多个csv文件

def load_dataset(dataset_path):
    all_data = pd.DataFrame(columns=colun)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for file in os.listdir(dataset_path):
        file_path=os.path.join(dataset_path,file)

        with open(file_path, 'rb') as f:
            data = pd.read_pickle(f)
        all_data = pd.concat([all_data,data], ignore_index=True)
    return all_data

def spilt(all_data,out_path):
    # 获取该列值相同的所有唯一值
    unique_values = all_data['CI'].unique()

    # 按照该列的唯一值分割成多个CSV文件
    for value in tqdm(unique_values):
        if value:
        # 选择该列值等于当前唯一值的行
            subset = all_data[all_data['CI'] == value]

            # 构建输出文件名
            output_filename = os.path.join(out_path,str(value)+'.csv')

            # 将subset保存为新的CSV文件
            subset.to_csv(output_filename, index=False)

    print("数据集处理完成并已分割为多个 CSV 文件")


#筛选文件
def filter_and_save_csv(input_folder, output_folder,temfolder,debug):
    """
    依次读取目录下的所有CSV文件，筛选数据，若uci在10-16之间但hm<5，舍去该值，统计建筑个数并将其保存到文件名中。
    :param input_folder: 输入文件夹，包含待处理的CSV文件
    :param output_folder: 输出文件夹，保存处理后的CSV文件
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有CSV文件
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)

            #
            data = pd.read_csv(file_path)
            data =data[~((data['UCI'] == 13) & (data['Hm'] > 20))]
            data=data[~((data['UCI'] == 14) & (data['Hm'] > 20))]
            data=data[~((data['UCI'] == 12) & ((data['Hm'] < 20) | (data['Hm'] >40)))]
            data = data[~((data['UCI'] == 11) & ((data['Hm'] < 40) | (data['Hm'] > 60)))]
            data = data[~((data['UCI'] == 10) & (data['Hm'] < 60))]
            data = data[~((data['UCI'].isin([1,2,3,4,5,6,7,8,9,17,18,19,20])) & (data['Hm'] > 20))]

            #删去同一个小区、同一位置、UCI与5*5网格内差异较大的值
            removeindice =get_crossed_buildings(data)

            data=data.drop(removeindice)


            data = data.drop(columns=[col for col in data.columns if col.startswith('Unnamed')], errors='ignore')

            # 删除所有包含空白值（空字符串或 NaN）的列
            cols_to_drop = data.columns[data.isin(['', ' ']).any() | data.isna().any()]
            data = data.drop(columns=cols_to_drop, errors='ignore')
            #判断是否视距

            data.to_csv(file_path)
            print(f'已筛选{file_path}数据')


def get_crossed_buildings(data):

    grid_size = 80
    grid_height = np.full((grid_size, grid_size), np.nan)
    uci_grid = np.full((grid_size, grid_size), np.nan)
    grid_alt = np.full((grid_size, grid_size), 0)
    # 定义搜索范围为 5x5 的网格，即偏移量范围为 -5 到 5
    search_range = 5
    duplicate_entries = []
    duplicate_indices = []  # 用于存储被覆盖行的索引
    filled_grid_info = {}  # key: (x, y) -> value: previous_index
    for index, row in data.iterrows():
        x, y, hm, uci, husr = int(row['x']), int(row['y']), row['Hm'], row['UCI'], row['Husr']
        # 检查该位置是否已经有值
        if np.isnan(grid_height[x, y]):  # 如果为空，则直接填入建筑高度
            grid_height[x, y] = hm
            uci_grid[x, y] = uci
            grid_alt[x, y] = husr
            filled_grid_info[(x, y)] = index  # 记录当前行的索引
        else:
            # 如果该位置已有值，记录为重复数据，稍后处理
            duplicate_entries.append((x, y, hm, uci, husr, index))


    # 处理重复的数据
    for x, y, hm, uci, husr, index in duplicate_entries:
        # 如果该位置已有值，获取5x5范围内的 UCI 值进行比较
        closest_hm = grid_height[x, y]  # 初始化为已有的建筑高度

        closest_uci = uci_grid[x, y]
        closest_index = filled_grid_info[(x, y)]
        uci_diff = []
        preuci_diff = []
        # 遍历5x5范围内的邻居
        for dx in range(-search_range, search_range + 1):
            for dy in range(-search_range, search_range + 1):
                nx, ny = x + dx, y + dy
                # 检查邻居是否在网格范围内
                if 0 <= nx < grid_size and 0 <= ny < grid_size and not np.isnan(uci_grid[nx, ny]):
                    neighbor_uci = uci_grid[nx, ny]
                    # 计算当前输入 UCI 与邻居 UCI 的差异
                    uci_diff.append(abs(neighbor_uci - uci))
                    preuci_diff.append(abs(neighbor_uci - closest_uci))
        if np.sum(uci_diff) < np.sum(preuci_diff):
            uci_grid[x, y] = uci
            grid_alt[x, y] = husr
            grid_height[x, y] = hm
            duplicate_indices.append(closest_index)

        elif np.sum(uci_diff) == np.sum(preuci_diff):
            if hm >= closest_hm:
                duplicate_indices.append(closest_index)
            else:
                duplicate_indices.append(index)
        else:
            duplicate_indices.append(index)

    return duplicate_indices

#将多个csv文件合成几个pickle文件
def convert_csv_to_pickle(csv_folder, output_folder, num_pickles=8):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Step 1: 读取文件夹中的所有CSV文件
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    all_data = pd.DataFrame()

    # Step 2: 逐个读取并合并到一个大的DataFrame中
    for csv_file in tqdm(csv_files):
        file_path = os.path.join(csv_folder, csv_file)
        data = pd.read_csv(file_path)
        all_data = pd.concat([all_data, data], ignore_index=True)

    #Step 3: 将数据划分为num_pickles个数据块
    chunk_size = len(all_data) // num_pickles
    data_chunks = [all_data[i:i + chunk_size] for i in range(0, len(all_data), chunk_size)]

    # 如果最后一块数据不足，合并到前一块
    if len(data_chunks) > num_pickles:
        data_chunks[-2] = pd.concat([data_chunks[-2], data_chunks[-1]])
        data_chunks = data_chunks[:-1]

    # Step 4: 保存每个数据块为一个pickle文件
    for i, chunk in enumerate(data_chunks):
        pickle_file = os.path.join(output_folder, f'data_chunk_{i + 1}.pickle')
        with open(pickle_file, 'wb') as f:
            pickle.dump(chunk, f)

    print(f'Successfully saved {num_pickles} pickle files.')
datapath=''
out_path=''
spilt(load_dataset(dataset_path=datapath,out_path=out_path))