import os
import pandas as pd
from tqdm import tqdm
import re
# # 正则表达式，匹配 "_buildings" 后面的数字
# pattern = r"_buildings(\d+)"
# #
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

            # 读取CSV文件
            data = pd.read_csv(file_path)
            data =data[~((data['UCI'] == 13) & (data['Hm'] > 20))]
            data=data[~((data['UCI'] == 14) & (data['Hm'] > 20))]
            data=data[~((data['UCI'] == 12) & ((data['Hm'] < 20) | (data['Hm'] >40)))]
            data = data[~((data['UCI'] == 11) & ((data['Hm'] < 40) | (data['Hm'] > 60)))]
            data = data[~((data['UCI'] == 10) & (data['Hm'] < 60))]
            data = data[~((data['UCI'].isin([1,2,3,4,5,6,7,8,9,17,18,19,20])) & (data['Hm'] > 20))]
            removeindice =get_crossed_buildings(data)
            print(removeindice)
            data=data.drop(removeindice)

            data=get_crossed_buildings1(data)
            data = data.drop(columns=[col for col in data.columns if col.startswith('Unnamed')], errors='ignore')

            # 删除所有包含空白值（空字符串或 NaN）的列
            cols_to_drop = data.columns[data.isin(['', ' ']).any() | data.isna().any()]
            data = data.drop(columns=cols_to_drop, errors='ignore')
            #判断是否视距
            data=islos(data)
            data.to_csv(file_path)
            # print(file_path)

            # # 统计 Hm > 5 的点数，并将其视为建筑个数
            building_count = (data['Hm'] >= 5).sum()
            #
            # # 构造新的文件名，包含建筑个数
            # base_name, ext = os.path.splitext(filename)
            # # new_base_name = re.sub(pattern, f"_buildings{building_count}", base_name)
            # new_filename = f"{base_name}_buildings{building_count}{ext}"
            # # new_filename = f"{new_base_name}{ext}"
            # output_file_path = os.path.join(output_folder, new_filename)
            #
            # # 保存筛选后的数据到输出目录
            # data.to_csv(output_file_path, index=False)
            # print(f"Processed and saved: {output_file_path}")
            #
            if building_count <= 50:
                target_dir = os.path.join(temfolder, '0-50_buildings')
            elif 50 < building_count <= 150:
                target_dir = os.path.join(temfolder, '50-150_buildings')
            else:
                target_dir = os.path.join(temfolder, '150_and_above_buildings')
            target_file_path = os.path.join(target_dir, filename)
            data.to_csv(target_file_path, index=False)
            # print(f"Moved file to: {target_file_path}")
# import os
# import time
# import pandas as pd
# import shutil
# #
# # #
# # # # 获取所有文件名的函数
# # def get_filename(root_dir):
# #     filenames = []
# #     for root, dirs, files in os.walk(root_dir, topdown=False):
# #         for name in files:
# #             if name.endswith('.csv'):  # 只读取CSV文件
# #                 print(name)
# #                 file_name = name
# #                 file_content = os.path.join(root, name)
# #                 filenames.append([file_name, file_content])
# #     return filenames
# # #
# # #
# # # 统计建筑数的函数
# # def count_buildings(data):
# #     # 统计 Hm > 5 的点数作为建筑数
# #     buildings_count = (data['Hm'] > 5).sum()
# #     return buildings_count
# # #
# # #
# # # 读取CSV文件并加载数据集，并将文件根据建筑数量移动到不同目录
# # def load_and_sort_dataset(dataset_path, output_base_dir, debug=False):
# #     t = time.time()
# #     sample_cnt = 0
# #
# #     # 获取所有CSV文件的路径
# #     path = get_filename(dataset_path)
# #
# #     # 依次读取每个文件
# #     for file_name, file_content in path:
# #         sample_cnt += 1
# #
# #         print(f"Processing {sample_cnt}: {file_content}")
# #
# #         # 读取CSV文件
# #         csv_data = pd.read_csv(file_content)
# #
# #         # 选择需要的列
# #         csv_data = csv_data[['CI', 'FB', 'RSP', 'RSRP',
# #                              'betaV', 'CCI', 'Hb', 'Husr', 'Hm',
# #                              'deltaH', 'deltaHv', 'L', 'D',
# #                              'UCI', 'cosA', 'cosB', 'cosC'
# #                              ]]
# #
# #         # 统计建筑数量（Hm > 5 的点数）
# #         buildings_count = count_buildings(csv_data)
# #         print(f"Buildings Count: {buildings_count}")
# #
# #         # 根据建筑数量将文件移动到不同的目录
# #         if buildings_count <= 50:
# #             target_dir = os.path.join(output_base_dir, '0-50_buildings')
# #         elif 50 < buildings_count <= 150:
# #             target_dir = os.path.join(output_base_dir, '50-150_buildings')
# #         else:
# #             target_dir = os.path.join(output_base_dir, '150_and_above_buildings')
# #
# #         # 如果目标目录不存在，则创建它
# #         if not os.path.exists(target_dir):
# #             os.makedirs(target_dir)
# #
# #         # 将文件移动到对应的目录
# #         target_file_path = os.path.join(target_dir, file_name)
# #         shutil.move(file_content, target_file_path)
# #         print(f"Moved file to: {target_file_path}")
# #
# #         # 如果开启debug模式，处理一个文件后停止
# #         if debug:
# #             if sample_cnt == 1:
# #                 break
# #
# #     print("Dataset processed and sorted in %.1f s" % (time.time() - t))
# #
# #
# # # 示例用法
# # dataset_path = './csv/ucifoilter/'  # 替换为实际的CSV文件目录路径
# # output_base_dir = './csv/filter/' # 设置输出的基础目录
# # load_and_sort_dataset(dataset_path, output_base_dir, debug=False)
#
# # # 示例用法
# # input_folder = './csv/tra/'  # 替换为实际的输入目录路径
# # output_folder = './csv/filter/'  # 替换为实际的输出目录路径
# # filter_and_save_csv(input_folder, output_folder)
#
#
#
import os
import pandas as pd
import numpy as np
import json
from shapely.geometry import Polygon, LineString
# 设定建筑的高度阈值
BUILDING_THRESHOLD = 5
# # 设定起始点
# # origin = (0, 0)
# #
# # # 获取所有 csv 文件的路径
# # directory = "csv/filter_all/18_buildings39.csv"
# #
# # # 读取 CSV 文件并合并
# # data_frames = []
# #
# #
# # # 合并所有 CSV 文件的数据
# # data = pd.read_csv(directory)
# #
# # # 提取出我们关心的列
# # data = data[['x', 'y', 'Hm', 'Husr', 'L']]
# #
# #
# #
# #
def get_crossed_buildings(data):
    building_polygons = []
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
        x, y, hm, uci,husr = int(row['x']), int(row['y']), row['Hm'], row['UCI'],row['Husr']
        # 检查该位置是否已经有值
        if np.isnan(grid_height[x, y]):  # 如果为空，则直接填入建筑高度
            grid_height[x, y] = hm
            uci_grid[x, y] = uci
            grid_alt[x,y]=husr
            filled_grid_info[(x, y)] = index  # 记录当前行的索引
        else:
            # 如果该位置已有值，记录为重复数据，稍后处理
            duplicate_entries.append((x, y, hm, uci,husr,index))

        # if hm > BUILDING_THRESHOLD:
        #     # 创建一个小的建筑区域，假设每个网格是一个单位正方形
        #     x, y = row['x'], row['y']
        #     building = Polygon([(x, y), (x + 1, y), (x + 1, y - 1), (x, y - 1)])
        #     building_polygons.append(building)

    # 处理重复的数据
    for x, y, hm, uci,husr,index in duplicate_entries:
        # 如果该位置已有值，获取5x5范围内的 UCI 值进行比较
        closest_hm = grid_height[x, y]  # 初始化为已有的建筑高度

        closest_uci =uci_grid[x,y]
        closest_index=filled_grid_info[(x,y)]
        uci_diff=[]
        preuci_diff=[]
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
        if np.sum(uci_diff)<np.sum(preuci_diff):
            uci_grid[x,y]=uci
            grid_alt[x,y]=husr
            grid_height[x,y]=hm
            duplicate_indices.append(closest_index)

        elif np.sum(uci_diff)==np.sum(preuci_diff):
            if hm>=closest_hm:
                duplicate_indices.append(closest_index)
            else:
                duplicate_indices.append(index)
        else:
            duplicate_indices.append(index)

    return duplicate_indices
#
# #
# #
# # # 示例用法
# # dataset_path = './csv/filter_all/'  # 替换为实际的CSV文件目录路径
# # output_base_dir = './csv/ucifoilter/' # 设置输出的基础目录
# # filter_and_save_csv(dataset_path, output_base_dir, debug=False)
# #
# #
#
def get_crossed_buildings1(data):

    building_polygons = []
    grid_size = 80
    grid_height = np.full((grid_size, grid_size), 0)
    ekpara_dict = { }
    grid_alt = np.full((grid_size, grid_size), 0)

    for index, row in data.iterrows():
        x, y, hm,husr = int(row['x']), int(row['y']), row['Hm'],row['Husr']

        grid_height[x, y] = hm

        grid_alt[x,y]=husr

        if hm > BUILDING_THRESHOLD:
            # 创建一个小的建筑区域，假设每个网格是一个单位正方形
            x, y = row['x'], row['y']
            building = Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
            building_polygons.append(building)
    for index, row in data.iterrows():
        ekpara=[]
        x, y, hm,husr = int(row['x']), int(row['y']), row['Hm'],row['Husr']
        line = LineString([(0,0), (x,y)])
        intersections = []
        for building in building_polygons:
            if line.intersects(building):
                intersection = line.intersection(building)
                intersections.append(intersection)
        for idx, intersection in enumerate(intersections):
            if isinstance(intersection, LineString):  # 判断是否为 LineString
                for x, y in intersection.coords: # 坐标保留小数点后两位
                    grid_x, grid_y = int(round(x)), int(round(y))
                    ekpara.append({"height": grid_height[grid_x,grid_y],
                                "distance": np.sqrt(x**2 + y**2) *5,
                                "elevation": grid_alt[grid_x,grid_y]})
        ekpara_dict[index]=ekpara
    data['ekparam'] = data.index.map(ekpara_dict)

    return data
#
# # dataset_path = './csv/tra/'  # 替换为实际的CSV文件目录路径
# # output_base_dir = './csv/ucifoilter/' # 设置输出的基础目录
# # temploder='./csv/filter/'
# filter_and_save_csv(dataset_path, output_base_dir,temploder, debug=False)
import ast
def islos(data):
    los_results = []

    for row, L, hm, hbs in zip(data['ekparam'].values, data['L'].values, data['Hm'].values, data['deltaH'].values):
        row=ast.literal_eval(row)
        los = True
        if row:
            # 每次提取两个字典:
            for i in range(0, len(row)-1, 2):

                d = (float(row[i]['distance']) + float(row[i + 1]['distance'])) / 2
                if hm / (L - d) > hbs / L:
                    los = False
                    break
        los_results.append(los)
    data['LOS'] = los_results
    return data

from formula import get_filename,load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from  tqdm import  tqdm
import os
import time
# 设置字体，SimHei 是黑体，确保系统有安装该字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
# plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题
# dataset_path = 'F:/wireless/huawei/csv/ucifoilter/'
# filter_and_save_csv(dataset_path,debug=False,output_folder=dataset_path,temfolder=temploder)
# data = load_dataset(dataset_path, debug=False)
# # data.to_csv('F:/wireless/huawei/sum.csv',index=False)
# # 绘制 'value' 列的直方图
# uci_counts = data['UCI'].value_counts().sort_values(ascending=False)
# # 绘制直方图
# # 绘制直方图
# fig, ax = plt.subplots(dpi=100, figsize=(8, 6))
# uci_counts.plot(kind='bar', color='skyblue', ax=ax,width=0.8)
#
# # 在每个柱顶显示出现次数
# for p in ax.patches:
#     ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha='center', va='baseline',fontsize=8,fontweight='bold')
# plt.ticklabel_format(style='plain', axis='y')
# # 显示图像
# plt.xlabel('地物类型索引')
# plt.ylabel('数量')
# plt.title('地物类型数量直方图')
# plt.xticks(rotation=0)  # 可选：让X轴标签水平显示
# plt.tight_layout()  # 防止字体超出图表
# plt.show()

import os
import pandas as pd
import pickle
# from tqdm import tqdm
# def convert_csv_to_pickle(csv_folder, output_folder, num_pickles=8):
#     # if not os.path.exists(output_folder):
#     #     os.makedirs(output_folder)
#     # Step 1: 读取文件夹中的所有CSV文件
#     csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
#     all_data = pd.DataFrame()
#
#     # Step 2: 逐个读取并合并到一个大的DataFrame中
#     for csv_file in tqdm(csv_files):
#         file_path = os.path.join(csv_folder, csv_file)
#         data = pd.read_csv(file_path)
#         all_data = pd.concat([all_data, data], ignore_index=True)
#     pickle_file = os.path.join(output_folder, f'150_and_above_buildings.pickle')
#     with open(pickle_file, 'wb') as f:
#         pickle.dump(all_data, f)
#     #Step 3: 将数据划分为num_pickles个数据块
#     chunk_size = len(all_data) // num_pickles
#     data_chunks = [all_data[i:i + chunk_size] for i in range(0, len(all_data), chunk_size)]
#
#     # 如果最后一块数据不足，合并到前一块
#     if len(data_chunks) > num_pickles:
#         data_chunks[-2] = pd.concat([data_chunks[-2], data_chunks[-1]])
#         data_chunks = data_chunks[:-1]
#
#     # Step 4: 保存每个数据块为一个pickle文件
#     for i, chunk in enumerate(data_chunks):
#         pickle_file = os.path.join(output_folder, f'data_chunk_{i + 1}.pickle')
#         with open(pickle_file, 'wb') as f:
#             pickle.dump(chunk, f)
#
#     print(f'Successfully saved {num_pickles} pickle files.')
#
# #示例使用
# csv_folder = './csv/filter/150_and_above_buildings/'        # CSV 文件夹路径
# output_folder = './csv/filter/'  # Pickle 文件保存路径
# convert_csv_to_pickle(csv_folder, output_folder, num_pickles=8)
# import os
# import re
# from tqdm import tqdm
# # 定义CSV文件所在的目录
# directory = 'csv/ucifoilter/'  # 替换为你的目录路径
# output='csv/ucifoilter/gai/'
# 获取目录下所有CSV文件
# files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# 定义一个函数来提取文件名前的数字
# def extract_number(filename):
#     match = re.match(r'(\d+)_buildings(\d+)\.csv', filename)
#     if match:
#         return int(match.group(1)), int(match.group(2))
#     return None
#
# # 对文件名进行排序
# files.sort(key=lambda f: extract_number(f))
#
# # 重命名文件
# for i, filename in tqdm(enumerate(files)):
#     # 提取原始数字
#     num1, num2 = extract_number(filename)
#     if num1 is not None:
#         # 生成新的文件名，确保自然排序
#         new_filename = f'{i+1}_buildings{num2}.csv'
#
#
#
#
#         old_path = os.path.join(directory, filename)
#         new_path = os.path.join(output, new_filename)
#         os.rename(old_path, new_path)
#         print(f'Renamed: {filename} -> {new_filename}')
# def find_csv_files_with_ci(directory, ci_value):
#
#
#     # 遍历目录下的所有 CSV 文件
#     for file in os.listdir(directory):
#         if file.endswith('.csv'):
#             file_path = os.path.join(directory, file)
#
#             # 读取 CSV 文件
#             data = pd.read_csv(file_path)
#
#             # 检查 'ci' 列是否存在且是否包含目标值
#             if 'CI' in data.columns:
#                 if ci_value ==data['CI'].values[0]:
#                     break
#
#     return file_path
# print(find_csv_files_with_ci(directory,1424001))
def merge_csv_files(input_dir, output_dir):
    all_data = {}

    # 读取每个CSV文件并将数据存储到字典中
    for file in tqdm(os.listdir(input_dir)):
        if file.endswith('.csv'):
            file_path = os.path.join(input_dir, file)
            data = pd.read_csv(file_path)

            # 获取CI列的值
            ci_value = data['CI'].iloc[0]

            if ci_value not in all_data:
                all_data[ci_value] =  {'dataframes': [], 'filenames': []}

            all_data[ci_value]['dataframes'].append(data)
            all_data[ci_value]['filenames'].append(file)  # 保存文件名
    # 合并数据并保存
    for ci_value, info in all_data.items():
        merged_data = pd.concat(info['dataframes'], ignore_index=True)

        # 解析文件名以获取合并后的文件名
        first_numbers = [int(filename.split('_')[0]) for filename in info['filenames']]
        second_numbers = [int(filename.split('_buildings')[1].split('.')[0]) for filename in info['filenames']]  # 去掉扩展名
        output_filename = f"{min(first_numbers)}_buildings{sum(second_numbers)}.csv"

        output_path = os.path.join(output_dir, output_filename)


        merged_data.to_csv(output_path, index=False)

        print(f"合并并保存为: {output_path}")


inputdir='./csv/ucifoilter/'
outputdir='./csv/result/'
merge_csv_files(input_dir=inputdir,output_dir=outputdir)