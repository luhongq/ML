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
#�ļ�����
# ��ȡԭʼpickle�ļ�������С����ŷ�Ϊ���csv�ļ�

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
    # ��ȡ����ֵ��ͬ������Ψһֵ
    unique_values = all_data['CI'].unique()

    # ���ո��е�Ψһֵ�ָ�ɶ��CSV�ļ�
    for value in tqdm(unique_values):
        if value:
        # ѡ�����ֵ���ڵ�ǰΨһֵ����
            subset = all_data[all_data['CI'] == value]

            # ��������ļ���
            output_filename = os.path.join(out_path,str(value)+'.csv')

            # ��subset����Ϊ�µ�CSV�ļ�
            subset.to_csv(output_filename, index=False)

    print("���ݼ�������ɲ��ѷָ�Ϊ��� CSV �ļ�")


#ɸѡ�ļ�
def filter_and_save_csv(input_folder, output_folder,temfolder,debug):
    """
    ���ζ�ȡĿ¼�µ�����CSV�ļ���ɸѡ���ݣ���uci��10-16֮�䵫hm<5����ȥ��ֵ��ͳ�ƽ������������䱣�浽�ļ����С�
    :param input_folder: �����ļ��У������������CSV�ļ�
    :param output_folder: ����ļ��У����洦����CSV�ļ�
    """
    # ȷ������ļ��д���
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ���������ļ����е�����CSV�ļ�
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

            #ɾȥͬһ��С����ͬһλ�á�UCI��5*5�����ڲ���ϴ��ֵ
            removeindice =get_crossed_buildings(data)

            data=data.drop(removeindice)


            data = data.drop(columns=[col for col in data.columns if col.startswith('Unnamed')], errors='ignore')

            # ɾ�����а����հ�ֵ�����ַ����� NaN������
            cols_to_drop = data.columns[data.isin(['', ' ']).any() | data.isna().any()]
            data = data.drop(columns=cols_to_drop, errors='ignore')
            #�ж��Ƿ��Ӿ�

            data.to_csv(file_path)
            print(f'��ɸѡ{file_path}����')


def get_crossed_buildings(data):

    grid_size = 80
    grid_height = np.full((grid_size, grid_size), np.nan)
    uci_grid = np.full((grid_size, grid_size), np.nan)
    grid_alt = np.full((grid_size, grid_size), 0)
    # ����������ΧΪ 5x5 �����񣬼�ƫ������ΧΪ -5 �� 5
    search_range = 5
    duplicate_entries = []
    duplicate_indices = []  # ���ڴ洢�������е�����
    filled_grid_info = {}  # key: (x, y) -> value: previous_index
    for index, row in data.iterrows():
        x, y, hm, uci, husr = int(row['x']), int(row['y']), row['Hm'], row['UCI'], row['Husr']
        # ����λ���Ƿ��Ѿ���ֵ
        if np.isnan(grid_height[x, y]):  # ���Ϊ�գ���ֱ�����뽨���߶�
            grid_height[x, y] = hm
            uci_grid[x, y] = uci
            grid_alt[x, y] = husr
            filled_grid_info[(x, y)] = index  # ��¼��ǰ�е�����
        else:
            # �����λ������ֵ����¼Ϊ�ظ����ݣ��Ժ���
            duplicate_entries.append((x, y, hm, uci, husr, index))


    # �����ظ�������
    for x, y, hm, uci, husr, index in duplicate_entries:
        # �����λ������ֵ����ȡ5x5��Χ�ڵ� UCI ֵ���бȽ�
        closest_hm = grid_height[x, y]  # ��ʼ��Ϊ���еĽ����߶�

        closest_uci = uci_grid[x, y]
        closest_index = filled_grid_info[(x, y)]
        uci_diff = []
        preuci_diff = []
        # ����5x5��Χ�ڵ��ھ�
        for dx in range(-search_range, search_range + 1):
            for dy in range(-search_range, search_range + 1):
                nx, ny = x + dx, y + dy
                # ����ھ��Ƿ�������Χ��
                if 0 <= nx < grid_size and 0 <= ny < grid_size and not np.isnan(uci_grid[nx, ny]):
                    neighbor_uci = uci_grid[nx, ny]
                    # ���㵱ǰ���� UCI ���ھ� UCI �Ĳ���
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

#�����csv�ļ��ϳɼ���pickle�ļ�
def convert_csv_to_pickle(csv_folder, output_folder, num_pickles=8):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Step 1: ��ȡ�ļ����е�����CSV�ļ�
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    all_data = pd.DataFrame()

    # Step 2: �����ȡ���ϲ���һ�����DataFrame��
    for csv_file in tqdm(csv_files):
        file_path = os.path.join(csv_folder, csv_file)
        data = pd.read_csv(file_path)
        all_data = pd.concat([all_data, data], ignore_index=True)

    #Step 3: �����ݻ���Ϊnum_pickles�����ݿ�
    chunk_size = len(all_data) // num_pickles
    data_chunks = [all_data[i:i + chunk_size] for i in range(0, len(all_data), chunk_size)]

    # ������һ�����ݲ��㣬�ϲ���ǰһ��
    if len(data_chunks) > num_pickles:
        data_chunks[-2] = pd.concat([data_chunks[-2], data_chunks[-1]])
        data_chunks = data_chunks[:-1]

    # Step 4: ����ÿ�����ݿ�Ϊһ��pickle�ļ�
    for i, chunk in enumerate(data_chunks):
        pickle_file = os.path.join(output_folder, f'data_chunk_{i + 1}.pickle')
        with open(pickle_file, 'wb') as f:
            pickle.dump(chunk, f)

    print(f'Successfully saved {num_pickles} pickle files.')
datapath=''
out_path=''
spilt(load_dataset(dataset_path=datapath,out_path=out_path))