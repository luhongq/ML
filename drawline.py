
# -*- coding: gbk -*-
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import numpy as np
import os,re
from shapely.geometry import Polygon, LineString
import pandas as pd
from pykrige.ok import OrdinaryKriging
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ������ɫ������ -> �� -> ��
colors = [(0, 0, 1), (0, 1, 0), (1, 1, 0)]  # ��ɫ����ɫ����ɫ
custom_cmap = LinearSegmentedColormap.from_list("BlueGreenYellow", colors)
# ȫ�ֱ�����ʼ��
grid_names = ['RSRP', 'COST231', 'SPM', 'TR38901', 'LR', 'KNN', 'DTR', 'RR', 'Lasso', 'GBR', 'RFR', 'ETR', 'BR']
grid_size = 80

# ��ʼ��ÿ����������
grid_data = {name: np.full((grid_size, grid_size), 0) for name in grid_names}

building_polygons = []

def kriging_interpolation(grid_data):
    # ʶ��0ֵ��ͷ�0ֵ��
    grid_x, grid_y = np.indices(grid_data.shape)

    # �ҵ���֪�㣨����ֵ�㣩
    known_x = grid_x[grid_data != 0].astype(float)  # ȷ��Ϊ float ����
    known_y = grid_y[grid_data != 0].astype(float)  # ȷ��Ϊ float ����
    known_values = grid_data[grid_data != 0].astype(float)  # ȷ��Ϊ float ����

    # �ҵ�δ֪�㣨��ֵ�㣩
    unknown_x = grid_x[grid_data == 0].astype(float)  # ȷ��Ϊ float ����
    unknown_y = grid_y[grid_data == 0].astype(float)  # ȷ��Ϊ float ����

    # ����������ֵģ��
    OK = OrdinaryKriging(known_x, known_y, known_values, variogram_model='linear', verbose=False, enable_plotting=False)

    # ��δ֪����в�ֵ
    z, _ = OK.execute('points', unknown_x, unknown_y)

    # �ò�ֵ������0ֵ��
    grid_data[grid_data == 0] = z

    return grid_data

def draw(df):
    # ��һ��ͼ���� distance Ϊ x��ele Ϊ y�����ͼ�·�Ϊǳ��ɫ
    plt.figure(figsize=(10, 6))

    plt.plot(df['distance'], df['ele'], label='Elevation', color='blue')
    plt.fill_between(df['distance'], df['ele'], color='#00ffff', alpha=0.5)  # �����ɫ

    # ���� y �᷶Χ
    y_min = df['ele'].min()
    y_max = df['ele'].max()
    y_range = y_max - y_min
    # �ر�����
    plt.grid(False)

    # �Զ����� y �����Сֵ�������Ǵ� 0 ��ʼ
    plt.ylim(bottom=y_min - y_range / 6, top=y_max + y_range / 6)
    plt.xlim(left=0)
    plt.xlabel('distance (m)', fontsize=12)
    plt.ylabel('elevation (m)', fontsize=12)
    plt.title('terrain profile', fontsize=14)
    plt.legend(loc='upper right')

    plt.savefig('F://jsdaima/result/elevation_plot.svg',format='svg')  # �����һ��ͼ
    plt.show()

    # �ڶ���ͼ�����ĸ���ͼ��ÿ����ͼ���� RSRP �Ͳ�ͬ���У���ɫָ��
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    colors = ['black', 'red', 'green', 'darkblue']  # ����������ɫ

    # ��һ��ͼ��RSRP��Cost231��SPM��TR38901
    axs[0, 0].plot(df['distance'], df['RSRP'], label='RSRP', color=colors[0])
    axs[0, 0].plot(df['distance'], df['COST231'], label='Cost231', color=colors[1])
    axs[0, 0].plot(df['distance'], df['SPM'], label='SPM', color=colors[2])
    axs[0, 0].plot(df['distance'], df['TR38901'], label='TR 38.901', color=colors[3])
    axs[0, 0].set_xlabel('Distance (m)')
    axs[0, 0].set_ylabel('Value')
    axs[0, 0].set_title('RSRP, Cost231, SPM, TR38901')
    axs[0, 0].legend(loc='upper right')

    axs[0, 0].grid(False)

    # �ڶ���ͼ��RSRP��LR��KNN��DTR
    axs[0, 1].plot(df['distance'], df['RSRP'], label='RSRP', color=colors[0])
    axs[0, 1].plot(df['distance'], df['LR'], label='LR', color=colors[1])
    axs[0, 1].plot(df['distance'], df['KNN'], label='KNN', color=colors[2])
    axs[0, 1].plot(df['distance'], df['DTR'], label='DTR', color=colors[3])
    axs[0, 1].set_xlabel('Distance (m)')
    axs[0, 1].set_ylabel('Value')
    axs[0, 1].set_title('RSRP, LR, KNN, DTR')
    axs[0, 1].legend(loc='upper right')


    axs[0, 1].grid(False)

    # ������ͼ��RSRP��RR��Lasso��GBR
    axs[1, 0].plot(df['distance'], df['RSRP'], label='RSRP', color=colors[0])
    axs[1, 0].plot(df['distance'], df['RR'], label='RR', color=colors[1])
    axs[1, 0].plot(df['distance'], df['Lasso'], label='Lasso', color=colors[2])
    axs[1, 0].plot(df['distance'], df['GBR'], label='GBR', color=colors[3])
    axs[1, 0].set_xlabel('Distance (m)')
    axs[1, 0].set_ylabel('Value')
    axs[1, 0].set_title('RSRP, RR, Lasso, GBR')
    axs[1, 0].legend(loc='upper right')


    axs[1, 0].grid(False)

    # ������ͼ��RSRP��RFR��ETR��BR
    axs[1, 1].plot(df['distance'], df['RSRP'], label='RSRP', color=colors[0])
    axs[1, 1].plot(df['distance'], df['RFR'], label='RFR', color=colors[1])
    axs[1, 1].plot(df['distance'], df['ETR'], label='ETR', color=colors[2])
    axs[1, 1].plot(df['distance'], df['BR'], label='BR', color=colors[3])
    axs[1, 1].set_xlabel('Distance (m)')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].set_title('RSRP, RFR, ETR, BR')
    axs[1, 1].legend(loc='upper right')


    axs[1, 1].grid(False)

    # ������ͼ֮��Ĳ���
    plt.tight_layout()

    # ����ڶ���ͼ
    plt.savefig('F://jsdaima/result/rsrp_subplots.svg', format='svg')
    plt.show()
def getarea(file_name):
    global  building_polygons, grid_data

    # ��ȡǰ�˴��ݵ�JSON����

    print(f"�ļ�����{file_name}")

    # �����µ�·��

    data = pd.read_csv(file_name)
    print(f'��ʼ���� {file_name} ����')

    # ���� area_alt �� building_polygons
    grid_data['area_alt'] = np.full((grid_size, grid_size), data['Husr'].min())
    building_polygons.clear()

    # ����ÿ����������
    for name in grid_names:
        grid_data[name] = np.full((grid_size, grid_size), 0)

    # ���д�������
    for _, row in data.iterrows():
        x, y = int(row['x']), int(row['y'])
        hm, husr = row['Hm'], row['Husr']

        # ���¸߶Ⱥͺ�����Ϣ
        grid_data['RSRP'][x, y] = row['RSRP']
        grid_data['area_alt'][x, y] = husr+hm
        filter_name=[name for name in grid_names if name != 'area_alt']
        # ����ÿ�� grid_name ������ֵ
        for name in filter_name:
            grid_data[name][x, y] = row[name]

        # ��齨���߶ȣ������������򴴽���������
        if hm > 5:
            building = Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
            building_polygons.append(building)


    grid_data['area_alt'][0, 0] = data['Hb'].values[0]

    # # �������в�ֵ������
    # for name in grid_names:
    #     grid_data[name] = kriging_interpolation(grid_data[name])

    print('�����ɹ�')
    # ����ͼ��
    plt.figure(figsize=(8, 8))  # ����ͼ���С

    # �������룬������������ֵΪ0�Ĳ����ڸ�
    masked_data = np.ma.masked_where(grid_data['RSRP'] == 0, grid_data['RSRP'])

    # ������ɫӳ�䣬��ֵ��ʾΪ��ɫ
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')  # �����루��ֵ���֣���ʾΪ��ɫ

    # ����ͼ��
    plt.imshow(masked_data, cmap=cmap, interpolation='nearest', origin='lower')
    plt.colorbar(label='Value')  # �����ɫ�������ñ�ǩ
    plt.title("80x80 Grid Visualization (Zeroes as White)")  # ���ñ���
    plt.xlabel("Y")  # ���� x ���ǩ
    plt.ylabel("X")  # ���� y ���ǩ
    plt.show()

    return {'status': 200}


def draw_line(x,y):
    # ����ͼ��
    plt.figure(figsize=(8, 8))

    # �������룬������������ֵΪ0�Ĳ����ڸ�
    masked_data = np.ma.masked_where(grid_data['RSRP'] == 0, grid_data['RSRP'])

    # ������ɫӳ�䣬��ֵ��ʾΪ��ɫ

    custom_cmap.set_bad(color='white')

    # ������������
    plt.imshow(masked_data, cmap=custom_cmap, interpolation='nearest',origin='lower')
    plt.colorbar(label='Value')
    plt.title("80x80 Grid Visualization with Line from Origin to Target")
    plt.xlabel("Y")
    plt.ylabel("X")



    # ����ԭ�㵽Ŀ��λ�õ�����
    plt.plot([0, y], [0, x], color='red', linewidth=2, linestyle='--', label="Line from (0, 0) to Target")

    # ��������յ�
    plt.scatter(0, 0, color='blue', s=50, label="Origin (0, 0)")
    plt.scatter(y, x, color='green', s=50, label=f"Target ({y}, {x})")

    # ���ͼ��
    plt.legend(loc="upper right")

    # ��ʾͼ��
    plt.show()
def get_path(x,y):
    global  building_polygons, grid_data

    # ��ȡǰ�˴��ݵ�JSON����


    path = []

    # ȷ������������Χ��
    target_x = min(max(x, 0), 79)
    target_y = min(max(y, 0), 79)
    line = LineString([(0, 0), (x, y)])
    intersections = []

    # ��ȡ��������·���Ľ���
    for building in building_polygons:
        if line.intersects(building):
            intersection = line.intersection(building)
            intersections.append(intersection)

    # �������㲢��ӵ�·��
    for idx, intersection in enumerate(intersections):
        if isinstance(intersection, LineString):
            for x_coord, y_coord in intersection.coords:
                grid_x, grid_y = int(round(x_coord)), int(round(y_coord))
                point_data = {
                    "distance": round(np.sqrt(x_coord**2 + y_coord**2) * 5, 2),
                    "ele": int(grid_data['area_alt'][grid_x, grid_y])
                }
                # ��̬���ÿ�����������
                for name in grid_names:
                    point_data[name] = float(grid_data[name][grid_x, grid_y])
                path.append(point_data)

    # �����ܾ��루�ף�
    total_distance = np.sqrt((target_x * 5) ** 2 + (target_y * 5) ** 2)

    # ����������
    distances = np.linspace(0, total_distance, round(total_distance / 2.5))
    for distance in distances:
        ratio = distance / total_distance if total_distance > 0 else 0
        x_coord = int(ratio * target_x)
        y_coord = int(ratio * target_y)

        point_data = {
            "distance": round(distance, 2),
            "ele": int(round(grid_data['area_alt'][x_coord, y_coord], 2))
        }
        # ��̬���ÿ�����������
        for name in grid_names:
            point_data[name] = float(grid_data[name][x_coord, y_coord])
        path.append(point_data)

    # ���·��������Ϊ CSV
    print(path)
    df = pd.DataFrame(path).sort_values(by='distance')

    draw(df)

    return path
if __name__ == '__main__':
    data_path='./csv/result/646_buildings319.csv'
    x,y=26,13
    getarea(data_path)

    draw_line(x,y)
    # �������в�ֵ������
    for name in grid_names:
        grid_data[name] = kriging_interpolation(grid_data[name])

    get_path(x,y)