# -*- coding: gbk -*-
import numpy as np
import os,re
from shapely.geometry import Polygon, LineString
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import numpy as np
from scipy.interpolate import make_interp_spline

app = Flask(__name__)
CORS(app)
area_alt=np.full((80, 80), 0)
RSRP = np.full((80, 80), 0)
COST231 = np.full((80, 80), 0)
SPM = np.full((80, 80), 0)
TR38901 = np.full((80, 80), 0)
LR = np.full((80, 80), 0)
KNN = np.full((80, 80), 0)
DTR = np.full((80, 80), 0)
RR = np.full((80, 80), 0)
Lasso = np.full((80, 80), 0)
GBR = np.full((80, 80), 0)
RFR = np.full((80, 80), 0)
ETR = np.full((80, 80), 0)
BR = np.full((80, 80), 0)

building_polygons = []



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
    plt.title('dixing', fontsize=14)
    plt.legend(loc='upper right')

    plt.savefig('elevation_plot.svg',format='svg')  # �����һ��ͼ
    plt.show()

    # �ڶ���ͼ�����ĸ���ͼ��ÿ����ͼ���� RSRP �Ͳ�ͬ���У���ɫָ��
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    colors = ['black', 'red', 'green', 'darkblue']  # ����������ɫ

    # ��һ��ͼ��RSRP��Cost231��SPM��TR38901
    axs[0, 0].plot(df['distance'], df['RSRP'], label='RSRP', color=colors[0])
    axs[0, 0].plot(df['distance'], df['Cost231'], label='Cost231', color=colors[1])
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
    plt.savefig('rsrp_subplots.svg', format='svg')
    plt.show()
@app.route('/api/get-files', methods=['POST'])
def get_files():
    directory = './csv/result/'  # ָ�����ļ�Ŀ¼
    files = os.listdir(directory)
    csv_files = csv_files = sorted(
    [os.path.splitext(file)[0] for file in files if file.endswith('.csv')],
    key=lambda x: int(re.search(r'(\d+)_buildings(\d+)', x).group(1)) if re.search(r'(\d+)_buildings(\d+)', x) else float('inf')
)


    return jsonify(csv_files)




file_name=''
grid_names = ['RSRP','COST231', 'SPM', 'TR38901', 'LR', 'KNN', 'DTR', 'RR', 'Lasso', 'GBR','RFR','ETR','BR']
@app.route('/api/getarea', methods=['POST'])
def getarea():
    global area_alt,file_name,building_polygons
    data = request.get_json()  # ��ȡǰ�˴��ݵ�JSON����
    file_name = data.get('name')  # �����ļ���

    data =pd.read_csv(file_name)
    print('��ʼ����'+file_name+'����')

    grid_size = 80

    grid_height = np.full((grid_size, grid_size), 0)

    grid_alt = np.full((grid_size, grid_size),  data['Husr'].min())

    for index, row in data.iterrows():
        x, y, hm,husr = int(row['x']), int(row['y']), row['Hm'],row['Husr']

        grid_height[x, y] = hm

        grid_alt[x,y]=husr
        for name in grid_names:
            globals()[name][x, y] = row[name]  # ��̬��ȡ��Ӧ�ı���

        if hm > 5:
            # ����һ��С�Ľ������򣬼���ÿ��������һ����λ������
            x, y = row['x'], row['y']
            building = Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
            building_polygons.append(building)
    area_alt=grid_alt+grid_height

    area_alt[0,0]=data['Hb'].values[0]

    for name in grid_names+['area_alt']:
        globals()[name] = kriging_interpolation(globals()[name]) # ��̬��ȡ��Ӧ�ı���
    print('�����ɹ�')
    return jsonify({'status': 200})




# �����������ݵĿ�����ֵ����
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


@app.route('/api/getcross', methods=['POST'])
def getcross():
    data = request.get_json()  # ��ȡǰ�˴��ݵ�JSON����

    x = data['x']
    y = data['y']
    path = []

    # ȷ������������Χ��
    target_x = min(max(x, 0), 79)
    target_y = min(max(y, 0), 79)
    line = LineString([(0, 0), (x, y)])
    intersections = []
    for building in building_polygons:
        if line.intersects(building):
            intersection = line.intersection(building)
            intersections.append(intersection)
    for idx, intersection in enumerate(intersections):
        if isinstance(intersection, LineString):  # �ж��Ƿ�Ϊ LineString
            for x, y in intersection.coords: # ���걣��С�������λ
                grid_x, grid_y = int(round(x)), int(round(y))
                path.append({"distance": np.sqrt(x**2 + y**2) *5,"ele":int(area_alt[grid_x, grid_y]),
                                    'RSRP':float(RSRP[grid_x,grid_y]),
                                    'Cost231':float(COST231[grid_x,grid_y]),
                                    'SPM':float(SPM[grid_x,grid_y]),
                                    'TR38901':float(TR38901[grid_x,grid_y]),
                                    'LR':float(LR[grid_x,grid_y]),
                                    'KNN':float(KNN[grid_x,grid_y]),
                                    'DTR':float(DTR[grid_x,grid_y]),
                                    'RR':float(RR[grid_x,grid_y]),
                                    'Lasso':float(Lasso[grid_x,grid_y]),
                                    'GBR':float(GBR[grid_x,grid_y]),
                                    'RFR':float(RFR[grid_x,grid_y]),
                                    'ETR':float(ETR[grid_x,grid_y]),
                                    'BR':float(BR[grid_x,grid_y])} )

    # �����ܾ��루�ף�
    total_distance = np.sqrt((target_x * 5) ** 2 + (target_y * 5) ** 2)

    # ����������
    distances = np.linspace(0, total_distance, round(total_distance/2.5))


    for distance in distances:
        # �����Ӧ����������
        ratio = distance / total_distance if total_distance > 0 else 0
        x = int(ratio * target_x)
        y = int(ratio * target_y)

        ele = area_alt[x,y]

        path.append({
            'distance': round(distance, 2),
            'ele': int(round(ele, 2)),
            'RSRP':float(RSRP[x,y]),
            'Cost231':float(COST231[x,y]),
            'SPM':float(SPM[x,y]),
            'TR38901':float(TR38901[x,y]),
            'LR':float(LR[x,y]),
            'KNN':float(KNN[x,y]),
            'DTR':float(DTR[x,y]),
            'RR':float(RR[x,y]),
            'Lasso':float(Lasso[x,y]),
            'GBR':float(GBR[x,y]),
            'RFR':float(RFR[x,y]),
            'ETR':float(ETR[x,y]),
            'BR':float(BR[x,y])
        })
    print(path)
    df=pd.DataFrame(path).sort_values(by='distance')
    df.to_csv('resultdown.csv', index=False)
    draw(df)

    return jsonify(path)

app.run(host='127.0.0.1', port=5000, debug=True)
