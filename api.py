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
    # 第一张图：以 distance 为 x，ele 为 y，填充图下方为浅蓝色
    plt.figure(figsize=(10, 6))

    plt.plot(df['distance'], df['ele'], label='Elevation', color='blue')
    plt.fill_between(df['distance'], df['ele'], color='#00ffff', alpha=0.5)  # 填充颜色

    # 计算 y 轴范围
    y_min = df['ele'].min()
    y_max = df['ele'].max()
    y_range = y_max - y_min
    # 关闭网格
    plt.grid(False)

    # 自动调整 y 轴的最小值，而不是从 0 开始
    plt.ylim(bottom=y_min - y_range / 6, top=y_max + y_range / 6)
    plt.xlim(left=0)
    plt.xlabel('distance (m)', fontsize=12)
    plt.ylabel('elevation (m)', fontsize=12)
    plt.title('dixing', fontsize=14)
    plt.legend(loc='upper right')

    plt.savefig('elevation_plot.svg',format='svg')  # 保存第一张图
    plt.show()

    # 第二张图：有四个子图，每个子图包含 RSRP 和不同的列，颜色指定
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    colors = ['black', 'red', 'green', 'darkblue']  # 定义线条颜色

    # 第一子图：RSRP、Cost231、SPM、TR38901
    axs[0, 0].plot(df['distance'], df['RSRP'], label='RSRP', color=colors[0])
    axs[0, 0].plot(df['distance'], df['Cost231'], label='Cost231', color=colors[1])
    axs[0, 0].plot(df['distance'], df['SPM'], label='SPM', color=colors[2])
    axs[0, 0].plot(df['distance'], df['TR38901'], label='TR 38.901', color=colors[3])
    axs[0, 0].set_xlabel('Distance (m)')
    axs[0, 0].set_ylabel('Value')
    axs[0, 0].set_title('RSRP, Cost231, SPM, TR38901')
    axs[0, 0].legend(loc='upper right')

    axs[0, 0].grid(False)

    # 第二子图：RSRP、LR、KNN、DTR
    axs[0, 1].plot(df['distance'], df['RSRP'], label='RSRP', color=colors[0])
    axs[0, 1].plot(df['distance'], df['LR'], label='LR', color=colors[1])
    axs[0, 1].plot(df['distance'], df['KNN'], label='KNN', color=colors[2])
    axs[0, 1].plot(df['distance'], df['DTR'], label='DTR', color=colors[3])
    axs[0, 1].set_xlabel('Distance (m)')
    axs[0, 1].set_ylabel('Value')
    axs[0, 1].set_title('RSRP, LR, KNN, DTR')
    axs[0, 1].legend(loc='upper right')


    axs[0, 1].grid(False)

    # 第三子图：RSRP、RR、Lasso、GBR
    axs[1, 0].plot(df['distance'], df['RSRP'], label='RSRP', color=colors[0])
    axs[1, 0].plot(df['distance'], df['RR'], label='RR', color=colors[1])
    axs[1, 0].plot(df['distance'], df['Lasso'], label='Lasso', color=colors[2])
    axs[1, 0].plot(df['distance'], df['GBR'], label='GBR', color=colors[3])
    axs[1, 0].set_xlabel('Distance (m)')
    axs[1, 0].set_ylabel('Value')
    axs[1, 0].set_title('RSRP, RR, Lasso, GBR')
    axs[1, 0].legend(loc='upper right')


    axs[1, 0].grid(False)

    # 第四子图：RSRP、RFR、ETR、BR
    axs[1, 1].plot(df['distance'], df['RSRP'], label='RSRP', color=colors[0])
    axs[1, 1].plot(df['distance'], df['RFR'], label='RFR', color=colors[1])
    axs[1, 1].plot(df['distance'], df['ETR'], label='ETR', color=colors[2])
    axs[1, 1].plot(df['distance'], df['BR'], label='BR', color=colors[3])
    axs[1, 1].set_xlabel('Distance (m)')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].set_title('RSRP, RFR, ETR, BR')
    axs[1, 1].legend(loc='upper right')


    axs[1, 1].grid(False)

    # 调整子图之间的布局
    plt.tight_layout()

    # 保存第二张图
    plt.savefig('rsrp_subplots.svg', format='svg')
    plt.show()
@app.route('/api/get-files', methods=['POST'])
def get_files():
    directory = './csv/result/'  # 指定的文件目录
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
    data = request.get_json()  # 获取前端传递的JSON数据
    file_name = data.get('name')  # 解析文件名

    data =pd.read_csv(file_name)
    print('开始创建'+file_name+'区域')

    grid_size = 80

    grid_height = np.full((grid_size, grid_size), 0)

    grid_alt = np.full((grid_size, grid_size),  data['Husr'].min())

    for index, row in data.iterrows():
        x, y, hm,husr = int(row['x']), int(row['y']), row['Hm'],row['Husr']

        grid_height[x, y] = hm

        grid_alt[x,y]=husr
        for name in grid_names:
            globals()[name][x, y] = row[name]  # 动态获取对应的变量

        if hm > 5:
            # 创建一个小的建筑区域，假设每个网格是一个单位正方形
            x, y = row['x'], row['y']
            building = Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
            building_polygons.append(building)
    area_alt=grid_alt+grid_height

    area_alt[0,0]=data['Hb'].values[0]

    for name in grid_names+['area_alt']:
        globals()[name] = kriging_interpolation(globals()[name]) # 动态获取对应的变量
    print('创建成功')
    return jsonify({'status': 200})




# 处理网格数据的克里金插值函数
def kriging_interpolation(grid_data):
    # 识别0值点和非0值点
    grid_x, grid_y = np.indices(grid_data.shape)

    # 找到已知点（非零值点）
    known_x = grid_x[grid_data != 0].astype(float)  # 确保为 float 类型
    known_y = grid_y[grid_data != 0].astype(float)  # 确保为 float 类型
    known_values = grid_data[grid_data != 0].astype(float)  # 确保为 float 类型

    # 找到未知点（零值点）
    unknown_x = grid_x[grid_data == 0].astype(float)  # 确保为 float 类型
    unknown_y = grid_y[grid_data == 0].astype(float)  # 确保为 float 类型

    # 创建克里金插值模型
    OK = OrdinaryKriging(known_x, known_y, known_values, variogram_model='linear', verbose=False, enable_plotting=False)

    # 对未知点进行插值
    z, _ = OK.execute('points', unknown_x, unknown_y)

    # 用插值结果填充0值点
    grid_data[grid_data == 0] = z

    return grid_data


@app.route('/api/getcross', methods=['POST'])
def getcross():
    data = request.get_json()  # 获取前端传递的JSON数据

    x = data['x']
    y = data['y']
    path = []

    # 确保坐标在网格范围内
    target_x = min(max(x, 0), 79)
    target_y = min(max(y, 0), 79)
    line = LineString([(0, 0), (x, y)])
    intersections = []
    for building in building_polygons:
        if line.intersects(building):
            intersection = line.intersection(building)
            intersections.append(intersection)
    for idx, intersection in enumerate(intersections):
        if isinstance(intersection, LineString):  # 判断是否为 LineString
            for x, y in intersection.coords: # 坐标保留小数点后两位
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

    # 计算总距离（米）
    total_distance = np.sqrt((target_x * 5) ** 2 + (target_y * 5) ** 2)

    # 创建采样点
    distances = np.linspace(0, total_distance, round(total_distance/2.5))


    for distance in distances:
        # 计算对应的网格坐标
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
