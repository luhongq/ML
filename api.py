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

# 全局变量初始化
grid_names = ['area_alt','RSRP', 'COST231', 'SPM', 'TR38901', 'LR', 'KNN', 'DTR', 'RR', 'Lasso', 'GBR', 'RFR', 'ETR', 'BR']
grid_size = 80

# 初始化每个网格数据
grid_data = {name: np.full((grid_size, grid_size), 0) for name in grid_names}

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
    plt.title('terrain profile', fontsize=14)
    plt.legend(loc='upper right')

    plt.savefig('F://jsdaima/result/elevation_plot.svg',format='svg')  # 保存第一张图
    plt.show()

    # 第二张图：有四个子图，每个子图包含 RSRP 和不同的列，颜色指定
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    colors = ['black', 'red', 'green', 'darkblue']  # 定义线条颜色

    # 第一子图：RSRP、Cost231、SPM、TR38901
    axs[0, 0].plot(df['distance'], df['RSRP'], label='RSRP', color=colors[0])
    axs[0, 0].plot(df['distance'], df['COST231'], label='Cost231', color=colors[1])
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
    plt.savefig('F://jsdaima/result/rsrp_subplots.svg', format='svg')
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
    global  building_polygons, grid_data

    # 获取前端传递的JSON数据
    data = request.get_json()
    file_name = data.get('name')
    print(f"文件名：{file_name}")

    # 设置新的路径
    # new_path = os.path.join('../xvym46l1.jp.wksmym.top/', file_name.lstrip('./'))
    data = pd.read_csv(file_name)
    print(f'开始创建 {file_name} 区域')

    # 重置 area_alt 和 building_polygons
    grid_data['area_alt'] = np.full((grid_size, grid_size), data['Husr'].min())
    building_polygons.clear()

    # 重置每个网格数据
    for name in grid_names:
        grid_data[name] = np.full((grid_size, grid_size), 0)

    # 逐行处理数据
    for _, row in data.iterrows():
        x, y = int(row['x']), int(row['y'])
        hm, husr = row['Hm'], row['Husr']

        # 更新高度和海拔信息
        grid_data['RSRP'][x, y] = row['RSRP']
        grid_data['area_alt'][x, y] = husr+hm


        filter_name=[name for name in grid_names if name != 'area_alt']
        # 更新每个 grid_name 的网格值
        for name in filter_name:
            grid_data[name][x, y] = row[name]

        # 检查建筑高度，若满足条件则创建建筑区域
        if hm > 5:
            building = Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
            building_polygons.append(building)


    grid_data['area_alt'][0, 0] = data['Hb'].values[0]

    # 批量进行插值并更新
    for name in grid_names:
        grid_data[name] = kriging_interpolation(grid_data[name])

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

def get_path():
    global  building_polygons, grid_data

    # 获取前端传递的JSON数据
    data = request.get_json()
    x = data['x']
    y = data['y']
    path = []

    # 确保坐标在网格范围内
    target_x = min(max(x, 0), 79)
    target_y = min(max(y, 0), 79)
    line = LineString([(0, 0), (x, y)])
    intersections = []

    # 获取建筑物与路径的交点
    for building in building_polygons:
        if line.intersects(building):
            intersection = line.intersection(building)
            intersections.append(intersection)

    # 遍历交点并添加到路径
    for idx, intersection in enumerate(intersections):
        if isinstance(intersection, LineString):
            for x_coord, y_coord in intersection.coords:
                grid_x, grid_y = int(round(x_coord)), int(round(y_coord))
                point_data = {
                    "distance": round(np.sqrt(x_coord**2 + y_coord**2) * 5, 2),
                    "ele": int(grid_data['area_alt'][grid_x, grid_y])
                }
                # 动态添加每个网格的数据
                for name in grid_names:
                    point_data[name] = float(grid_data[name][grid_x, grid_y])
                path.append(point_data)

    # 计算总距离（米）
    total_distance = np.sqrt((target_x * 5) ** 2 + (target_y * 5) ** 2)

    # 采样点生成
    distances = np.linspace(0, total_distance, round(total_distance / 2.5))
    for distance in distances:
        ratio = distance / total_distance if total_distance > 0 else 0
        x_coord = int(ratio * target_x)
        y_coord = int(ratio * target_y)

        point_data = {
            "distance": round(distance, 2),
            "ele": int(round(grid_data['area_alt'][x_coord, y_coord], 2))
        }
        # 动态添加每个网格的数据
        for name in grid_names:
            point_data[name] = float(grid_data[name][x_coord, y_coord])
        path.append(point_data)

    # 输出路径并保存为 CSV
    print(path)
    df = pd.DataFrame(path).sort_values(by='distance')
    df.to_csv('F://jsdaima/result/resultdown.csv', index=False)
    draw(df)

    return jsonify(path)

app.run(host='127.0.0.1', port=5000, debug=True)
