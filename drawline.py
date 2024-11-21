
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

# 定义颜色条：蓝 -> 绿 -> 黄
colors = [(0, 0, 1), (0, 1, 0), (1, 1, 0)]  # 蓝色、绿色、黄色
custom_cmap = LinearSegmentedColormap.from_list("BlueGreenYellow", colors)
# 全局变量初始化
grid_names = ['RSRP', 'COST231', 'SPM', 'TR38901', 'LR', 'KNN', 'DTR', 'RR', 'Lasso', 'GBR', 'RFR', 'ETR', 'BR']
grid_size = 80

# 初始化每个网格数据
grid_data = {name: np.full((grid_size, grid_size), 0) for name in grid_names}

building_polygons = []

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
def getarea(file_name):
    global  building_polygons, grid_data

    # 获取前端传递的JSON数据

    print(f"文件名：{file_name}")

    # 设置新的路径

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

    # # 批量进行插值并更新
    # for name in grid_names:
    #     grid_data[name] = kriging_interpolation(grid_data[name])

    print('创建成功')
    # 创建图形
    plt.figure(figsize=(8, 8))  # 设置图像大小

    # 创建掩码，将网格数据中值为0的部分掩盖
    masked_data = np.ma.masked_where(grid_data['RSRP'] == 0, grid_data['RSRP'])

    # 设置颜色映射，零值显示为白色
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')  # 将掩码（零值部分）显示为白色

    # 绘制图像
    plt.imshow(masked_data, cmap=cmap, interpolation='nearest', origin='lower')
    plt.colorbar(label='Value')  # 添加颜色条并设置标签
    plt.title("80x80 Grid Visualization (Zeroes as White)")  # 设置标题
    plt.xlabel("Y")  # 设置 x 轴标签
    plt.ylabel("X")  # 设置 y 轴标签
    plt.show()

    return {'status': 200}


def draw_line(x,y):
    # 创建图形
    plt.figure(figsize=(8, 8))

    # 创建掩码，将网格数据中值为0的部分掩盖
    masked_data = np.ma.masked_where(grid_data['RSRP'] == 0, grid_data['RSRP'])

    # 设置颜色映射，零值显示为白色

    custom_cmap.set_bad(color='white')

    # 绘制网格数据
    plt.imshow(masked_data, cmap=custom_cmap, interpolation='nearest',origin='lower')
    plt.colorbar(label='Value')
    plt.title("80x80 Grid Visualization with Line from Origin to Target")
    plt.xlabel("Y")
    plt.ylabel("X")



    # 绘制原点到目标位置的连线
    plt.plot([0, y], [0, x], color='red', linewidth=2, linestyle='--', label="Line from (0, 0) to Target")

    # 标记起点和终点
    plt.scatter(0, 0, color='blue', s=50, label="Origin (0, 0)")
    plt.scatter(y, x, color='green', s=50, label=f"Target ({y}, {x})")

    # 添加图例
    plt.legend(loc="upper right")

    # 显示图像
    plt.show()
def get_path(x,y):
    global  building_polygons, grid_data

    # 获取前端传递的JSON数据


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

    draw(df)

    return path
if __name__ == '__main__':
    data_path='./csv/result/646_buildings319.csv'
    x,y=26,13
    getarea(data_path)

    draw_line(x,y)
    # 批量进行插值并更新
    for name in grid_names:
        grid_data[name] = kriging_interpolation(grid_data[name])

    get_path(x,y)