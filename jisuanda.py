import pandas as pd
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# 设定建筑的高度阈值
BUILDING_THRESHOLD = 5
import numpy as np
print(np.arctan([1]))
# # 设定起始点和目标点
# origin = (0, 0)
# target = (61, 39)
# # 初始化绘图
# fig, ax = plt.subplots()
# # 从CSV文件中读取数据
# # 假设你已经有一个包含x, y, hm等数据的DataFrame
# df = pd.read_csv('csv/ucifoilter/18_buildings38.csv')
#
#
#
# # 创建建筑物集合
# building_polygons = []
#
# # 遍历 DataFrame，hm > 5 的为建筑
# for index, row in df.iterrows():
#     if row['Hm'] > BUILDING_THRESHOLD:
#         # 创建一个小的建筑区域，假设每个网格是一个单位正方形
#         x, y = row['x'], row['y']
#         building = Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
#         building_polygons.append(building)
#         # 将建筑多边形绘制到图像上
#         patch = patches.Polygon(building.exterior.coords, closed=True, edgecolor='black', facecolor='gray')
#         ax.add_patch(patch)
# # 设置坐标轴
# ax.set_xlim(0, 80)
# ax.set_ylim(0, 80)
# # 定义从起点到目标点的连线
# line = LineString([origin, target])
# x, y = line.xy
# ax.plot(x, y, label="Line", color='blue')
# # 显示图像
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()
#
#
# # 检查连线是否与建筑物相交
# intersections = []
# for building in building_polygons:
#     if line.intersects(building):
#         intersection = line.intersection(building)
#         intersections.append(intersection)
#
#
# for idx, intersection in enumerate(intersections):
#     if isinstance(intersection, LineString):  # 判断是否为 LineString
#         rounded_coords = [(round(x, 2), round(y, 2)) for x, y in intersection.coords]  # 坐标保留小数点后两位
#         print(f"Intersection {idx + 1}: {rounded_coords}")
#
