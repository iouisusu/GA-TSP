import os
import turtle
import numpy as np
import my_utils.ga_tsp_config as conf
import my_utils.ga
import matplotlib.pyplot as plt

from my_utils.dw import Video, Pic


# 从tsp数据集中提取城市坐标
def read_tsp_data(path):
    tsp_data = np.empty((0, config.pos_dimension))
    # 打开并读取文件内容
    with open(path, 'r') as file:
        lines = file.readlines()
    # 找到 NODE_COORD_SECTION 标签行的索引
    start_index = 0
    for index, line in enumerate(lines):
        if line.strip() == 'NODE_COORD_SECTION':
            start_index = index + 1
            break
    # 读取城市坐标数据
    for line in lines[start_index:]:
        # 跳过 EOF 行
        if line.strip() == 'EOF':
            break
        # 清理行首和行尾的空白字符
        cleaned_line = line.strip()
        # 分割坐标值、转换成浮点数、加入list
        coordinate_values = list(map(float, cleaned_line.split()[1:]))
        # 如果城市坐标不为空，则添加到总坐标列表
        tsp_data = np.vstack([tsp_data, coordinate_values])

    return tsp_data


# 计算城市的距离矩阵
def build_dist_mat(input_list):
    n = config.city_num
    # 初始化一个大小为 n×n 的全零 numpy 数组
    dist_mat = np.zeros([n, n])
    for i in range(n):
        for j in range(i+1, n):
            # 坐标差向量
            d = input_list[i, :] - input_list[j, :]
            # 欧几里得距离
            dist = np.linalg.norm(d)
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist
    return dist_mat


config = conf.get_config()

# 随机生成城市坐标
# city_pos_list = np.random.rand(config.city_num, config.pos_dimension)
# city_pos_list = np.random.uniform(1, 10, size=(config.city_num, config.pos_dimension))
# 调用tsp数据集的城市坐标
city_pos_list = read_tsp_data(config.city_data)
# 处理距离矩阵
city_dist_mat = build_dist_mat(city_pos_list)
print("\ncity_pos_list: \n", city_pos_list)
print("\ncity_dist_mat: \n", city_dist_mat)
print("OK1\n")


# 遗传算法
ga = my_utils.ga.Ga(city_pos_list, city_dist_mat)
# result_list路线结果图,该list中存放每一代对应的路线。  fitness_list适应度结果图
result_list, fitness_list = ga.train(city_pos_list)

# 从 result_list 中取出最后一个元素，即代表最后一代遗传算法得到的最优解
result = result_list[-1]
answer = fitness_list[-1]
print("\nBest answer: \n", answer)
# 此时result中为最优解的城市序列，在city中映射出坐标序列
result_pos_list = city_pos_list[result, :]


# 使用图片展示最优路径和适应度曲线
pic = Pic(result_pos_list, fitness_list,answer)
pic.create_pic()

# 暂停程序，给时间来查看图形
# turtle.done()

# 使用视频动态展示最优路径寻找过程
# video = Video(config.pic_path, config.vid_name, 15)
# video.create_video()
# print("ok!")
# if os.name == 'nt':  # 判断是否为Windows系统
#     os.system(f'start {config.vid_name}')  # 打开视频文件
