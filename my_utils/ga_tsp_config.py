import argparse

parser = argparse.ArgumentParser(description='Configuration file')
arg_list = []


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_list.append(arg)
    return arg


data_arg = add_argument_group('Date')


# 城市数据集
data_arg.add_argument('--city_data', type=str, default="data/bayg29.tsp", help='data path')
# 城市数量
data_arg.add_argument('--city_num', type=int, default=29, help='city num')
# 城市位置维度
data_arg.add_argument('--pos_dimension', type=int, default=2, help='pos_dimension')
# 种群个体数（50-100）
data_arg.add_argument('--individual_num', type=int, default=70, help='individual num')
# 迭代次数
data_arg.add_argument('--gen_num', type=int, default=1000, help='generation num')
# 精英保留比例
data_arg.add_argument('--elite_ratio', type=float, default=0.04, help='probability of elite')
# 交叉概率
data_arg.add_argument('--cross_prob', type=float, default=0.98, help='probability of cross')
# 变异概率
data_arg.add_argument('--mutate_prob', type=float, default=0.05, help='probability of mutate')
# 路径图片存放地址
data_arg.add_argument('--pic_path', type=str, default="./video/pic", help='picture path')
# 路径视频存放地址
data_arg.add_argument('--vid_name', type=str, default="./video/vid/vi.mp4", help='video name')


# 设置参数配置
def get_config():
    config, unparsed = parser.parse_known_args()
    return config


# 打印配置参数
def print_config():
    config = get_config()
    print('\n')
    print('Data Config:')
    print('* city num:', config.city_num)
    print('* individual num:', config.individual_num)
    print('* generation num:', config.gen_num)
    print('* probability of mutate:', config.mutate_prob)
