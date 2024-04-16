import math
import os

import numpy as np
import my_utils.ga_tsp_config as conf
import matplotlib.pyplot as plt
import random
from dw import *

city_dist_mat = None
draw_cnt = 0
plt.ion()

# 设置参数
config = conf.get_config()
# 个体基因长度--城市个数
gene_len = config.city_num
# 种群个体数量--生成解个数
individual_num = config.individual_num
# 遗传代数--迭代次数
gen_num = config.gen_num
# 强者比例
elite_ratio = config.elite_ratio
# 变异概率
mutate_prob = config.mutate_prob
# 交叉概率
cross_prob = config.cross_prob


# 深拷贝
def copy_list(old_arr: [int]):
    new_arr = []
    for i in old_arr:
        new_arr.append(i)
    return new_arr


# 个体类，设置该个体的路径及该个体的适应度
class Individual:
    def __init__(self, genes=None):
        if genes is None:
            # 生成一个包含从0到(gene_len-1)整数的列表
            genes = [i for i in range(gene_len)]
            random.shuffle(genes)
        self.genes = genes
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        # 计算个体适应度
        fitness = 0.0
        for i in range(gene_len - 1):
            # 起始城市和目标城市
            from_idx = self.genes[i]
            to_idx = self.genes[i + 1]
            fitness += city_dist_mat[from_idx, to_idx]
        # 连接首尾
        fitness += city_dist_mat[self.genes[-1], self.genes[0]]
        fitness = 1000.0 / max(fitness, 1e-3)  # 适应度倒数
        return fitness


# 遗传过程类 -- 选择、交叉、变异
class Ga:
    # dw = Draw()
    citys = np.array([])

    def __init__(self, pos_list, input_list):
        global city_dist_mat
        city_dist_mat = np.array(input_list)  # 确保 city_dist_mat 是 NumPy 数组
        self.best = None
        self.individual_list = []  # 某一代的个体列表
        self.result_list = []  # 所有代最优解的个体
        self.fitness_list = []  # 所有代的最优解的适应度
        dis_mat = city_dist_mat.tolist()  # 将矩阵转换为可迭代的列表形式以便于贪心算法处理
        self.individual_list = [Individual(greedy_path) for
                                greedy_path in self.greedy_init
                                (dis_mat)]
        self.citys = pos_list
        # self.dw.bound_x = [np.min(self.citys[:, 0]), np.max(self.citys[:, 0])]  # 计算绘图时的X界
        # self.dw.bound_y = [np.min(self.citys[:, 1]), np.max(self.citys[:, 1])]  # 计算绘图时的Y界
        # self.dw.set_xybound(self.dw.bound_x, self.dw.bound_y)  # 设置边界

    @staticmethod
    def random_init(dis_mat):
        result = []
        for i in range(individual_num):
            result_one = [i for i in range(gene_len)]
            random.shuffle(result_one)
            result.append(result_one)
        return result

    @staticmethod
    def greedy_init(dis_mat):
        start_index = 0
        result = []
        for i in range(individual_num):
            rest = [x for x in range(0, gene_len)]
            # 所有起始点都已经生成了
            if start_index >= gene_len:
                # start_index = np.random.randint(0, gene_len)
                # result.append(result[start_index].copy())
                random_result = [i for i in range(gene_len)]
                random.shuffle(random_result)
                result.append(random_result.copy())
                continue
            current = start_index
            rest.remove(current)
            # 找到一条最近邻路径
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x

                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
            # start_index = (start_index + 1) % gene_len   改进循环起点的处理
        return result

    def select(self):
        elite_num = int(individual_num * config.elite_ratio)
        # elite_individuals = sorted(self.individual_list, key=lambda x: x.fitness, reverse=True)[:elite_num]
        elite_individuals = sorted(self.individual_list, key=lambda x: x.fitness, reverse=True)[:elite_num]  # 适应度倒数
        non_elite_individuals = [indiv for indiv in self.individual_list if indiv not in elite_individuals]

        group_num = 10
        group_size = 20
        # 每次遗传保存individual_num个个体，每组胜出group_winner个个体
        group_winner = (individual_num - elite_num) // group_num

        # 取出锦标赛每组获胜的个体作为选择的个体
        winners = []
        # 锦标赛算法，先分若干组，再从每组中选精英
        for i in range(group_num):
            group = []
            # 随机组成小组，（可能重复）
            for j in range(group_size):
                # 先随机选一个
                selected = random.choice(non_elite_individuals)
                player = Individual(selected.genes)
                group.append(player)
            # 从 group 列表中截取前 group_winner 个元素
            group = Ga.rank(group)
            winners += group[:group_winner]
            # group2 = sorted(group, key=lambda x: x.fitness)[:group_winner]
            # winners += group2

        # self.individual_list = elite_individuals + winners
        # while len(self.individual_list) < individual_num:
        #     # 从父代中随机选择一个个体补充到新一代中
        #     additional_individual = random.choice(self.individual_list)
        #     self.individual_list.append(additional_individual)

        temp_individual = elite_individuals + winners
        while len(temp_individual) < individual_num:
            # 从父代中随机选择一个个体补充到新一代中
            additional_individual = random.choice(self.individual_list)
            temp_individual.append(additional_individual)
            # print('选择的个体数：\n', len(self.individual_list))
        self.individual_list = temp_individual.copy()

    def cross(self):

        # print("------")
        # print("qian")
        # qian = sorted(self.individual_list, key=lambda x: x.fitness, reverse = True)[:10].copy()
        # for i in range(len(qian)):
        #     print(f"no{i}:{qian[i].fitness}")

        new_gen = []
        # 对后cross_prob比例的个体进行交叉
        cross_start = int(math.ceil((1 - cross_prob) * len(self.individual_list)))  # 计算交叉起始位置
        # 前1-cross_prob比例的个体直接加入结果
        new_gen.extend(self.individual_list[:cross_start])
        # 将后cross_prob比例的个体存入临时变量并随机排序
        temp_list = self.individual_list[cross_start:].copy()
        if len(temp_list) % 2 == 1:
            new_gen.append(Individual(copy_list(temp_list[0].genes)))
            temp_list.pop(0)
        # 随机 打乱
        random.shuffle(temp_list)
        for i in range(0, len(temp_list)-1, 2):
            genes1 = copy_list(temp_list[i].genes)
            genes2 = copy_list(temp_list[i + 1].genes)
            index1 = random.randint(0, gene_len - 2)  # 相当于在交叉过程中选取基因片段
            index2 = random.randint(index1, gene_len - 1)  # 随机生成两个index，保证index2>index1
            pos1_recorder = {value: idx for idx, value in enumerate(genes1)}  # 记录初始基因片段对应的位置
            pos2_recorder = {value: idx for idx, value in enumerate(genes2)}  # 解决交叉时产生的位置冲突问题
            # 交叉
            # 如果g2准备给g1的基因中存在冲突，那就把g1当前交叉位置的基因与g1中产生冲突的基因交换位置
            for j in range(index1, index2):
                value1, value2 = genes1[j], genes2[j]
                pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
                genes1[j], genes1[pos1] = genes1[pos1], genes1[j]
                genes2[j], genes2[pos2] = genes2[pos2], genes2[j]
                pos1_recorder[value1], pos1_recorder[value2] = pos1, j
                pos2_recorder[value1], pos2_recorder[value2] = j, pos2
            new_gen.append(Individual(genes1))  # 交叉之后生成两个新个体，放到new_gen中
            new_gen.append(Individual(genes2))
        self.individual_list = new_gen.copy()
        new_gen.clear()

        # print("hou:")
        # hou = sorted(self.individual_list, key=lambda x: x.fitness, reverse=True)[:10].copy()
        # for i in range(len(hou)):
        #     print(f"no{i}:{hou[i].fitness}")

        return

    def mutate(self):
        # 仅对前mutate_prob的最差个体进行变异
        sorted(self.individual_list, key=lambda x: x.fitness)
        mutate_split = int(math.ceil(mutate_prob * len(self.individual_list)))

        for individual in self.individual_list[:mutate_split]:
            # 翻转切片法进行变异
            old_gen = copy_list(individual.genes)
            index1 = random.randint(0, gene_len - 2)
            index2 = random.randint(index1, gene_len - 1)
            mutate_genes = old_gen[index1:index2]
            mutate_genes.reverse()
            temp_gen = old_gen[:index1] + mutate_genes + old_gen[index2:]
            temp_individual = Individual(temp_gen)
            # 变异后适应度低的个体保留原来的
            if temp_individual.fitness > individual.fitness:  # 适应度倒数
                individual.genes = temp_gen
                individual.fitness = temp_individual.fitness
        return

    @staticmethod
    def rank(group):
        # 冒泡排序 升序排列
        for i in range(1, len(group)):
            for j in range(0, len(group) - i):
                if group[j].fitness < group[j + 1].fitness:  # 适应度倒数
                    group[j], group[j + 1] = group[j + 1], group[j]
        return group

    # 产生下一代的过程：选择、交叉、变异
    def next_gen(self):
        # 选择
        # print('选择前的个体数：\n', len(self.individual_list))
        # print("初始化：\n")
        # for x in self.individual_list:
        #     print(x.genes)
        self.select()
        # print('选择前的个体数：\n', len(self.individual_list))
        # print("选择后：\n")
        # for x in self.individual_list:
        #     print(x.genes)
        # 交叉
        self.cross()
        # 对交叉后的个体变异
        # print('变异前的个体数：\n', len(self.individual_list))
        # print("交叉后：\n")
        # for x in self.individual_list:
        #     print(x.genes)
        self.mutate()
        # print('变异后的个体数：\n', len(self.individual_list))
        # print("变异后：\n")
        # for x in self.individual_list:
        #     print(x.genes)

        # 适应度分析
        for individual in self.individual_list:
            if individual.fitness > self.best.fitness:  # 适应度倒数
                self.best = individual

        # 绘图
        # self.dw.ax.cla()
        # self.re_draw()
        # self.dw.plt.pause(0.001)

    # Adaptive adjustment mechanism
    def adaptive_adjustment(self, gen):
        if gen <= 1.0 * gen_num / 4.0:
            return
        else:
            if gen <= 3.0 * gen_num / 4.0:
                config.cross_prob = 0.95
                config.mutate_prob = 0.3
            else:
                config.cross_prob = 0.90
                config.mutate_prob = 0.3

    def train(self, pos_list):
        # 初代种群，在Individual中使用随即顺序构造
        # self.individual_list = [Individual() for _ in range(individual_num)]
        # 初始先随即设置一个最适应的个体
        self.best = self.individual_list[0]
        # 迭代，共gen_num次
        for i in range(gen_num):
            self.adaptive_adjustment(i)
            self.next_gen()
            result = copy_list(self.best.genes)
            result.append(result[0])
            # 画图
            # if i % 10 == 0:
            #     self.draw_current(result, pos_list, i+1)
            self.result_list.append(result)
            self.fitness_list.append(1000.0 / max(self.best.fitness, 1e-3))  # 适应度倒数

        # self.re_draw()
        # self.dw.plt.show()
        return self.result_list, self.fitness_list

    def draw_current(self, best_res, pos_list, cnt):
        temp_pos_list = pos_list[best_res, :]
        plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体解决中文显示问题
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        # 创建绘图窗口
        fig = plt.figure()
        # 使用实心圆点，红色直线，在散点图中表示最优解的城市坐标集合
        plt.plot(temp_pos_list[:, 0], temp_pos_list[:, 1], 'o-r')
        global draw_cnt
        plt.title(f"最优解{draw_cnt}路线")
        plt.legend(["My legend"], fontsize="x-large")
        output_folder = "./video/pic"
        output_path = os.path.join(output_folder, f"{draw_cnt}.png")  # 假设保存为PNG格式
        draw_cnt += 1
        plt.savefig(output_path, dpi=300)  # dpi可调整图像分辨率，这里设置为300
        plt.show()
