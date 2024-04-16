from moviepy.editor import *


class Video:

    def __init__(self, input_filename, output_filename, fps):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.fps = fps

    def create_video(self):
        images_folder = self.input_filename
        file_names = sorted([images_folder + "/" + f for f in os.listdir(images_folder) if
                             f.endswith('.png') or f.endswith('.jpg')])  # 按照时间戳或其他方式排序图片文件

        # 使用imageio读取图片序列并生成视频
        with imageio.get_writer(self.output_filename, format='mp4', fps=self.fps) as writer:
            for filename in file_names:
                image = imageio.imread(filename)
                writer.append_data(image)


# class Draw(object):
#     bound_x = []
#     bound_y = []
#
#     def __init__(self):
#         self.fig, self.ax = plt.subplots()
#         self.plt = plt
#         self.set_font()
#
#     def draw_line(self, p_from, p_to):
#         line1 = [(p_from[0], p_from[1]), (p_to[0], p_to[1])]
#         (line1_xs, line1_ys) = zip(*line1)
#         self.ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='blue'))
#
#     def draw_points(self, pointx, pointy):
#         self.ax.plot(pointx, pointy, 'ro')
#
#     def set_xybound(self, x_bd, y_bd):
#         self.ax.axis([x_bd[0], x_bd[1], y_bd[0], y_bd[1]])
#
#     def draw_text(self, x, y, text, size=8):
#         self.ax.text(x, y, text, fontsize=size)
#
#     def set_font(self, ft_style='SimHei'):
#         plt.rcParams['font.sans-serif'] = [ft_style]  # 用来正常显示中文标签
#
#     def draw_citys_way(self, gen):
#         """
#         根据一条基因gen绘制一条旅行路线
#         :param gen:
#         :return:
#         """
#         tsp = self
#         dw = self.dw
#         m = len(gen)
#         tsp.dw.set_xybound(tsp.dw.bound_x, tsp.dw.bound_y)
#         for i in range(m):
#             if i < m - 1:
#                 best_i = tsp.best.genes[i]
#                 next_best_i = tsp.best.genes[i + 1]
#                 best_icity = tsp.citys[best_i]
#                 next_best_icity = tsp.citys[next_best_i]
#                 dw.draw_line(best_icity, next_best_icity)
#         start = tsp.citys[tsp.best.genes[0]]
#         end = tsp.citys[tsp.best.genes[-1]]
#         dw.draw_line(end, start)
#
#     def re_draw(self):
#         # 重绘图；每次迭代后绘制一次，动态展示。
#         tsp = self
#         self.dw.ax.cla()
#         tsp.dw.draw_points(tsp.citys[:, 0], tsp.citys[:, 1])
#         tsp.draw_citys_way(self.best.genes)
#         global draw_cnt
#         print("第几次画图", draw_cnt)
#         draw_cnt += 1
