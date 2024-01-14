import math
import random

# 城市坐标数据，格式为 (x, y)
cities = {
    'A': (0, 0),
    'B': (1, 2),
    'C': (3, 1),
    'D': (5, 2),
    'E': (6, 0)
}


# 计算两个城市之间的距离
def distance(city1, city2):
    x1, y1 = cities[city1]
    x2, y2 = cities[city2]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# 计算路径长度
def total_distance(path):
    total = 0
    for i in range(len(path) - 1):
        total += distance(path[i], path[i + 1])
    total += distance(path[-1], path[0])  # 回到起始城市
    return total


# 模拟退火算法求解TSP
def simulated_annealing_tsp(initial_path, initial_temperature, temperature_decay, inner_loop_stop_condition):
    current_path = initial_path
    current_temperature = initial_temperature

    while not inner_loop_stop_condition(current_temperature):
        # 从邻域中随机选择一个新解
        neighbor_path = generate_neighbor_path(current_path)

        # 计算目标函数差异
        delta_distance = total_distance(neighbor_path) - total_distance(current_path)

        # 判断是否接受新解
        if delta_distance <= 0 or math.exp(-delta_distance / current_temperature) > random.random():
            current_path = neighbor_path

        # 更新温度
        current_temperature = temperature_decay(current_temperature)

    return current_path


# 生成邻域中的随机解，交换两个城市的位置
def generate_neighbor_path(current_path):
    path = current_path.copy()
    idx1, idx2 = random.sample(range(len(path)), 2)
    path[idx1], path[idx2] = path[idx2], path[idx1]
    return path


# 温度衰减函数
def temperature_decay(current_temperature):
    return 0.9 * current_temperature


# 内循环停止条件函数，可以根据需要进行调整
def inner_loop_stop_condition(current_temperature):
    return current_temperature < 1e-5


if __name__ == "__main__":
    # 设置初始解、初始温度等参数
    initial_path = ['A', 'B', 'C', 'D', 'E', 'A']
    initial_temperature = 1000.0

    # 运行模拟退火算法
    final_path = simulated_annealing_tsp(initial_path, initial_temperature, temperature_decay,
                                         inner_loop_stop_condition)

    # 输出结果
    print("Final Path:", final_path)
    print("Total Distance:", total_distance(final_path))
