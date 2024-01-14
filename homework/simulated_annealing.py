import math
import random
import time


def generate_random_cities(num_cities, x_range=(0, 10), y_range=(0, 10)):
    cities = {}
    for i in range(num_cities):
        city_name = chr(ord('A') + i)
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        cities[city_name] = (x, y)
    return cities


# 计算两个城市之间的距离
def distance(city1, city2):
    x1, y1 = cities[city1]
    x2, y2 = cities[city2]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def stop_condition(num_iterations):
    return num_iterations > 1000


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
    num_iterations = 0
    while not inner_loop_stop_condition(current_temperature):
        # 从邻域中随机选择一个新解
        neighbor_path = generate_neighbor_path(current_path)

        # 计算目标函数差异
        delta_distance = total_distance(neighbor_path) - total_distance(current_path)

        # 判断是否接受新解
        if delta_distance <= 0 or math.exp(-delta_distance / current_temperature) > random.random():
            current_path = neighbor_path

        # 更新温度
        num_iterations += 1
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
    num_cities = 5
    initial_temperature = 1000.0
    cities = generate_random_cities(num_cities=5)
    initial_path = list(cities.keys())
    # 输出TSP图
    print("城市坐标:", cities)
    print("初始路径:", initial_path)
    # 计算城市间距离
    print("城市间距离:")
    for city1 in cities:
        for city2 in cities:
            print(f"{city1} -> {city2}: {distance(city1, city2):.2f}", end=' ; ')
    print()
    start_time = time.time()
    # 运行模拟退火算法
    final_path = simulated_annealing_tsp(initial_path, initial_temperature, temperature_decay,
                                         inner_loop_stop_condition)
    end_time = time.time()
    execution_time = end_time - start_time
    # 输出结果
    print(f"代码执行时间：{execution_time * 1000}ms")
    print("最终路径:", final_path)
    print("总距离:", total_distance(final_path))
