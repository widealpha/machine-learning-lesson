import random
import time


# 生成0/1背包数据集
def generate_knapsack_data(num_items, max_weight, max_value):
    weights = [random.randint(1, max_weight) for _ in range(num_items)]
    values = [random.randint(1, max_value) for _ in range(num_items)]
    # # 计算单位价值并组合为元组 (weight, value, value_per_weight)
    # data = list(zip(weights, values, [v / w for v, w in zip(values, weights)]))
    # # 按照单位价值从大到小排序
    # data.sort(key=lambda x: x[2], reverse=True)
    # # 解包排序后的数据
    # weights, values, _ = zip(*data)
    capacity = sum(weights) // 2  # 背包容量设定为总重量的一半
    return weights, values, capacity


# 初始化种群
def initialize_population(N, chromosome_length):
    population = []
    for _ in range(N):
        chromosome = [random.randint(0, 1) for _ in range(chromosome_length)]
        population.append(chromosome)
    return population


# 计算适应性函数（背包价值）
def fitness(chromosome, weights, values, capacity):
    total_weight = sum(chromosome[i] * weights[i] for i in range(len(chromosome)))
    total_value = sum(chromosome[i] * values[i] for i in range(len(chromosome)))
    if total_weight <= capacity:
        return total_value
    else:
        return 0  # 超过背包容量，价值为0


# 选择染色体
def select_chromosomes(population, probabilities):
    selected_indices = random.choices(range(len(population)), weights=probabilities, k=len(population))
    selected_chromosomes = [population[i] for i in selected_indices]
    return selected_chromosomes


# 交叉操作
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    else:
        return parent1, parent2


# 变异操作
def mutate(chromosome, mutation_rate):
    mutated_chromosome = chromosome.copy()
    for i in range(len(mutated_chromosome)):
        if random.random() < mutation_rate:
            mutated_chromosome[i] = 1 - mutated_chromosome[i]
    return mutated_chromosome


# 主算法
def genetic_algorithm_knapsack(N, chromosome_length, generations, crossover_rate, mutation_rate, weights, values,
                               capacity):
    # 初始化种群
    population = initialize_population(N, chromosome_length)

    for t in range(generations):
        # 计算适应性函数
        fitness_values = [fitness(chromosome, weights, values, capacity) for chromosome in population]

        # 选择染色体
        total_fitness = sum(fitness_values)
        probabilities = [fitness_value / total_fitness + 0.00001 for fitness_value in fitness_values]
        new_population = select_chromosomes(population, probabilities)

        # 交叉操作
        cross_population = []
        for i in range(0, N, 2):
            parent1 = new_population[i]
            parent2 = new_population[i + 1]
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            cross_population.extend([child1, child2])

        # 变异操作
        mut_population = [mutate(chromosome, mutation_rate) for chromosome in cross_population]

        # 更新种群
        population = mut_population

    # 找到最优解
    best_chromosome = max(population, key=lambda x: fitness(x, weights, values, capacity))
    best_fitness = fitness(best_chromosome, weights, values, capacity)

    return best_chromosome, best_fitness


def dynamic_programming_knapsack(weights, values, capacity):
    n = len(weights)
    # 创建一个表格用于存储子问题的解决方案
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # 从底向上构建DP表格
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # 如果当前物品的重量大于剩余容量，跳过该物品
            if weights[i - 1] > w:
                dp[i][w] = dp[i - 1][w]
            else:
                # 选择包括或排除当前物品的最大价值
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])

    # 回溯过程，获取解向量
    selected_items = [0] * n
    i, w = n, capacity
    while i > 0 and w > 0:
        if dp[i][w] != dp[i - 1][w]:
            # 当前物品被选择
            selected_items[i - 1] = 1
            w -= weights[i - 1]
        i -= 1

    # 返回最大价值和解向量
    return dp[n][capacity], selected_items


def main():
    num_items = 100  # 物品数量
    max_weight = 100  # 最大重量范围
    max_value = 10  # 最大价值范围
    N = 100  # 种群大小
    chromosome_length = num_items  # 染色体长度等于物品数量
    generations = 1000  # 迭代次数
    crossover_rate = 0.8  # 交叉概率
    mutation_rate = 0.1  # 变异概率

    weights, values, capacity = generate_knapsack_data(num_items, max_weight, max_value)
    print(f"物品数量:{num_items},""背包容量:", capacity, "最大重量:" ,max_weight, )
    print("物品重量:", weights)
    print("物品价值:", values)
    print()

    start_time = time.time()
    optimal_value, selected_items = dynamic_programming_knapsack(weights, values, capacity)
    # 记录结束时间
    end_time = time.time()
    # 计算执行时间
    execution_time = end_time - start_time
    print(f"代码执行时间：{execution_time * 1000}ms")

    weight = sum(weights[i] * selected_items[i] for i in range(len(weights)))
    print("动态规划最优值:", optimal_value)
    print("动态规划总重量:", weight)
    print("动态规划选中的物品索引:", selected_items)
    print()

    start_time = time.time()
    best_solution, best_fitness = genetic_algorithm_knapsack(N, chromosome_length, generations, crossover_rate,
                                                             mutation_rate, weights, values, capacity)
    # 记录结束时间
    end_time = time.time()
    # 计算执行时间
    execution_time = end_time - start_time
    print(f"代码执行时间：{execution_time * 1000}ms")

    weight = sum(weights[i] * best_solution[i] for i in range(len(weights)))
    print("遗传算法总价值:", best_fitness)
    print("遗传算法总重量:", weight)
    print("遗传算法选中的物品索引:", best_solution)
    print()
    print("与最优比率:", 1.0 * best_fitness / optimal_value)


if __name__ == '__main__':
    main()
