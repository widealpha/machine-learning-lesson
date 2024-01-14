import queue
import random
import time


class Position:
    def __init__(self, row, col):
        self.row = row
        self.col = col


def find_path(start, finish, m, n, grid):
    # 找到最短布线路径，则返回真，否则返回假
    if start.row == finish.row and start.col == finish.col:
        return True, 0, []

    # 设置方向移动坐标值：东、南、西、北
    offset = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # 相邻的方格数
    num_neigh_blo = 4

    # 设置当前方格，即搜索单位
    here = start

    # 由于0和1用于表示方格的开放和封锁，故距离：2-0 3-1
    grid[start.row][start.col] = 0

    # 队列式搜索，标记可达相邻方格
    q_find_path = queue.Queue()

    while True:
        num = 0  # 方格未标记个数
        select_position = []  # 选择位置保存

        for i in range(num_neigh_blo):
            # 达到四个方向
            nbr = Position(here.row + offset[i][0], here.col + offset[i][1])
            if grid[nbr.row][nbr.col] == -1:
                # 该方格未标记
                grid[nbr.row][nbr.col] = grid[here.row][here.col] + 1
                if nbr.row == finish.row and nbr.col == finish.col:
                    break

                select_position.append(nbr)
                num += 1

        if num > 0:  # 如果标记，则在这么多个未标记个数中随机选择一个位置
            # 随机选一个入队
            q_find_path.put(select_position[random.randint(0, num - 1)])

        # 是否到达目标位置finish
        if nbr.row == finish.row and nbr.col == finish.col:
            break

        # 活结点队列是否为空
        if q_find_path.empty():
            return False, 0, []

        # 访问对首元素出队
        here = q_find_path.get()

    # 构造最短布线路径
    path_len = grid[finish.row][finish.col]
    path = [Position(0, 0) for _ in range(path_len)]

    # 从目标位置finish开始向起始位置回溯
    for j in range(path_len - 1, -1, -1):
        path[j] = here
        # 找前驱位置
        for i in range(num_neigh_blo):
            nbr = Position(here.row + offset[i][0], here.col + offset[i][1])
            if grid[nbr.row][nbr.col] == j:  # 距离加2正好是前驱位置
                break
        here = nbr

    return True, path_len, path


if __name__ == "__main__":
    print("---------Las Vegas 分支限界布线问题--------")
    m, n = map(int, input("在一个m*n的棋盘上，请分别输入m和n，代表行数和列数，然后输入回车\n").split())

    # 创建棋盘格
    grid = [[-1 for _ in range(n + 2)] for _ in range(m + 2)]

    # 初始化棋盘格
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            grid[i][j] = -1

    # 设置方格阵列的围墙
    for i in range(n + 2):
        grid[0][i] = grid[m + 1][i] = -2  # 上下的围墙

    for i in range(m + 2):
        grid[i][0] = grid[i][n + 1] = -2  # 左右的围墙

    print("初始化棋盘格和加围墙")
    print("-------------------------------")
    for row in grid:
        print(" ".join(map(str, row)))
    print("-------------------------------")

    print("请输入已经占据的位置，行坐标 列坐标，代表此位置不能布线")
    print("例如输入 2 2，表示坐标 2 2 不能布线;当输入的坐标为 0 0，表示结束输入")

    # 添加已经布线的棋盘格
    while True:
        ci, cj = map(int, input().split())
        if ci > m or cj > n:
            print("输入非法！！！！！行坐标 < ", m, " ,列坐标 < ", n, " 当输入的坐标为  0,0，结束输入")
            continue
        elif ci == 0 and cj == 0:
            break
        else:
            grid[ci][cj] = -3

    # 布线前的棋盘格
    print("布线前的棋盘格")
    print("-------------------------------")
    for row in grid:
        print(" ".join(map(str, row)))
    print("-------------------------------")

    start_row, start_col = map(int, input("请输入起点位置坐标\n").split())
    finish_row, finish_col = map(int, input("请输入终点位置坐标\n").split())

    start = Position(start_row, start_col)
    finish = Position(finish_row, finish_col)

    start_time = time.time()
    result, path_len, path = find_path(start, finish, m, n, grid)
    end_time = time.time()

    if result:
        print("-------------------------------")
        print("$ 代表围墙")
        print("# 代表已经占据的点")
        print("* 代表布线路线")
        print("= 代表还没有布线的点")
        print("-------------------------------")

        for i in range(m + 2):
            for j in range(n + 2):
                if grid[i][j] == -2:
                    print("$ ", end="")
                elif grid[i][j] == -3:
                    print("# ", end="")
                else:
                    for pos in path:
                        if i == pos.row and j == pos.col:
                            print("* ", end="")
                            break
                    else:
                        if i == start.row and j == start.col:
                            print("* ", end="")
                        else:
                            print("= ", end="")
            print()

        print("-------------------------------")
        print("路径坐标和长度")
        print()
        print("({}, {}) ".format(start.row, start.col), end="")
        for pos in path:
            print("({}, {}) ".format(pos.row, pos.col), end="")
        print()
        print()
        print("路径长度：", path_len + 1)
        print()
        print("布线完毕，查找", time, "次")
        print("运行时间: {:.2f} ms".format((end_time - start_time) * 1000))
    else:
        print()
        print("经过多次尝试，仍然没有找到路线")
