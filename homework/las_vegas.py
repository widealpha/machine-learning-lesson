import random

import numpy as np


def generate_board(board_size, num_obstacles):
    # 生成指定大小的线路板，并在其中随机放置一定数量的障碍物
    board = np.zeros(board_size, dtype=int)
    obstacles = set()

    for _ in range(num_obstacles):
        obstacle = (random.randint(0, board_size[0] - 1), random.randint(0, board_size[1] - 1))
        while obstacle in obstacles:
            obstacle = (random.randint(0, board_size[0] - 1), random.randint(0, board_size[1] - 1))
        obstacles.add(obstacle)
        board[obstacle] = 1  # 1 表示障碍物

    return board, obstacles


class LineRoutingProblem:
    def __init__(self, board_size, obstacles):
        self.board_size = board_size
        self.obstacles = obstacles
        self.solution = None
        self.min_cost = float('inf')

    def is_valid_move(self, x, y):
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1] and (x, y) not in self.obstacles

    def random_valid_move(self, current_pos):
        # 随机选择有效移动
        moves = [(1, 0), (0, 1)]  # 只考虑向右和向下移动
        valid_moves = [move for move in moves if self.is_valid_move(current_pos[0] + move[0], current_pos[1] + move[1])]
        if valid_moves:
            return random.choice(valid_moves)
        return None

    def branch_and_bound(self, path, cost, current_pos):
        if current_pos == (self.board_size[0] - 1, self.board_size[1] - 1):
            if cost < self.min_cost:
                self.min_cost = cost
                self.solution = path[:]
            return

        # 改进为拉斯维加斯算法，随机选择下一步移动
        move = self.random_valid_move(current_pos)
        if move:
            next_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
            new_path = path + [next_pos]
            new_cost = cost + 1
            self.branch_and_bound(new_path, new_cost, next_pos)

    def solve(self):
        # 从起点开始搜索
        self.branch_and_bound([(0, 0)], 0, (0, 0))
        return self.solution, self.min_cost


if __name__ == '__main__':
    board_size = (10, 10)
    num_obstacles = 10
    board, obstacles = generate_board(board_size, num_obstacles)

    print("生成的线路板:")
    print(board)

    problem = LineRoutingProblem(board_size, obstacles)
    solution, min_cost = problem.solve()

    print("\n最优线路:", solution)
    print("最小成本:", min_cost)
