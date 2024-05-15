from tqdm import tqdm
import numpy as np
import time
import csv
import os


# 节点类
class Node:
    def __init__(self, point):
        self.point = np.array(point)
        self.parent = None
        self.cost = 0


class RRTStar3D:
    def __init__(self, st, gl, ot, rand_area, step_size, max_iter, search_radius, safe_distance, k_att, k_rep, use_apf):
        self.start = Node(st)  # 起始点
        self.goal = Node(gl)  # 目标点
        self.obstacle_list = ot  # 障碍物列表
        self.min_rand = rand_area[0]  # 随机生成点的范围最小值
        self.max_rand = rand_area[1]  # 随机生成点的范围最大值
        self.origin_step_size = step_size
        self.init_step_size = step_size  # 初始步长
        self.step_size = 0
        self.max_iter = max_iter  # 最大迭代次数
        self.search_radius = search_radius  # 搜索半径
        self.safe_distance = safe_distance  # 安全距离
        self.k_att = k_att  # 引力系数
        self.k_rep = k_rep  # 斥力系数

        self.node_list = []  # 记录所有探索到的点
        self.distance_history = []  # 记录距离目标的历史列表
        self.distance_to_goal = 999  # 记录当前点到目标点的距离
        self.collision_num = 0  # 记录碰撞次数
        self.use_apf = use_apf

    def generate_random_point(self):
        # 在随机生成点的范围内生成随机点
        point = np.random.uniform(low=self.min_rand, high=self.max_rand, size=3)
        return point

    def nearest_node(self, point):
        # 找到距离指定点最近的节点的索引
        distances = [np.linalg.norm(np.array(node.point) - np.array(point)) for node in self.node_list]
        return np.argmin(distances)

    def collision_detect(self, point1, point2):
        for obstacle in self.obstacle_list:
            obstacle_center = np.array(obstacle[0])
            obstacle_radius = obstacle[1]
            link1 = point2 - point1
            link2 = obstacle_center - point1
            link3 = obstacle_center - point2
            judge = np.dot(link1, link2) / np.dot(link1, link1)
            if 0 <= judge <= 1:  # 垂足在线段上
                distance = np.linalg.norm(np.cross(link1, link2)) / np.linalg.norm(link1)
            else:  # 垂足不在线段上
                distance = min(np.linalg.norm(link2), np.linalg.norm(link3))
            if distance < obstacle_radius + self.safe_distance:
                return False
        return True

    # 根据当前探索方向动态调整步长
    def adjust_step_size(self, from_node, to_node):
        # 计算两个向量
        vec1 = np.array(to_node.point) - np.array(from_node.point)
        vec2 = np.array(self.goal.point) - np.array(from_node.point)
        # 计算两个向量的点积
        dot_product = np.dot(vec1, vec2)
        # 计算两个向量的模
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        # 计算夹角（弧度）
        cos_theta = dot_product / (norm_vec1 * norm_vec2)
        theta_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        # 将弧度转换为角度
        theta_degrees = np.degrees(theta_radians)
        # 调整步长
        if theta_degrees < 90:
            self.step_size = self.init_step_size * (1 + dot_product / (norm_vec1 * norm_vec2))
        else:
            self.step_size = self.init_step_size
        return vec1, norm_vec1

    def apf(self, from_node, to_node):
        # 计算引力
        att_force = self.k_att * (self.goal.point - from_node.point)
        # 计算斥力
        rep_force = np.zeros(3)
        for obstacle in self.obstacle_list:
            obstacle_center = np.array(obstacle[0])
            direction = from_node.point - obstacle_center
            direction_norm = np.linalg.norm(direction)
            rep_force += self.k_rep * (1 / direction_norm ** 2) * direction
        # 计算合力
        total_force = att_force + rep_force
        to_node.point += total_force.astype(np.int64)

    def steer(self, from_node, to_node):
        # 根据引力和斥力调整to_node
        if self.use_apf:
            self.apf(from_node, to_node)
        # 调整步长
        direction, distance = self.adjust_step_size(from_node, to_node)
        unit_direction = direction / distance
        new_node_point = from_node.point + self.step_size * unit_direction
        # 两节点间碰撞检测
        if self.collision_detect(from_node.point, new_node_point):
            new_node = Node(new_node_point)
            new_node.parent = from_node
            # 记录距离历史
            self.distance_to_goal = np.linalg.norm(new_node.point - self.goal.point)

            return new_node
        else:
            self.collision_num += 1
            return False

    def near_nodes(self, node):
        # 找到在搜索半径内的所有节点的索引'
        distances = [np.linalg.norm(np.array(node.point) - np.array(other_node.point)) for other_node in self.node_list]
        near_nodes_idx = [idx for idx, distance in enumerate(distances) if self.search_radius >= distance > 0]
        return near_nodes_idx

    def choose_parent(self, new_node, near_nodes):
        # 选择新节点的父节点，使得到达新节点的路径代价最小
        costs = []
        for idx in near_nodes:
            near_node = self.node_list[idx]
            if self.collision_detect(near_node.point, new_node.point):
                cost = near_node.cost + np.linalg.norm(np.array(new_node.point) - np.array(near_node.point))
                costs.append(cost)
            else:
                costs.append(float('inf'))
        all_inf = all(cost == float('inf') for cost in costs)
        # 判断搜索半径内是否没有可用节点
        if not all_inf:
            min_cost_idx = near_nodes[np.argmin(costs)]
            new_node.cost = self.node_list[min_cost_idx].cost + np.linalg.norm(
                np.array(new_node.point) - np.array(self.node_list[min_cost_idx].point))
            new_node.parent = self.node_list[min_cost_idx]
        else:
            print("choose_parent failed ")

    def rewire(self, new_node, near_nodes):
        # 重新连接节点，使得新节点附近节点有更好的父节点
        for idx in near_nodes:
            near_node = self.node_list[idx]
            if self.collision_detect(new_node.point, near_node.point):
                new_cost = new_node.cost + np.linalg.norm(np.array(new_node.point) - np.array(near_node.point))
                if near_node.cost > new_cost:
                    near_node.parent = new_node
                    near_node.cost = new_cost

    def plan(self):
        start_time = time.time()
        self.node_list.append(self.start)
        for iteration in range(self.max_iter):
            # 生成节点
            if np.random.rand() < 0.1:
                random_point = self.goal.point
            else:
                random_point = self.generate_random_point()
            nearest_node_idx = self.nearest_node(random_point)
            nearest_node = self.node_list[nearest_node_idx]
            # 连接节点（过程中需要动态调整步长）
            new_node = self.steer(nearest_node, Node(random_point))
            if not new_node:
                continue
            # 重新选择父节点
            near_nodes_idx = self.near_nodes(new_node)
            self.choose_parent(new_node, near_nodes_idx)
            # 添加新节点
            self.node_list.append(new_node)
            # 重布线
            self.rewire(new_node, near_nodes_idx)

            # 判断是否接近终点
            if self.distance_to_goal < self.step_size:
                if self.collision_detect(new_node.point, self.goal.point):
                    self.goal.parent = new_node
                    self.goal.cost = new_node.cost + self.distance_to_goal

                    end_time = time.time()
                    time_taken = end_time - start_time
                    cost = self.goal.cost
                    node_num = len(self.node_list)
                    collision_num = self.collision_num
                    return time_taken, cost, node_num, collision_num, True
        return None, None, None, None, False


def test_rrt_star(num_tests, use_apf):
    start = [1, 1, 1]
    goal = [13, 13, 13]
    obstacle_list = [([6, 6, 6], 1.5), ([9, 10, 9], 1.5), ([10, 6, 10], 1.5), ([8, 6, 8], 1), ([6, 10, 6], 1.5),
                     ([12, 12, 12], 0.5)]
    rand_area = [0, 15]
    step_size = 0.4
    max_iter = 700
    search_radius = 3.5
    safe_distance = 0.2
    k_att = 0.3
    k_rep = 1.4

    results = []

    desc = "Running tests of using apf" if use_apf else "Running tests of not using apf"
    for _ in tqdm(range(num_tests), desc=desc):
        rrt_star = RRTStar3D(start, goal, obstacle_list, rand_area, step_size, max_iter, search_radius, safe_distance,
                             k_att, k_rep,
                             use_apf)
        time_taken, cost, node_num, collision_num, found_path = rrt_star.plan()
        results.append((time_taken, cost, node_num, collision_num, found_path))
    return results


def save_results_to_csv(results, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["测试次数", "总耗时(s)", "平均耗时(s)", "平均路径长度（dm）", "平均探索节点数",
                         "平均连接新节点时发生碰撞次数", "成功率"])
        total_time = 0
        total_cost = 0
        total_node_num = 0
        total_collision_num = 0
        success_count = 0
        for time_taken, cost, node_num, collision_num, found_path in results:
            if time_taken is not None:
                total_time += time_taken
                total_cost += cost
                total_node_num += node_num
                total_collision_num += collision_num
                if found_path:
                    success_count += 1
        writer.writerow([len(results), total_time, total_time / success_count, total_cost / success_count,
                         total_node_num / success_count, total_collision_num / success_count,
                         success_count / len(results)])


if __name__ == '__main__':
    num_tests = 1000
    if not os.path.exists('res'):
        os.makedirs('res')
    results_with_apf = test_rrt_star(num_tests, True)
    save_results_to_csv(results_with_apf, "res/results_with_apf.csv")

    results_without_apf = test_rrt_star(num_tests, False)
    save_results_to_csv(results_without_apf, "res/results_without_apf.csv")
