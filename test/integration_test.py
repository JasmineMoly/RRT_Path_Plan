import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy as np
import time


class Node:
    def __init__(self, point):
        self.point = np.array(point)
        self.parent = None
        self.cost = 0


class RRTStar3D:
    def __init__(self, st, gl, ot, rand_area, step_size, max_iter, search_radius, safe_distance):
        self.start = Node(st)  # 起始点
        self.goal = Node(gl)  # 目标点
        self.obstacle_list = ot  # 障碍物列表
        self.min_rand = rand_area[0]  # 随机生成点的范围最小值
        self.max_rand = rand_area[1]  # 随机生成点的范围最大值
        self.origin_step_size = step_size
        self.step_size = step_size  # 步长
        self.max_iter = max_iter  # 最大迭代次数
        self.search_radius = search_radius  # 搜索半径
        self.safe_distance = safe_distance  # 安全距离
        self.node_list = []
        self.step_size_history = []  # 记录步长变化的历史列表
        self.distance_history = []  # 记录距离目标的历史列表
        # 设置max_distance和min_distance的比例参数
        self.min_distance = 0
        self.max_distance = 0
        self.max_distance_ratio = 0.7  # max_distance的比例系数
        self.min_distance_ratio = 1.5  # min_distance的比例系数
        self.a = 0
        self.b = 0
        self.c = 0
        # 计算max_distance和min_distance
        self.compute_distances()

    def compute_distances(self):
        # 计算规划空间对角线长度
        diag_length = np.linalg.norm(np.array(self.goal.point) - np.array(self.start.point))

        # 计算max_distance和min_distance
        self.max_distance = self.max_distance_ratio * diag_length
        self.min_distance = self.min_distance_ratio * self.step_size
        print("max_distance:", self.max_distance, "min_distance:", self.min_distance)

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
    def test(self):
        from_node = self.goal
        to_node = from_node.parent
        print("Testing path:")
        while to_node is not None:
            print("from:", from_node.point, ",to:", to_node.point, ",collision free:", self.collision_detect(from_node.point, to_node.point))
            from_node = to_node
            to_node = to_node.parent


    def steer(self, from_node, to_node):
        # 从一个节点向另一个节点延伸一步
        direction = np.array(to_node.point) - np.array(from_node.point)
        distance = np.linalg.norm(direction)
        if distance <= self.step_size:
            new_node_point = to_node.point
        else:
            unit_direction = direction / distance
            new_node_point = from_node.point + self.step_size * unit_direction
        if self.collision_detect(from_node.point, new_node_point):
            new_node = Node(new_node_point)
            new_node.parent = from_node
            return new_node
        else:
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

    def trace_path(self):
        # 从目标节点回溯到起始节点，形成最终路径
        final_path = []
        current_node = self.goal
        while current_node is not None:
            final_path.append(current_node.point)
            current_node = current_node.parent
        return np.array(final_path[::-1])

    def adjust_step_size(self, new_node):
        # 根据距离目标的距离动态调整步长
        distance_to_goal = np.linalg.norm(new_node.point - self.goal.point)
        self.distance_history.append(distance_to_goal)  # 记录距离历史
        if distance_to_goal > self.max_distance:
            self.step_size = self.origin_step_size
            self.a += 1
        elif distance_to_goal < self.min_distance:
            self.step_size = self.origin_step_size * 0.2
            self.b += 1
        else:
            k = 0.8 * self.origin_step_size / (self.max_distance - self.min_distance)
            b = self.origin_step_size - k * self.max_distance
            self.step_size = k * distance_to_goal + b
            self.c += 1
        return distance_to_goal

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
            # 连接节点
            new_node = self.steer(nearest_node, Node(random_point))
            if not new_node:
                continue
            # 重新选择父节点
            near_nodes_idx = self.near_nodes(new_node)
            self.choose_parent(new_node, near_nodes_idx)
            self.node_list.append(new_node)
            # 重布线
            self.rewire(new_node, near_nodes_idx)
            # 动态调整步长
            distance_to_goal = self.adjust_step_size(new_node)
            self.step_size_history.append(self.step_size)  # 记录步长变化历史
            # 判断是否接近终点
            if distance_to_goal < self.min_distance:
                if self.collision_detect(new_node.point, self.goal.point):
                    self.goal.parent = new_node
                    self.goal.cost = new_node.cost + distance_to_goal
                    end_time = time.time()
                    self.test()
                    print("Time taken:", end_time - start_time, "seconds, Total cost:", self.goal.cost)
                    final_path = self.trace_path()
                    return final_path
        return None

    def draw_path(self, final_path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.start.point[0], self.start.point[1], self.start.point[2], c='green', marker='o', label='Start')
        ax.scatter(self.goal.point[0], self.goal.point[1], self.goal.point[2], c='red', marker='o', label='Goal')
        for obstacle in self.obstacle_list:
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x = obstacle[0][0] + obstacle[1] * np.cos(u) * np.sin(v)
            y = obstacle[0][1] + obstacle[1] * np.sin(u) * np.sin(v)
            z = obstacle[0][2] + obstacle[1] * np.cos(v)
            ax.plot_surface(x, y, z, color='y', alpha=0.5)
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 50)
            x = obstacle[0][0] + obstacle[1] * np.outer(np.cos(u), np.sin(v))
            y = obstacle[0][1] + obstacle[1] * np.outer(np.sin(u), np.sin(v))
            z = obstacle[0][2] + obstacle[1] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, rstride=4, cstride=4, color='y', linewidth=0, alpha=0.5)
        ax.plot(final_path[:, 0], final_path[:, 1], final_path[:, 2], c='blue', label='Path')
        ax.set_xlim(self.min_rand, self.max_rand)
        ax.set_ylim(self.min_rand, self.max_rand)
        ax.set_zlim(self.min_rand, self.max_rand)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    def bezier(self, final_path):
        x = final_path[:, 0]
        y = final_path[:, 1]
        z = final_path[:, 2]
        cs_x = CubicSpline(range(len(x)), x)
        cs_y = CubicSpline(range(len(y)), y)
        cs_z = CubicSpline(range(len(z)), z)
        t = np.linspace(0, len(x) - 1, 1000)
        x_new = cs_x(t)
        y_new = cs_y(t)
        z_new = cs_z(t)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='b', marker='o', label='Original points')
        ax.plot(x, y, z, c='y', label='Original path')
        ax.plot(x_new, y_new, z_new, label='Interpolated points', color='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend(loc='best')
        plt.show()

    def plot_step_size_history(self):
        plt.plot(range(len(self.step_size_history)), self.step_size_history)
        plt.xlabel('Iteration')
        plt.ylabel('Step Size')
        plt.title('Step Size Variation During Iterations')
        plt.show()

    def plot_distance_history(self):
        plt.plot(range(len(self.distance_history)), self.distance_history)
        plt.xlabel('Iteration')
        plt.ylabel('Distance to Goal')
        plt.title('Distance to Goal Variation During Iterations')
        plt.show()


if __name__ == '__main__':
    start = [1, 1, 1]
    goal = [13, 13, 13]
    obstacle_list = [([6, 6, 6], 1.5), ([9, 10, 9], 1.5), ([10, 6, 10], 1.5), ([8, 6, 8], 1), ([6, 10, 6], 1.5)]
    rrt_star = RRTStar3D(start, goal, obstacle_list, rand_area=[0, 15], step_size=0.5, max_iter=500, search_radius=3.8, safe_distance=0.3)
    path = rrt_star.plan()
    print("大于max_distance", rrt_star.a)
    print("小于min_distance", rrt_star.b)
    print("中间", rrt_star.c)
    if path is None:
        print("No valid path found!")
    else:
        print("Found!")
        rrt_star.draw_path(path)
        rrt_star.bezier(path)
        rrt_star.plot_step_size_history()
        rrt_star.plot_distance_history()
