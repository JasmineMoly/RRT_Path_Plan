import matplotlib.pyplot as plt
import numpy as np
import time


class Node:
    def __init__(self, point):
        self.point = np.array(point)  # 节点坐标
        self.parent = None  # 父节点指针


class RRTConnect:
    def __init__(self, start, goal, obstacle_list, rand_area, step_size, max_iter, safe_distance):
        self.start = Node(start)  # 起始状态节点
        self.goal = Node(goal)  # 目标状态节点
        self.obstacle_list = obstacle_list  # 障碍物列表
        self.min_rand = rand_area[0]  # 随机采样区域最小值
        self.max_rand = rand_area[1]  # 随机采样区域最大值
        self.step_size = step_size  # 步长
        self.max_iter = max_iter  # 最大迭代次数
        self.safe_distance = safe_distance  # 安全距离
        self.tree_start = [self.start]  # 起始树
        self.tree_goal = [self.goal]  # 目标树

    def generate_random_point(self):
        # 在随机采样区域生成随机点
        point = np.random.uniform(low=self.min_rand, high=self.max_rand, size=3)
        return point

    def nearest_node(self, point, tree):
        # 找到树中距离给定点最近的节点
        distances = [np.linalg.norm(np.array(node.point) - np.array(point)) for node in tree]
        return tree[np.argmin(distances)]

    # def steer(self, from_node, to_node):
    #     # 从一个节点向目标节点扩展一定距离
    #     direction = np.array(to_node.point) - np.array(from_node.point)
    #     distance = np.linalg.norm(direction)
    #     if distance <= self.step_size:
    #         new_node_point = to_node.point
    #     else:
    #         unit_direction = direction / distance
    #         new_node_point = from_node.point + self.step_size * unit_direction
    #     new_node = Node(new_node_point)
    #     new_node.parent = from_node
    #     return new_node
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

    def collision_free(self, point):
        # 检查给定点是否与障碍物碰撞
        for obstacle in self.obstacle_list:
            obstacle_center = np.array(obstacle[0])
            obstacle_radius = obstacle[1]
            distance = np.linalg.norm(obstacle_center - point)
            if distance <= obstacle_radius:
                return False
        return True

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

    def plan(self):
        start_time = time.time()  # 记录开始时间
        for _ in range(self.max_iter):
            random_point = self.generate_random_point()  # 生成随机点
            nearest_start_node = self.nearest_node(random_point, self.tree_start)  # 找到起始树中最近的节点

            new_start_node = self.steer(nearest_start_node, Node(random_point))  # 从最近节点向随机点扩展
            if not new_start_node:
                continue
            self.tree_start.append(new_start_node)  # 将新节点添加到起始树

            nearest_goal_node = self.nearest_node(new_start_node.point, self.tree_goal)  # 找到目标树中最近的节点
            new_goal_node = self.steer(nearest_goal_node, new_start_node)  # 从最近目标节点向新起始节点扩展
            if not new_goal_node:
                continue
            self.tree_goal.append(new_goal_node)  # 将新节点添加到目标树

            if np.linalg.norm(np.array(new_goal_node.point) - np.array(new_start_node.point)) <= self.step_size:
                # 如果新目标节点与新起始节点之间距离小于等于步长，则路径找到
                path_start = self.trace_path(new_start_node)
                path_goal = self.trace_path(new_goal_node)
                path_goal.reverse()
                end_time = time.time()  # 记录结束时间
                return path_start, path_goal, end_time - start_time  # 返回路径和计算时间
        end_time = time.time()  # 记录结束时间
        return None, None, end_time - start_time  # 返回空路径和计算时间

    def trace_path(self, node):
        # 从节点追溯路径
        path = [node.point]
        current_node = node
        while current_node.parent:
            path.append(current_node.parent.point)
            current_node = current_node.parent
        return path

    def draw_path(self, path_start, path_goal):
        # 绘制路径和障碍物
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
        path_start = np.array(path_start)
        ax.plot(path_start[:, 0], path_start[:, 1], path_start[:, 2], c='blue', label='Path from Start')
        path_goal = np.array(path_goal)
        ax.plot(path_goal[:, 0], path_goal[:, 1], path_goal[:, 2], c='purple', label='Path from Goal')
        ax.set_xlim(self.min_rand, self.max_rand)
        ax.set_ylim(self.min_rand, self.max_rand)
        ax.set_zlim(self.min_rand, self.max_rand)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()


if __name__ == '__main__':
    start = [1, 1, 1]
    goal = [13, 13, 13]
    obstacle_list = [([6, 6, 6], 1.5), ([9, 10, 9], 1.5), ([10, 6, 10], 1.5), ([8, 6, 8], 1), ([6, 10, 6], 1.5)]
    rrt_connect = RRTConnect(start, goal, obstacle_list, rand_area=[0, 15], step_size=0.6, max_iter=1000, safe_distance=0.3)
    path_start, path_goal, time_taken = rrt_connect.plan()

    if path_start is None or path_goal is None:
        print("No valid path found!")
    else:
        print("Found!")
        print("Time taken:", time_taken, "seconds")
        rrt_connect.draw_path(path_start, path_goal)
