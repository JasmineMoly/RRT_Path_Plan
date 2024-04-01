import csv
import os
import time
import numpy as np
from tqdm import tqdm


class Node:
    def __init__(self, point):
        self.point = np.array(point)
        self.parent = None
        self.cost = 0


class RRTStar3D:
    def __init__(self, st, gl, ot, rand_area, step_size, max_iter, search_radius):
        self.start = Node(st)
        self.goal = Node(gl)
        self.obstacle_list = ot
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.node_list = []

    def generate_random_point(self):
        point = np.random.uniform(low=self.min_rand, high=self.max_rand, size=3)
        return point

    def nearest_node(self, point):
        distances = [np.linalg.norm(np.array(node.point) - np.array(point)) for node in self.node_list]
        return np.argmin(distances)

    def steer(self, from_node, to_node):
        direction = np.array(to_node.point) - np.array(from_node.point)
        distance = np.linalg.norm(direction)
        if distance <= self.step_size:
            new_node_point = to_node.point
        else:
            unit_direction = direction / distance
            new_node_point = from_node.point + self.step_size * unit_direction
        new_node = Node(new_node_point)
        new_node.parent = from_node
        return new_node

    def collision_free(self, point):
        for obstacle in self.obstacle_list:
            obstacle_center = np.array(obstacle[0])
            obstacle_radius = obstacle[1]
            distance = np.linalg.norm(obstacle_center - point)
            if distance <= obstacle_radius:
                return False
        return True

    def near_nodes(self, node):
        distances = [np.linalg.norm(np.array(node.point) - np.array(other_node.point)) for other_node in self.node_list]
        near_nodes_idx = [idx for idx, distance in enumerate(distances) if self.search_radius >= distance > 0]
        return near_nodes_idx

    def choose_parent(self, new_node, near_nodes):
        if not near_nodes:
            return False
        costs = []
        for idx in near_nodes:
            near_node = self.node_list[idx]
            if self.collision_free(near_node.point) and np.linalg.norm(
                    np.array(new_node.point) - np.array(near_node.point)) <= self.step_size:
                cost = near_node.cost + np.linalg.norm(np.array(new_node.point) - np.array(near_node.point))
                costs.append(cost)
            else:
                costs.append(float('inf'))
        min_cost_idx = near_nodes[np.argmin(costs)]
        new_node.cost = self.node_list[min_cost_idx].cost + np.linalg.norm(
            np.array(new_node.point) - np.array(self.node_list[min_cost_idx].point))
        new_node.parent = self.node_list[min_cost_idx]
        return True

    def rewire(self, new_node, near_nodes):
        for idx in near_nodes:
            near_node = self.node_list[idx]
            if self.collision_free(new_node.point) and np.linalg.norm(
                    np.array(new_node.point) - np.array(near_node.point)) <= self.step_size:
                new_cost = new_node.cost + np.linalg.norm(np.array(new_node.point) - np.array(near_node.point))
                if near_node.cost > new_cost:
                    near_node.parent = new_node
                    near_node.cost = new_cost

    @staticmethod
    def trace_path(goal_node, goal_point):
        final_path = [goal_point]
        current_node = goal_node
        while current_node is not None:
            final_path.append(current_node.point)
            current_node = current_node.parent
        return np.array(final_path[::-1])

    def plan(self, use_goal_point=True):
        start_time = time.time()
        self.node_list.append(self.start)
        for _ in range(self.max_iter):
            if use_goal_point and np.random.rand() < 0.1:
                random_point = self.goal.point
            else:
                random_point = self.generate_random_point()
            nearest_node_idx = self.nearest_node(random_point)
            nearest_node = self.node_list[nearest_node_idx]
            new_node = self.steer(nearest_node, Node(random_point))
            if self.collision_free(new_node.point):
                near_nodes_idx = self.near_nodes(new_node)
                if self.choose_parent(new_node, near_nodes_idx):
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_nodes_idx)

        goal_node = None
        for node in self.node_list:
            if np.linalg.norm(np.array(node.point) - np.array(self.goal.point)) <= self.step_size:
                if self.collision_free(node.point):
                    goal_node = node
                    break

        if goal_node is None:
            return None, False

        end_time = time.time()
        time_taken = end_time - start_time
        final_path = self.trace_path(goal_node, self.goal.point)
        return time_taken, True


def test_rrt_star(num_tests, use_goal_point=True):
    start = [1, 1, 1]
    goal = [13, 13, 13]
    obstacle_list = [([6, 6, 6], 1.5), ([9, 10, 9], 1.5), ([10, 6, 10], 1.5), ([8, 6, 8], 1), ([6, 10, 6], 1.5)]
    rand_area = [0, 15]
    step_size = 0.7
    max_iter = 700
    search_radius = 4

    results = []

    for _ in tqdm(range(num_tests), desc="Running tests"):
        rrt_star = RRTStar3D(start, goal, obstacle_list, rand_area, step_size, max_iter, search_radius)
        time_taken, found_path = rrt_star.plan(use_goal_point)
        results.append((time_taken, found_path))

    return results


def save_results_to_csv(results, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Test nums", "Total Time (s)", "Average Time (s)", "Success Rate"])
        total_time = 0
        success_count = 0
        for time_taken, found_path in results:
            if time_taken is not None:
                total_time += time_taken
                if found_path:
                    success_count += 1
        writer.writerow([len(results), total_time, total_time / success_count, success_count / len(results)])


if __name__ == '__main__':
    num_tests = 500
    if not os.path.exists('res'):
        os.makedirs('res')
    results_with_goal_point = test_rrt_star(num_tests, use_goal_point=True)
    save_results_to_csv(results_with_goal_point, "res/results_with_goal_point.csv")

    results_without_goal_point = test_rrt_star(num_tests, use_goal_point=False)
    save_results_to_csv(results_without_goal_point, "res/results_without_goal_point.csv")
