import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def collision_free(point1, point2, obstacle_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制连线段
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color='b')

    for obstacle in obstacle_list:
        obstacle_center = np.array(obstacle[0])
        obstacle_radius = obstacle[1]
        # 绘制障碍物球体
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = obstacle_center[0] + obstacle_radius * np.outer(np.cos(u), np.sin(v))
        y = obstacle_center[1] + obstacle_radius * np.outer(np.sin(u), np.sin(v))
        z = obstacle_center[2] + obstacle_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='r', alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    for obstacle in obstacle_list:
        obstacle_center = np.array(obstacle[0])
        obstacle_radius = obstacle[1]
        print(obstacle_center)
        distance = 0
        link1 = point2 - point1
        link2 = obstacle_center - point1
        link3 = obstacle_center - point2
        judge = np.dot(link1, link2) / np.dot(link1, link1)
        if 0 <= judge <= 1:  # 垂足在线段上
            print("link1:", link1)
            print("link2:", link2)
            print("np.dot(link1, link2):", np.dot(link1, link2))
            print("np.dot(link1, link1):", np.dot(link1, link1))
            distance = np.linalg.norm(np.cross(link1, link2)) / np.linalg.norm(link1)
            print("垂足在线段上")
            print(distance)
        else:  # 垂足不在线段上
            distance = min(np.linalg.norm(link2), np.linalg.norm(link3))
            print("垂足不在线段上")
            print(distance)
        if distance < obstacle_radius:
            return False
    return True


# 测试示例
p1 = np.array([10.3325219, 10.37560086,  10.50058212])
p2 = np.array([9.85731, 9.90652773, 10.1590502])
obstacle_list = [([6, 6, 6], 1.5), ([9, 10, 9], 1.5), ([10, 6, 10], 1.5), ([8, 6, 8], 1), ([6, 10, 6], 1.5)]
print(collision_free(p1, p2, obstacle_list))
