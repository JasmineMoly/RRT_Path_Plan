import numpy as np


def angle_between_vectors(from_node, to_node, goal_node):
    # 计算两个向量
    vec1 = np.array(to_node) - np.array(from_node)
    vec2 = np.array(goal_node) - np.array(from_node)

    # 计算两个向量的点积
    dot_product = np.dot(vec1, vec2)

    # 计算两个向量的模
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # 计算夹角（弧度）
    cos_theta = dot_product / (norm_vec1 * norm_vec2)
    theta_radians = np.arccos(cos_theta)
    print(theta_radians)

    # 将弧度转换为角度
    theta_degrees = np.degrees(theta_radians)
    print(dot_product / (norm_vec1 * norm_vec2))

    return theta_degrees


# 测试示例
from_node = [10.88516298, 11.44665602, 10.70583948]
to_node = [10.88516298, 11.44665602, 10.70583948]
goal_node = [13, 13, 13]
angle = angle_between_vectors(from_node, to_node, goal_node)
print("夹角（角度）：", angle)
