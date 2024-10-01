import joblib
import copy
import adjancy_matrix_gen
import pandas as pd
import os
import ast

# Tải mô hình học máy
model = joblib.load('path_finder_model.pkl')

# Đường dẫn tới file dữ liệu huấn luyện
train_data_file = 'shortest_path_training_data.csv'

# Tạo hoặc đọc file CSV chứa dữ liệu huấn luyện
if os.path.exists(train_data_file):
    df = pd.read_csv(train_data_file)
else:
    df = pd.DataFrame(columns=['start', 'destination', 'obstacles', 'path'])


# Hàm lưu dữ liệu huấn luyện
def save_new_case(src, destination, obstacles, path):
    # Chuyển danh sách chướng ngại vật thành chuỗi để lưu trữ trong DataFrame
    obstacles_sorted = sorted(obstacles)
    obstacles_str = str(obstacles_sorted)

    # Kiểm tra xem trường hợp đã tồn tại chưa
    if not ((df['start'] == src) & (df['destination'] == destination) & (df['obstacles'] == obstacles_str)).any():
        # Thêm dữ liệu mới vào DataFrame
        new_row = {'start': src, 'destination': destination, 'obstacles': obstacles_str, 'path': path}
        df.loc[len(df)] = new_row
        df.to_csv(train_data_file, index=False)
        print("Trường hợp mới đã được thêm vào file huấn luyện.")
    else:
        print("Trường hợp đã tồn tại trong tập huấn luyện.")


# Hàm lấy danh sách chướng ngại vật từ chuỗi khi cần sử dụng lại
def get_obstacles_from_str(obstacles_str):
    return ast.literal_eval(obstacles_str)


def backened(src, obstacles, destination):
    # Dự đoán đường đi ngắn nhất bằng mô hình học máy
    def predict_shortest_path(src, dest, obstacles):
        num_obstacles = len(obstacles)
        input_data = [[src, dest, num_obstacles]]
        predicted_path_length = model.predict(input_data)
        print(f"Dự đoán độ dài đường đi là: {predicted_path_length[0]}")
        return predicted_path_length[0]

    # Dự đoán đường đi với mô hình
    predicted_length = predict_shortest_path(src, destination, obstacles)

    # Nếu độ dài dự đoán không khả thi, chạy Dijkstra
    if predicted_length < 1:  # Nếu không có đường đi hợp lệ
        print("Không thể tìm thấy đường đi bằng mô hình. Chuyển sang Dijkstra.")

    def min_distance(dist, sp_set):
        min = 10 ** 10
        global min_index
        for v in range(400):
            if sp_set[v] == False and dist[v] <= min:
                min = dist[v]
                min_index = v
        return min_index

    graph, size = copy.deepcopy(adjancy_matrix_gen.return_matrix())
    parent = [-2 for _ in range(400)]

    for value in obstacles:
        for z in range(size):
            graph[z][value] = 0  # Ngắt kết nối với chướng ngại vật

    dist = [10 ** 10 for _ in range(size)]
    sp_set = [False for _ in range(size)]
    dist[src] = 0
    parent[src] = -1

    for i in range(size - 1):
        u = min_distance(dist, sp_set)
        sp_set[u] = True

        for v in range(size):
            if sp_set[v] is False and graph[u][v] != 0 and dist[u] != 10 ** 10 and dist[u] + graph[u][v] < dist[v]:
                dist[v] = dist[u] + graph[u][v]
                parent[v] = u
    # Kiểm tra nếu không thể tìm thấy đường đến đích
    if dist[destination] == 10 ** 10:  # Không có đường đi đến đích
        return None  # Trả về None để báo hiệu không tìm được đường
    def ancestor(dest):
        path = []
        stop = dest
        while parent[stop] != -1:
            path.append(parent[stop])
            stop = parent[stop]
        return path

    destination_parent = ancestor(destination)

    if not destination_parent:
        print("Không thể tìm thấy đường đi bằng Dijkstra.")
    else:
        # Lưu trường hợp mới vào tập dữ liệu huấn luyện
        save_new_case(src, destination, obstacles, destination_parent)

    return destination_parent
