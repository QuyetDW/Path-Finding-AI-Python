import random
import pandas as pd
import backend  # Import backend module, nơi chứa thuật toán Dijkstra
import adjancy_matrix_gen  # Import module sinh ma trận kề


# Hàm tạo một trường hợp dữ liệu
def generate_case():
    size = 400  # Kích thước đồ thị 20x20
    src = random.randint(0, size - 1)  # Vị trí bắt đầu ngẫu nhiên
    dest = random.randint(0, size - 1)  # Vị trí kết thúc ngẫu nhiên

    # Tạo danh sách các chướng ngại vật ngẫu nhiên
    num_obstacles = random.randint(30, 100)  # Số lượng chướng ngại vật ngẫu nhiên
    obstacles = random.sample(range(size), num_obstacles)

    # Tìm đường đi ngắn nhất sử dụng Dijkstra
    path = backend.backened(src, obstacles, dest)

    # Nếu không có đường đi, bỏ qua trường hợp này
    if not path:
        return None

    return {
        'start': src,
        'destination': dest,
        'obstacles': obstacles,
        'path': path
    }


# Hàm sinh tập dữ liệu
def generate_training_data(num_samples):
    data = []
    for _ in range(num_samples):
        case = generate_case()
        if case:
            data.append(case)

    return pd.DataFrame(data)


# Tạo tập dữ liệu huấn luyện với số lượng mẫu mong muốn
if __name__ == "__main__":
    num_samples = 500  # Số lượng mẫu dữ liệu cần tạo
    training_data = generate_training_data(num_samples)

    # Lưu tập dữ liệu ra file CSV để huấn luyện mô hình
    training_data.to_csv('shortest_path_training_data.csv', index=False)
    print("Đã tạo tập dữ liệu huấn luyện và lưu vào 'shortest_path_training_data.csv'")
