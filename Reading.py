import ast
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Tải dữ liệu huấn luyện
data = pd.read_csv('shortest_path_training_data.csv')

# Chuyển cột 'obstacles' từ chuỗi sang danh sách
data['obstacles'] = data['obstacles'].apply(lambda x: ast.literal_eval(x))

# Một cách đơn giản: sử dụng số lượng chướng ngại vật làm một đặc trưng
data['num_obstacles'] = data['obstacles'].apply(len)

# Chuẩn bị đầu vào và đầu ra cho mô hình học máy
X = data[['start', 'destination', 'num_obstacles']].values  # Sử dụng num_obstacles làm đặc trưng
y = data['path'].apply(lambda x: len(ast.literal_eval(x))).values  # Dự đoán chiều dài đường đi

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Lưu mô hình sau khi huấn luyện vào file path_finder_model.pkl
joblib.dump(model, 'path_finder_model.pkl')
print("Mô hình đã được huấn luyện và lưu thành công dưới dạng 'path_finder_model.pkl'.")
# Đánh giá lỗi dự đoán (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f'Đánh giá lỗi dự đoán: {mse:.2f}')
