import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Bước 1: Đọc dữ liệu từ file CSV
df = pd.read_csv("bbc_clean.csv")
print("Danh sách cột trong DataFrame:", df.columns)

# Bước 2: Kiểm tra và sử dụng cột phù hợp chứa nội dung văn bản
if 'final_content' in df.columns:
    X = df["final_content"]
elif 'content' in df.columns:  # Thử một tên cột khác nếu 'final_content' không tồn tại
    X = df["content"]
else:
    raise KeyError("Không tìm thấy cột chứa nội dung văn bản. Kiểm tra lại dữ liệu đầu vào.")

# Bước 3: Tách nhãn (thể loại)
if 'label' in df.columns:
    y = df["label"]
else:
    raise KeyError("Không tìm thấy cột chứa nhãn (label). Kiểm tra lại dữ liệu đầu vào.")

# Bước 4: Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Bước 5: Chuẩn hóa dữ liệu bằng TF-IDF
# Khởi tạo bộ trích xuất đặc trưng TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

# Áp dụng TF-IDF cho tập huấn luyện
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()  # Chuyển sang mảng để phù hợp với GMM

# Áp dụng TF-IDF cho tập kiểm tra
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Bước 6: Huấn luyện một mô hình Gaussian Mixture Model
# Lấy danh sách các lớp (thể loại) duy nhất trong tập huấn luyện
classes = np.unique(y_train)

# Huấn luyện một mô hình Gaussian Mixture Model (GMM) cho từng lớp
gmm_models = {}
for cls in classes:
    # Lấy dữ liệu thuộc lớp hiện tại
    X_cls = X_train_tfidf[np.array(y_train == cls)]
    # Khởi tạo và huấn luyện mô hình GMM
    gmm = GaussianMixture(n_components=1, random_state=42)  # 1 thành phần (cluster) cho mỗi lớp
    gmm.fit(X_cls)
    gmm_models[cls] = gmm

# Bước 7: Dự đoán nhãn cho tập kiểm tra
y_pred = []
for x in X_test_tfidf:
    # Tính log-likelihood cho từng lớp
    log_likelihoods = {cls: gmm.score([x]) for cls, gmm in gmm_models.items()}
    # Chọn lớp có log-likelihood cao nhất
    y_pred.append(max(log_likelihoods, key=log_likelihoods.get))

# Bước 8: Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)  # Tính độ chính xác
print(f"Accuracy: {accuracy:.4f}")

# Báo cáo chi tiết về kết quả phân loại
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Ma trận nhầm lẫn để xem chi tiết lỗi
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Bước 9: Dự đoán dữ liệu mới
new_data = [
    "The proliferation of electricity-hungry data centers to power our digital lives – and increasingly, the AI technology that tech giants say is the future – now means that energy demand could soon outstrip supply. And that would be a problem for tech companies who are angling for their AI technology to revolutionize almost everything about the way we live and work."
]
new_data_tfidf = vectorizer.transform(new_data).toarray()  # Chuyển đổi dữ liệu mới sang đặc trưng TF-IDF

# Kiểm tra dự đoán cho dữ liệu mới
try:
    new_data_pred = []
    for x in new_data_tfidf:
        log_likelihoods = {cls: gmm.score([x]) for cls, gmm in gmm_models.items()}
        new_data_pred.append(max(log_likelihoods, key=log_likelihoods.get))  # Chọn lớp có likelihood cao nhất
    print("Predicted label for new data:", new_data_pred)
except Exception as e:
    print("Lỗi khi dự đoán dữ liệu mới:", str(e))
