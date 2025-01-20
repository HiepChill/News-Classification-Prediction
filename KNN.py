import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Bước 1: Đọc dữ liệu từ tệp CSV
# Tệp dữ liệu chứa nội dung văn bản (final_content) và nhãn (label)
data = pd.read_csv('bbc_clean.csv')  # Thay đường dẫn tới tệp của bạn

# Bước 2: Tiền xử lý dữ liệu
X = data['final_content']  # Biến đặc trưng: nội dung văn bản
y = data['label']          # Biến mục tiêu: nhãn phân loại

# Chuyển đổi dữ liệu văn bản sang dạng số bằng TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Giới hạn số lượng đặc trưng là 5000
X_tfidf = vectorizer.fit_transform(X)           # Tạo ma trận TF-IDF

# Bước 3: Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
# 80% dữ liệu để huấn luyện, 20% để kiểm tra
# random_state đảm bảo kết quả chia dữ liệu luôn giống nhau

# Bước 4: Tìm giá trị k tối ưu cho KNN bằng Cross-Validation
k_range = range(1, 31)  # Thử nghiệm các giá trị k từ 1 đến 30
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)  # Mô hình KNN với k láng giềng
    # Tính điểm chính xác với Cross-Validation (cv=10)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')  
    k_scores.append(scores.mean())  # Lưu điểm trung bình của mỗi k

# Vẽ đồ thị giá trị k và độ chính xác Cross-Validated
plt.plot(k_range, k_scores)
plt.xlabel('Giá trị K cho KNN')
plt.ylabel('Độ chính xác Cross-Validated')
plt.show()

# Chọn giá trị k tốt nhất (k có độ chính xác cao nhất)
best_k = k_range[k_scores.index(max(k_scores))]
print(f"Giá trị k tối ưu là: {best_k}")

# Bước 5: Huấn luyện mô hình KNN với k tối ưu
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)  # Huấn luyện mô hình với tập huấn luyện

# Bước 6: Đánh giá mô hình
y_pred = knn.predict(X_test)  # Dự đoán trên tập kiểm tra
print("Accuracy:", accuracy_score(y_test, y_pred))  # Tính độ chính xác
# Báo cáo chi tiết hiệu suất của mô hình: độ chính xác, recall, F1-score
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
# Ma trận nhầm lẫn: phân phối các dự đoán đúng và sai
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Bước 7: Dự đoán trên dữ liệu mới
new_data = [
    "The proliferation of electricity-hungry data centers to power our digital lives – and increasingly, the AI technology that tech giants say is the future – now means that energy demand could soon outstrip supply. And that would be a problem for tech companies who are angling for their AI technology to revolutionize almost everything about the way we live and work."
]  # Văn bản ví dụ
new_data_tfidf = vectorizer.transform(new_data)  # Chuyển đổi văn bản mới sang ma trận TF-IDF
prediction = knn.predict(new_data_tfidf)  # Dự đoán nhãn
print("Nhãn dự đoán cho dữ liệu mới:", prediction)

# Bước 8: Tạo Word Cloud cho từng nhãn
labels = data['label'].unique()  # Lấy danh sách các nhãn duy nhất
for label in labels:
    # Gộp tất cả văn bản thuộc nhãn cụ thể thành 1 chuỗi
    text = " ".join(content for content in data[data['label'] == label]['content'])
    # Tạo Word Cloud từ văn bản
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Vẽ Word Cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Tắt hiển thị trục
    plt.title(f'Word Cloud cho nhãn: {label}')
    plt.show()
