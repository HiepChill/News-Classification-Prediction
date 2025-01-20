import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Bước 1: Đọc dữ liệu từ file CSV
data = pd.read_csv('D:/KPDL/bbc_clean.csv')

# Bước 2: Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(data['final_content'], 
                                                    data['label'], 
                                                    test_size=0.2, 
                                                    random_state=42)

# Bước 3: Sử dụng TfidfVectorizer để chuyển đổi văn bản thành ma trận số
vectorizer = TfidfVectorizer(stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Bước 4: Huấn luyện mô hình Naive Bayes
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Bước 5: Dự đoán và đánh giá mô hình
y_pred = clf.predict(X_test_tfidf)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))