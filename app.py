import streamlit as st
import joblib
import re

# Hàm làm sạch văn bản
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Loại bỏ ký tự đặc biệt
    text = re.sub(r'http[s]?://\S+', '', text)  # Loại bỏ URL
    return ' '.join(text.split())  # Chuẩn hóa khoảng trắng

# Tải mô hình và vectorizer
try:
    model = joblib.load('spam_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Không tìm thấy 'spam_classifier_model.pkl' hoặc 'tfidf_vectorizer.pkl'. Vui lòng đặt các tệp này trong cùng thư mục với app.py!")
    st.stop()

# Tiêu đề ứng dụng
st.title("Dự đoán Email Spam")
st.markdown("Nhập nội dung email để kiểm tra xem đó là **Spam** hay **Không Spam**. Mô hình sử dụng Logistic Regression với F1-score ~0.951.")

# Sidebar
st.sidebar.header("Thông tin mô hình")
st.sidebar.markdown("""
- **Mô hình**: Logistic Regression  
- **F1-score (cross-validation)**: ~0.951  
- **Dữ liệu huấn luyện**: 5130 email (3880 Không Spam, 1250 Spam)  
- **Vector hóa**: TF-IDF (max_features=10000, ngram_range=(1,2))
""")
st.sidebar.header("Ví dụ email")
st.sidebar.markdown("""
**Spam**:  
"Win a free iPhone now! Click here to claim your prize!"  

**Không Spam**:  
"Meeting tomorrow at 10 AM, please confirm your attendance."
""")

# Ô nhập văn bản
st.subheader("Nhập nội dung email")
email_input = st.text_area("Dán nội dung email vào đây:", height=200, placeholder="Ví dụ: Win a free iPhone now! Click here...")

# Nút dự đoán
if st.button("Dự đoán"):
    if email_input.strip() == "":
        st.warning("Vui lòng nhập nội dung email!")
    else:
        # Làm sạch văn bản
        cleaned_email = clean_text(email_input)
        
        # Chuyển đổi thành đặc trưng TF-IDF
        email_vector = vectorizer.transform([cleaned_email])
        
        # Dự đoán
        prediction = model.predict(email_vector)[0]
        prob = model.predict_proba(email_vector)[0]
        
        # Hiển thị kết quả
        if prediction == 1:
            st.error(f"🚨 Email này là **Spam** (Xác suất: {prob[1]:.2%})")
        else:
            st.success(f"✅ Email này **Không phải Spam** (Xác suất: {prob[0]:.2%})")
        
        # Hiển thị văn bản đã làm sạch và từ khóa quan trọng
        with st.expander("Xem chi tiết"):
            st.markdown("**Nội dung đã làm sạch**:")
            st.write(cleaned_email)
            st.markdown("**Top 5 từ quan trọng trong email**:")
            feature_names = vectorizer.get_feature_names_out()
            email_features = email_vector.toarray()[0]
            top_features = sorted(zip(email_features, feature_names), reverse=True)[:5]
            for score, word in top_features:
                if score > 0:
                    st.write(f"- {word}: {score:.4f}")

# Hướng dẫn sử dụng
st.subheader("Hướng dẫn sử dụng")
st.markdown("""
1. Nhập hoặc dán nội dung email vào ô văn bản.  
2. Nhấn nút **Dự đoán** để xem kết quả.  
3. Kết quả sẽ hiển thị email là **Spam** (🚨) hay **Không Spam** (✅), kèm xác suất.  
4. Nhấn **Xem chi tiết** để xem nội dung đã làm sạch và các từ khóa quan trọng.
""")

# Footer
st.markdown("---")
st.markdown("Ứng dụng được xây dựng bằng Streamlit. Mô hình được huấn luyện trên dữ liệu email SpamAssassin.")