import streamlit as st

st.set_page_config(page_title="Home", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title("ĐỒ ÁN MÔN HỌC MÁY")
st.header("Phần 1: Giới thiệu")
st.subheader("Giảng viên hướng dẫn")
st.write("ThS. Trần Tiến Đức")

st.subheader("Sinh viên thực hiện")
st.write("- Nguyễn Minh Cường  - 20110132")
st.write("- Nguyễn Minh Nhựt   - 20110534")

st.header("Phần 2: Các chức năng chính")
st.subheader("Phát Hiện Khuôn Mặt")
st.write("Chức năng này sử dụng các thuật toán Máy học để phát hiện và nhận dạng khuôn mặt trong CAMERA.")

st.subheader("Nhận Diện Khuôn Mặt")
st.write("Chức năng này sử dụng các mô hình Máy học đã huấn luyện để nhận dạng và xác định danh tính của các khuôn mặt trong CAMERA")

st.subheader("Nhận diện ký tự số")
st.write("Chức năng này sử dụng các mô hình Máy học đã huấn luyện để nhận dạng các ký tự số học đơn giản")

st.subheader("Dự báo giá nhà Cali")
st.write("Chức năng này sử dụng các Thuật toán Hồi quy rừng ngẫu nhiên và các thuộc tính khác nhau ảnh hưởng đến giá nhà ở California để tính toán ra giá nhà chính xác")

st.subheader("Nhận diện trái cây - Yolo5")
st.write("Chức năng này sử dụng các mô hình Object Detection để nhận dạng 5 loại trái cây (Bưởi, Cam, Cóc, Khế, Mít) đã được huấn luyện")

st.subheader("Nhận dạng chữ viết tay Tiếng Việt")
st.write("Sử dụng các mô hình Máy học:")
st.write("- CRNN & CTC")
st.write("- TransformerOCR")
st.write("Chức năng sử dụng 2 mô hình đã huấn luyện trên để nhận diện ảnh chữ viết tay Tiếng Việt")
