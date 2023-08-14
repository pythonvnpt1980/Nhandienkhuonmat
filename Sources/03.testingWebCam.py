import cv2
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np

# Hàm dự đoán
def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.7):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_frame: ảnh dự đoán
    :param knn_clf: đối tượng phân loại knn. nếu không sẽ lấy từ model_save_path.
    :param model_path: đường dẫn đến bộ phân loại knn.
    :param distance_threshold: ngưỡng khoảng cách để phân loại khuôn mặt. càng lớn, càng có nhiều cơ hội của việc phân loại sai một người chưa biết thành một người đã biết.
    :return: danh sách tên và vị trí khuôn mặt cho các khuôn mặt được nhận dạng trong ảnh: [(tên lớp được phân loại, bounding box), ...].
        Đối với khuôn mặt của những người không được công nhận, tên 'unknown' sẽ được trả lại.
    """
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load model KNN đã huấn luyện
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    # nhận diện vị trí khuôn mặt
    X_face_locations = face_recognition.face_locations(X_frame)

    # Nếu không tìm thấy khuôn mặt nào trong ảnh, trả về kết quả trống.
    if len(X_face_locations) == 0:
        return []

    # Tìm mã hóa cho khuôn mặt trong hình ảnh
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    # Sử dụng mô hình KNN để tìm kết quả phù hợp nhất cho khuôn mặt
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=5)

    # So Sánh với ngưỡng
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    #temp = knn_clf.predict(faces_encodings)
    #print(temp)
    # Dự đoán các lớp và loại bỏ các phân loại không nằm trong ngưỡng
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def show_prediction_labels_on_image(frame, predictions):
    """
    Hiển thị kết quả nhận dạng khuôn mặt.

    :param frame: khung để hiển thị các dự đoán.
    :param predictions: kết quả hàm dự đoán
    """
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # phóng to các dự đoán cho hình ảnh có kích thước đầy đủ. nhân lên 4 lần
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Vẽ box xung quanh khuôn mặt
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        name = name.encode("UTF-8")

        # Vẽ nhãn có tên bên dưới khuôn mặt
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Xóa thư viện bản vẽ khỏi bộ nhớ.
    del draw
    # Lưu hình ảnh ở định dạng open-cv để hiển thị .
    opencvimage = np.array(pil_image)
    return opencvimage

if __name__ == "__main__":
    # Lấy dữ liệu từ webcam
    cap = cv2.VideoCapture(0)
    process_this_frame = True
    while 1 > 0:
        ret, frame = cap.read()
        if ret:
            # Resize ảnh
            img = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            predictions = []
            if process_this_frame:
                predictions = predict(img, model_path="models/weight/trained_knn_model.weight")
            frame = show_prediction_labels_on_image(frame, predictions)
            cv2.imshow('camera', frame)
            if ord('q') == cv2.waitKey(10):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)
