'''
$ pip3 install scikit-learn
$ pip3 install numpy
$ pip3 install opencv-contrib-python
'''
import cv2
import math
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
import matplotlib.pyplot as plt

# Allow image with extension
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}

# function training k-nearest neighbors classifier
def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Duyệt thư mục từng phân lớp
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Duyệt từng ảnh trong từng thư mục
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            # Load ảnh, convert ảnh sang mảng bẳng np.array
            image = face_recognition.load_image_file(img_path)
            # Sử dụng dlib phát hiện vị trí khuôn mặt trong ảnh - đầu ra face_bounding_boxes là tọa độ (top, right, bottom, left) ảnh chứa gương mặt
            face_bounding_boxes = face_recognition.face_locations(image)
            if len(face_bounding_boxes) != 1:
                # Nếu không có khuôn mặt trong ảnh (hoặc quá nhiều người) trong một hình ảnh huấn luyện, bỏ qua hình ảnh đó.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Thêm khuôn mặt cho hình ảnh hiện tại vào tập huấn luyện
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                # Thêm phân lớp tương ứng với khuôn mặt
                y.append(class_dir)

    # Xác định số lượng hàng xóm sẽ sử dụng để tính trọng số trong bộ phân loại KNN
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)
    # cắt 20% cho test, 805 huấn luyện
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Tạo bộ phân loại KNN
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    # Huấn luyện
    knn_clf.fit(X_train, y_train)
    # Kiểm thử, tính độ chính xác
    score = knn_clf.score(X_test, y_test)
    print(" -- Accuracy: ", score * 100, " %")
    # Test thử dự đoán
    predicted = knn_clf.predict(X_test)
    print(" -- Predictions from the classifier:")
    print(predicted)
    print(" -- Target values:")
    print(y_test)

    # Lưu lại file trọng số
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

if __name__ == "__main__":
    print(" -- Training ..............")
    classifier = train("models/train", model_save_path="models/weight/trained_knn_model.weight", n_neighbors=2)
    print(" -- Training complete!")
