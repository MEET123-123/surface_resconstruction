import sys
import time

from ultralytics import YOLO
from ultralytics.data.utils import autosplit
import cv2
import numpy as np
import os
import shutil
from collections import deque
from sksurgerynditracker.nditracker import NDITracker
import rtde

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

left_camera_matrix = np.array([[1486.1,0,576.8997],
                               [0,1476.1,345.9153],
                               [0,0,1]])
right_camera_matrix = np.array([[1480.6,0,603.4164],
                               [0,1471.3,319.7381],
                               [0,0,1]])

# k1 k2 k3,p1 p2
# k1 k2 p1 p2 k3
left_dist_coeffs = np.array([-0.4820,0.3408,0.0077,-0.0042,-0.2191])
right_dist_coeffs = np.array([-0.4638,-0.0451,0.0082,-0.0011,2.3658])
R = np.array([[1.0000,0.0005,0.0052],
             [-0.00058766,1.000,0.0075],
             [-0.0052,-0.0075,1.0000]])
T = np.array([[-59.5834],[0.0843],[-0.0506]])
# pretrained_model = 'E:/surface_resconstruction/deep_learning/ultralytics_main/ultralytics_main/backbone/yolov8n.pt'

# 根据识别另一个标志物来定位其左右平移了多少，四个角放四个标志物，取其平均值就是其平移的距离，然后根据平移的距离来判断其对应关系，根据其对应
# 关系来判断其对应点，再根据对应点来判断其深度图像。
# TODO：加2个标志物来确定其四个角，再根据其确定其偏移关系。再确定其点。
# 定位点
# 确定对应关系后（采用前面的找对应点的方法）
class stereo_surface:
    def __init__(self,left_camera_matrix,right_camera_matrix,left_dist_coeffs,right_dist_coeffs,R,T,model):
        self.left_camera_matrix = left_camera_matrix
        self.right_camera_matrix = right_camera_matrix
        self.left_dist_coeffs = left_dist_coeffs
        self.right_dist_coeffs = right_dist_coeffs
        self.R = R
        self.T = T
        self.model = model

        # b : baseline,stands for the distance between two camera
        # b = ||T||
        # b = 59.5834811202736
        # f : focal length
        # d : dispatity
        # d = ul - ur
        # ul : left camera pixel coordinate value
        # ur : right camera pixel coordinate value
        # Z = b * f / d
        # d = xl - xr
        self.focal = 1486.1
        self.baseline = 59.5835
        self.cx = 576.8997
        self.cy = 345.9153

        self.img_size = (1280,720)
        self.rectangular_point = (0,0)
        self.circle_point = (0,0)
        self.distance_pic = 0

    def calibration(self,frame):
        left_frame = frame[0:720, 0:1280]
        right_frame = frame[0:720, 1280:2560]

        left_rectify, right_rectify, left_map, right_map, Q, valid_roi1, valid_roi2 = cv2.stereoRectify(
            self.left_camera_matrix,
            self.left_dist_coeffs,
            self.right_camera_matrix,
            self.right_dist_coeffs,
            self.img_size, self.R, self.T)
        maplx, maply = cv2.initUndistortRectifyMap(self.left_camera_matrix,
                                                   self.left_dist_coeffs,
                                                   left_rectify, left_map,
                                                   self.img_size, cv2.CV_16SC2)
        maprx, mapry = cv2.initUndistortRectifyMap(self.right_camera_matrix,
                                                   self.right_dist_coeffs,
                                                   right_rectify, right_map,
                                                   self.img_size, cv2.CV_16SC2)
        left_image_corrected = cv2.remap(left_frame, maplx, maply, cv2.INTER_LINEAR)
        right_image_corrected = cv2.remap(right_frame, maprx, mapry, cv2.INTER_LINEAR)
        return left_image_corrected, right_image_corrected

    def take_pic(self,save_path_left,save_path_right):
        # 创建 VideoCapture 对象并打开摄像头
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 60)
        # camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        frame_count = 0

        while camera.isOpened():
            ret, frame = camera.read()  # 读取摄像头
            print('ret:', ret)
            # 拆分成两个框
            left_frame = frame[0:720, 0:1280]
            right_frame = frame[0:720, 1280:2560]
            if not ret:
                break
            cv2.imshow('left', left_frame)
            cv2.imshow('right', right_frame)

            # 等待用户按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # 按下空格键保存图像
                frame_count += 1
                filename_left = f"left_{frame_count}.jpg"
                filename_right = f"right_{frame_count}.jpg"
                cv2.imwrite(os.path.join(save_path_left, filename_left), left_frame)
                cv2.imwrite(os.path.join(save_path_right, filename_right), right_frame)

                print(f"图片已保存: {filename_left}")
                print(f"图片已保存: {filename_right}")

            if key == ord('q'):
                break

    def train_model(self):
        # 0 : cross
        # 1 : rectangular
        # 2 : circle

        # must in function of if __name__ == '__main__':
        # load a pretrained model (recommended for training)
        model = YOLO(
            'E:/surface_resconstruction/deep_learning/ultralytics_main/ultralytics_main/backbone/yolov8n.pt')
        # Train the model
        model.train(data='E:/surface_resconstruction/deep_learning/ultralytics_main/ultralytics_main/cross.yaml',
                    epochs=100, imgsz=640)

    def interferce(self,model_path):

        # 使用单幅图像测试
        model_interferce = YOLO(model_path)
        # image = cv2.imread("E:/surface_resconstruction/datasets/crop/images/2.jpg")
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 60)
        # camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        frame_count = 0
        start_time = time.time()

        queue_left_rectangular = deque(maxlen=5)
        queue_left_circle = deque(maxlen=5)
        queue_right_rectangular = deque(maxlen=5)
        queue_right_circle = deque(maxlen=5)

        cv2.namedWindow('Heatmap',cv2.WINDOW_NORMAL)
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        while camera.isOpened():
            ret, frame = camera.read()  # 读取摄像头
            print('ret:', ret)
            # 拆分成两个框
            left_frame,right_frame = self.calibration(frame)
            if not ret:
                break
            results_left = model_interferce(left_frame)
            results_right = model_interferce(right_frame)
            # annotated_left = results_left[0].plot(conf=False)
            # annotated_right = results_right[0].plot(conf=False)
            # cv2.imshow('annotated_left', annotated_left)
            # cv2.imshow('annotated_right', annotated_right)

            boxes_left = results_left[0].boxes
            boxes_right = results_right[0].boxes

            cls_left = boxes_left.cls
            xyxy_left = boxes_left.xyxy
            cls_right = boxes_right.cls
            xyxy_right = boxes_right.xyxy
            xyxy_left_numpy = xyxy_left.cpu().numpy()
            xyxy_right_numpy = xyxy_right.cpu().numpy()
            # (24, )
            cls_left_numpy = cls_left.cpu().numpy()
            cls_right_numpy = cls_right.cpu().numpy()
            # (1,24)
            cls_left_numpy = cls_left_numpy.reshape(1,-1)
            cls_right_numpy = cls_right_numpy.reshape(1,-1)

            # 2*24
            center_left = np.array([(xyxy_left_numpy[:, 0] + xyxy_left_numpy[:, 2]) / 2,
                                    (xyxy_left_numpy[:, 1] + xyxy_left_numpy[:, 3]) / 2])
            center_right = np.array([(xyxy_right_numpy[:, 0] + xyxy_right_numpy[:, 2]) / 2,
                                     (xyxy_right_numpy[:, 1] + xyxy_right_numpy[:, 3]) / 2])

            # 3*24
            all_info_left = np.vstack((cls_left_numpy,center_left))
            all_info_right = np.vstack((cls_right_numpy,center_right))
            print('all_info_left.shape=',all_info_left.shape)
            print('all_info_right.shape=', all_info_right.shape)
            print('all_info_left',all_info_left)
            print('all_info_right',all_info_right)


            if(center_left.shape[1]>0):
                for i in range(center_left.shape[1]):
                    cv2.circle(left_frame, (int(center_left[0, i]), int(center_left[1, i])), 5, (0, 255, 0), 2)

            if(center_right.shape[1]>0):
                for i in range(center_right.shape[1]):
                    cv2.circle(right_frame, (int(center_right[0,i]), int(center_right[1,i])), 5, (0,255,0), 2)

            cv2.imshow('left_frame',left_frame)
            cv2.imshow('right_frame',right_frame)

            # 1,2用来定位，其余点用来匹配
            # 定位点放在一个队列中，来及时更新其位置，取到20个定位点之后开始定位
            indices_one = np.where(all_info_left[0]==1)[0]
            if (indices_one > 0).any():
                column_index = indices_one[0]
                point = all_info_left[:,column_index]
                queue_left_rectangular.append(point)
            if len(queue_left_rectangular) > 5:
                queue_left_rectangular.popleft()


            indices_one = np.where(all_info_left[0] == 2)[0]
            if (indices_one > 0).any():
                column_index = indices_one[0]
                point = all_info_left[:, column_index]
                queue_left_circle.append(point)
            if len(queue_left_circle) > 5:
                queue_left_circle.popleft()


            indices_one = np.where(all_info_right[0] == 1)[0]
            if (indices_one > 0).any():
                column_index = indices_one[0]
                point = all_info_right[:, column_index]
                queue_right_rectangular.append(point)
            if len(queue_right_rectangular) > 5:
                queue_right_rectangular.popleft()


            indices_one = np.where(all_info_right[0] == 2)[0]
            if (indices_one > 0).any():
                column_index = indices_one[0]
                point = all_info_right[:, column_index]
                queue_right_circle.append(point)
            if len(queue_right_circle) > 5:
                queue_right_circle.popleft()
            # 为了保证实时性，需要保证其队列元素尽可能小

            if len(queue_left_rectangular) > 4 and len(queue_left_circle) > 4 and len(queue_right_rectangular) > 4 and len(queue_right_circle) > 4:
                left_rectangular = sum(queue_left_rectangular)/len(queue_left_rectangular)
                right_rectangular = sum(queue_right_rectangular) / len(queue_right_rectangular)

                left_circle = sum(queue_left_circle)/len(queue_left_circle)
                right_circle = sum(queue_right_circle)/len(queue_right_circle)

                distance_rec = abs(left_rectangular[1]-right_rectangular[1])
                distance_cir = abs(left_circle[1]-right_circle[1])
                print('distance_rec=',distance_rec)
                print('distance_cir=',distance_cir)

                self.distance_pic = (distance_rec+distance_cir)/2
                print('self.distance_pic=',self.distance_pic)

            center_left = center_left.transpose()
            center_right = center_right.transpose()

            # 匹配两幅图
            points_ls = np.zeros((1, 2))
            points_rs = np.zeros((1, 2))
            center_left[:,0] -= self.distance_pic

            for point_l in center_left:
                # 左右两幅图中的点进行匹配
                for point_r in center_right:
                    dist = abs(point_r[0] - point_l[0]) + abs(point_r[1] - point_l[1])
                    if dist < 10:
                        points_ls = np.append(points_ls, point_l)
                        points_rs = np.append(points_rs, point_r)
                        break

            points_ls = points_ls.reshape((-1, 2))
            points_rs = points_rs.reshape((-1, 2))

            points_ls = points_ls[1:, :]
            points_rs = points_rs[1:, :]

            points_ls[:, 0] += self.distance_pic
            print('points_ls', points_ls)
            print('points_rs',points_rs)


            # Z = B*f/d;
            # X = (u-cx)*Z/f
            # Y =(v-cy)*Z/f
            Z = self.focal*self.baseline/(points_ls[:,0]-points_rs[:,0])
            X = (points_ls[:,0]-self.cx)*Z/self.focal
            Y = (points_ls[:,1]-self.cy)*Z/self.focal

            if(Z.shape[0]>1):
                X_min = int(np.min(X))
                X_max = int(np.max(X))
                Y_min = int(np.min(Y))
                Y_max = int(np.max(Y))

                # z = ax^2+by^2+cxy+dx+ey+f
                A = np.column_stack((X ** 2, Y ** 2, X * Y, X, Y, np.ones_like(X)))
                coefficients = np.linalg.lstsq(A, Z, rcond=None)[0]
                # 打印系数
                print("拟合的曲面方程为：")
                print(f"z = {coefficients[0]:.4f}x^2 + "
                      f"{coefficients[1]:.4f}y^2 + "
                      f"{coefficients[2]:.4f}xy + "
                      f"{coefficients[3]:.4f}x + "
                      f"{coefficients[4]:.4f}y + "
                      f"{coefficients[5]:.4f}")
                # filter
                # image = np.zeros((720,1280,3),dtype=np.uint8)
                x = np.linspace(X_min, X_max, 640)
                y = np.linspace(Y_min, Y_max, 480)
                x_grid, y_grid = np.meshgrid(x, y)
                z_grid = self.fit_surface(x_grid, y_grid, coefficients)
                z_grid = z_grid/10
                z_grid_normalized = (z_grid - z_grid.min()) / (z_grid.max() - z_grid.min()) * 255
                z_grid_normalized = z_grid_normalized.astype(np.uint8)

                heatmap = cv2.applyColorMap(z_grid_normalized, cv2.COLORMAP_JET)
                # 创建一个和热力图一样大小的空白图像
                canvas[:heatmap.shape[0],:heatmap.shape[1]] = heatmap
                cv2.imshow('Heatmap',canvas)
            # return points_ls,points_rs
            # 等待用户按键
            frame_count += 1
            if(frame_count>120):
                end_time = time.time()
                total_time = end_time - start_time
                fps = frame_count/total_time
                print(f"FPS={fps:.2f}FPS")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyAllWindows()

    def process_txt(self):
        # 定义文本文件的路径
        txt_file_path = 'E:/surface_resconstruction/datasets/crop/autosplit_val.txt'  # 替换为您的文本文件的实际路径

        # 读取文件内容，去除每行开头的空格，然后写回文件
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()

        with open(txt_file_path, 'w') as file:
            for line in lines:
                # 去除每行开头的空格并写入文件
                stripped_line = line.lstrip()
                file.write(stripped_line)  # 添加换行符

        print("Leading spaces have been removed from each line in the file.")

    def split_relative_path(self,split_datasets=True):
        if split_datasets:
            # 定义数据集目录和分割比例
            data_dir = "E:/surface_resconstruction/datasets/crop"
            weights = (0.9, 0.1, 0.0)  # (train, validation, test) fractional splits
            annotated_only = False

            # 运行 autosplit 函数
            autosplit(
                path=os.path.join(data_dir, "images"),
                weights=weights,
                annotated_only=annotated_only
            )

    def split_data(self,split_datasets=False):
        if split_datasets:
            # 定义数据集目录和分割比例
            data_dir = "E:/surface_resconstruction/datasets/crop"
            weights = (0.9, 0.1, 0.0)  # (train, validation, test) fractional splits
            annotated_only = False

            # 运行 autosplit 函数
            autosplit(
                path=os.path.join(data_dir, "images"),
                weights=weights,
                annotated_only=annotated_only
            )

        # 定义数据集目录和txt文件路径
        # 定义源目录
        src_image_dir = "E:/surface_resconstruction/datasets/crop/images"
        src_label_dir = "E:/surface_resconstruction/datasets/crop/labels"

        # 定义目标目录
        dst_dir = "E:/surface_resconstruction/datasets/crop/train"
        dst_image_dir = os.path.join(dst_dir, "images")
        dst_label_dir = os.path.join(dst_dir, "labels")

        # 创建目标目录
        os.makedirs(dst_image_dir, exist_ok=True)
        os.makedirs(dst_label_dir, exist_ok=True)

        # e.g
        # 0.jpg
        # 1.jpg
        # 10.jpg
        # 100.jpg
        # 101.jpg
        # 102.jpg
        # 103.jpg
        # 104.jpg
        # 105.jpg
        # 106.jpg
        # 108.jpg
        # 109.jpg
        # 11.jpg
        # 110.jpg
        # 111.jpg
        # 114.jpg
        # 读取文本文件中的图像文件名

        with open("E:/surface_resconstruction/datasets/crop/autosplit_train.txt", 'r',encoding='utf-8') as file:
            for line in file:
                # 去掉文件名末尾的换行符
                file_name = line.strip()
                # 构建源图像和标签文件的路径
                src_img_path = os.path.join(src_image_dir, file_name)
                src_label_path = os.path.join(src_label_dir, file_name.replace('.jpg', '.txt'))
                # 构建目标图像和标签文件的路径
                dst_img_path = os.path.join(dst_image_dir, file_name)
                dst_label_path = os.path.join(dst_label_dir, file_name.replace('.jpg', '.txt'))

                # 复制图像和标签文件到目标目录
                shutil.copy(src_img_path, dst_img_path)
                shutil.copy(src_label_path, dst_label_path)

        print("Files have been copied to the images and labels folders.")

    def crop_data(self,input_annotation, input_image, output_image, output_annotation):
        image = cv2.imread(input_image)
        data_raw = np.genfromtxt(input_annotation, delimiter=' ', usecols=(0, 1, 2, 3, 4))
        data = data_raw[:,1:]
        data_annotation = data_raw[:,0]
        data_annotation = data_annotation.reshape(-1,1)
        multipliers = np.array([1280, 720, 640, 360])
        data = data * multipliers
        x_crop = np.random.randint(0, image.shape[1] - 640)
        y_crop = np.random.randint(0, image.shape[0] - 640)
        cropped_image = image[y_crop:y_crop + 640, x_crop:x_crop + 640]
        cv2.imwrite(output_image, cropped_image)
        selected_points = data[(data[:, 0] >= x_crop) & (data[:, 0] <= x_crop + 640) &
                               (data[:, 1] >= y_crop) & (data[:, 1] <= y_crop + 640)]

        row_indices = np.where((data[:, 0] >= x_crop) & (data[:, 0] <= x_crop + 640) &
                               (data[:, 1] >= y_crop) & (data[:, 1] <= y_crop + 640))[0]

        data_annotation = data_annotation[row_indices,:]

        selected_points[:, 0] -= x_crop
        selected_points[:, 1] -= y_crop

        # selected points
        divide = np.array([640, 640, 320, 320])
        coco_data = selected_points / divide
        print('data_annotation.shape=',data_annotation.shape)
        print('coco_data.shape=',coco_data.shape)
        coco_data = np.hstack((data_annotation, coco_data))

        # file_path = 'E:/surface_resconstruction/deep_learning/crop_data/crop1.txt'
        np.savetxt(output_annotation, coco_data, fmt=['%d'] + ['%.8f'] * (coco_data.shape[1] - 1),
                   delimiter=' ')

    def process_crop_data(self,nums_pic):
        # TODO : concat string
        # 根据COCO格式来随机分割图片(640*640)
        # 一张原始图形分割5张图片
        # 分割后将其按照1，2，3...的序列进行排序
        # input_annotation_path:输入图形的标注路径
        # input_image_path：输入图像的路径
        # output_annotation_path：输出图像的标注路径
        # output_image_path： 输出图像的路径
        for i in range(nums_pic):
            input_annotation_path = f'E:/surface_resconstruction/datasets/right_datasets/annotations/right_{i+1}.txt'
            input_image_path = f'E:/surface_resconstruction/datasets/right_datasets/images/right_{i+1}.jpg'
            for j in range(5):
                temp_order = 344+ i * 5 + j
                output_annotation_path = f'E:/surface_resconstruction/datasets/crop/annotations/{temp_order}.txt'
                output_image_path = f'E:/surface_resconstruction/datasets/crop/images/{temp_order}.jpg'
                self.crop_data(input_annotation_path, input_image_path, output_image_path, output_annotation_path)

    def recognize(self,left_pic,right_pic):

        # 0 : cross
        # 1 : rectangular
        # 2 : circle

        results_left = model(left_pic)
        results_right = model(right_pic)

        boxes_left = results_left[0].boxes
        boxes_right = results_right[0].boxes

        xyxy_left = boxes_left.xyxy
        xyxy_right = boxes_right.xyxy
        xyxy_left_numpy = xyxy_left.cpu().numpy()
        xyxy_right_numpy = xyxy_right.cpu().numpy()

        center_left = np.array([(xyxy_left_numpy[:, 0] + xyxy_left_numpy[:, 2]) / 2,
                                (xyxy_left_numpy[:, 1] + xyxy_left_numpy[:, 3]) / 2])
        center_right = np.array([(xyxy_right_numpy[:, 0] + xyxy_right_numpy[:, 2]) / 2,
                                 (xyxy_right_numpy[:, 1] + xyxy_right_numpy[:, 3]) / 2])


        # TODO:distance between two pictures
        self.distance_pic = 20
        return center_left,center_right


        points_ls = np.zeros((1, 2))
        points_rs = np.zeros((1, 2))

        center_left[:, 0] += distance_two_pic

        for point_l in center_left:
            # 左右两幅图中的点进行匹配
            for point_r in center_right:
                dist = abs(point_r[0] - point_l[0]) + abs(point_r[1] - point_l[1])
                if dist < 10:
                    points_ls = np.append(points_ls, point_l)
                    points_rs = np.append(points_rs, point_r)
                    break

        points_ls = points_ls.reshape((-1, 2))
        points_rs = points_rs.reshape((-1, 2))

        points_ls = points_ls[1:, :]
        points_rs = points_rs[1:, :]

        points_ls[:, 0] -= distance_two_pic

    def process_dataset(self):
        # TODO: datasets
        # albumentations
        # import torchvision.transforms as transforms
        pass

    def fit_surface(self,x,y,coefficients):
        return coefficients[0] * x ** 2 + \
            coefficients[1] * y ** 2 + \
            coefficients[2] * x * y + \
            coefficients[3] * x + \
            coefficients[4] * y + \
            coefficients[5]

    def test_matrix(self):
        all_info_left = np.array([
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [897.14, 740.25, 593.5, 912.24, 833.66, 906.36, 756.17, 818.62, 610.38, 828.27, 511.42, 672.41, 535.65,
             667.06, 808.64, 600.66, 673.01, 681.18, 528.08, 520.38, 749.84, 897.16, 744.3, 602.31],
            [343.3, 329.72, 316.17, 75.967, 68.084, 163.27, 62.272, 250.4, 49.653, 154.34, 306.88, 235.24, 43.294,
             322.56, 337.64, 229.17, 142.92, 55.755, 130.43, 221.42, 149.14, 256.11, 243.95, 136.05]
        ])

        all_info_right = np.array([
            [2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [717.48, 417.22, 577.87, 646.83, 560.65, 732.72, 345.93, 723.33, 360.07, 716.51, 496.55, 336.64, 425.48,
             630.61, 495.07, 565.76, 637.14, 654.71, 503.85, 353.14, 571.17, 488.43, 425.22, 433.63],
            [344.65, 315.11, 62.66, 156.91, 330.63, 75.902, 221.97, 162.13, 43.583, 255.87, 235.94, 307.86, 229.14,
             336.75, 143.18, 242.62, 248.71, 68.583, 55.374, 129.23, 147.27, 323.37, 134.66, 47.967]
        ])

        # TODO: Calc the sum of absolute difference
        all_info_right = np.round(all_info_right)
        all_info_left = np.round(all_info_left)

        distance = 178
        all_info_left = all_info_left[1:,:]
        all_info_right = all_info_right[1:,:]

        all_info_right = all_info_right.T
        all_info_left = all_info_left.T

        points_ls = np.zeros((1,2))
        points_rs = np.zeros((1, 2))
        all_info_left[:,0] -= distance

        for point_l in all_info_left:
            # 左右两幅图中的点进行匹配
            for point_r in all_info_right:
                dist = abs(point_r[0] - point_l[0]) + abs(point_r[1] - point_l[1])
                if dist < 10:
                    points_ls = np.append(points_ls, point_l)
                    points_rs = np.append(points_rs, point_r)
                    break

        points_ls = points_ls.reshape((-1, 2))
        points_rs = points_rs.reshape((-1, 2))

        points_ls = points_ls[1:, :]
        points_rs = points_rs[1:, :]

        points_ls[:, 0] += distance
        print('points_ls', points_ls)
        print('points_rs', points_rs)

        # Z = B*f/d;
        # X = (u-cx)*Z/f
        # Y =(v-cy)*Z/f
        Z = self.focal * self.baseline / (points_ls[:, 0] - points_rs[:, 0])
        X = (points_ls[:, 0] - self.cx) * Z / self.focal
        Y = (points_ls[:, 1] - self.cy) * Z / self.focal

        X = X.reshape((-1, 1))
        Y = Y.reshape((-1, 1))
        Z = Z.reshape((-1, 1))

        # 计算出X,Y的范围，再根据函数拟合出一个曲面。
        X_min = int(np.min(X))
        X_max = int(np.max(X))
        Y_min = int(np.min(Y))
        Y_max = int(np.max(Y))

        # z = ax^2+by^2+cxy+dx+ey+f
        A = np.column_stack((X**2,Y**2,X*Y,X,Y,np.ones_like(X)))
        coefficients = np.linalg.lstsq(A,Z,rcond=None)[0]
        # 打印系数
        print("拟合的曲面方程为：")
        print(f"z = {coefficients[0][0]:.4f}x^2 + "
              f"{coefficients[1][0]:.4f}y^2 + "
              f"{coefficients[2][0]:.4f}xy + "
              f"{coefficients[3][0]:.4f}x + "
              f"{coefficients[4][0]:.4f}y + "
              f"{coefficients[5][0]:.4f}")
        # filter
        #image = np.zeros((720,1280,3),dtype=np.uint8)
        x = np.linspace(X_min,X_max,640)
        y = np.linspace(Y_min,Y_max,480)
        x_grid,y_grid = np.meshgrid(x,y)
        z_grid = self.fit_surface(x_grid,y_grid,coefficients)

        z_grid_normalized = (z_grid-z_grid.min())/(z_grid.max()-z_grid.min())*255
        z_grid_normalized = z_grid_normalized.astype(np.uint8)

        heatmap = cv2.applyColorMap(z_grid_normalized,cv2.COLORMAP_JET)
        # 创建一个和热力图一样大小的空白图像
        heatmap_full_size = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 将热力图映射到对应的位置
        heatmap_full_size[:heatmap.shape[0], :heatmap.shape[1]] = heatmap

        # 显示图像
        cv2.imshow('Heatmap', heatmap_full_size)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


img_size = (1280,720)

model = YOLO('E:/surface_resconstruction/runs/detect/train109/weights/best.pt')
#camera = cv2.VideoCapture(0)

#camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
#camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#camera.set(cv2.CAP_PROP_FPS, 60)
test_stereo = stereo_surface(left_camera_matrix,right_camera_matrix,left_dist_coeffs,right_dist_coeffs,R,T,model)

# 制作数据集，先拍照，再对数据集的处理类进行继承，处理数据。
left_save_path = 'E:/calib_pic/left_calib'
right_save_path = 'E:/calib_pic/right_calib'
# test_stereo.take_pic(left_save_path,right_save_path)
# test_stereo.split_data()
# test_stereo.split_relative_path()
# test_stereo.process_txt()
model_path = 'E:/surface_resconstruction/runs/detect/train124/weights/best.pt'
test_stereo.interferce(model_path)
# test_matrix()
# 总图像数(左图像)
# num_pic = 69
# test_stereo.process_crop_data(num_pic)
save_path = 'E:/surface_resconstruction/datasets/crop'
# train_model(save_path)
# TODO: filter the surface and display


'''
#######################初始化NDI##############################
SETTINGS = {
        "tracker type": "polaris",
#         "romfiles": ["./NDI_Rom/8700339qjx.rom"]
        "romfiles": ["E:/surface_resconstruction/NDI_after/NDI/NDI_Rom/8700339.rom"]
}
TRACKER = NDITracker(SETTINGS)
TRACKER.start_tracking()

while(True):
    port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
    print(tracking[0])



TRACKER.stop_tracking()
TRACKER.close()

'''

'''
model = YOLO(
            'E:/surface_resconstruction/deep_learning/ultralytics_main/ultralytics_main/backbone/yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data=r'E:/surface_resconstruction/deep_learning/ultralytics_main/ultralytics_main/cross.yaml',
                    epochs=100, imgsz=640)
model.save(save_path)

'''
'''
if not camera.isOpened():
    print("VideoCapture 未成功打开")
    exit()

while camera.isOpened():
    success, frame = camera.read()
    if success:
        left_frame = frame[0:720, 0:1280]
        right_frame = frame[0:720, 1280:2560]
        pic1,pic2 = test_stereo.calibration(frame)
        cv2.imshow("pic1",pic1)
        cv2.imshow("pic2",pic2)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:break
'''

'''
start_tick = cv2.getTickCount()
frame_count = 0

while camera.isOpened():
    success, frame = camera.read()
    if success:

        frame_count+=1

        if frame_count>=120:
            end_tick = cv2.getTickCount()
            time_interval = (end_tick-start_tick)/cv2.getTickFrequency()
            fps=frame_count/time_interval
            print(f"Estimated FPS:{fps}")

        left_frame = frame[0:720, 0:1280]
        right_frame = frame[0:720, 1280:2560]
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 60)

        left_rectify, right_rectify, left_map, right_map, Q, valid_roi1, valid_roi2 = cv2.stereoRectify(
            left_camera_matrix,
            left_dist_coeffs,
            right_camera_matrix,
            right_dist_coeffs,
            img_size, R, T)
        maplx, maply = cv2.initUndistortRectifyMap(left_camera_matrix, left_dist_coeffs, left_rectify, left_map,
                                                   img_size,
                                                   cv2.CV_16SC2)
        maprx, mapry = cv2.initUndistortRectifyMap(right_camera_matrix, right_dist_coeffs, right_rectify, right_map,
                                                   img_size, cv2.CV_16SC2)
        left_image_corrected = cv2.remap(left_frame, maplx, maply, cv2.INTER_LINEAR)
        right_image_corrected = cv2.remap(right_frame, maprx, mapry, cv2.INTER_LINEAR)

        results_left = model(left_image_corrected)  # 对当前帧进行推理
        results_right = model(right_image_corrected)

        boxes_left = results_left[0].boxes
        boxes_right = results_right[0].boxes

        xyxy_left = boxes_left.xyxy
        xyxy_right = boxes_right.xyxy
        xyxy_left_numpy = xyxy_left.cpu().numpy()
        xyxy_right_numpy = xyxy_right.cpu().numpy()
        print('xyxy_left_numpy_shape=',xyxy_left_numpy.shape)
        center_left = np.array([(xyxy_left_numpy[:,0]+xyxy_left_numpy[:,2])/2,
                                (xyxy_left_numpy[:,1]+xyxy_left_numpy[:,3])/2])
        center_right = np.array([(xyxy_right_numpy[:,0]+xyxy_right_numpy[:,2])/2,
                                (xyxy_right_numpy[:,1]+xyxy_right_numpy[:,3])/2])
        # TODO:distance between two pictures
        distance_two_pic = 20

        points_ls = np.zeros((1, 2))
        points_rs = np.zeros((1, 2))

        center_left[:, 0] += distance_two_pic

        for point_l in center_left:
            # 左右两幅图中的点进行匹配
            for point_r in center_right:
                dist = abs(point_r[0] - point_l[0]) + abs(point_r[1] - point_l[1])
                if dist < 10:
                    points_ls = np.append(points_ls, point_l)
                    points_rs = np.append(points_rs, point_r)
                    break

        points_ls = points_ls.reshape((-1, 2))
        points_rs = points_rs.reshape((-1, 2))

        points_ls = points_ls[1:, :]
        points_rs = points_rs[1:, :]

        points_ls[:, 0] -= distance_two_pic

        # calculate the 3D coordinates
        # Calcalate the depth
        # b : baseline,stands for the distance between two camera
        # f : focal length
        # d : dispatity
        # d = ul - ur
        # ul : left camera pixel coordinate value
        # ur : right camera pixel coordinate value
        # Z = b * f / d
        # d = xl - xr

        # left_camera
        # TODO: change T:world coordinate
        T = np.array([[164.5134], [-0.4882], [3.5058]])
        # TODO: change f:focal
        baseline = 164.5514
        f = 882.1833

        merged_matrix = np.hstack((points_ls, points_rs))
        print('merged_matrix_shape=', merged_matrix.shape)

        depth = baseline * f / (merged_matrix[:, 0] - merged_matrix[:, 2])
        depth = depth.reshape((-1, 1))

        merged_matrix_depth = np.hstack((merged_matrix, depth))
        print('merged_matrix_depth=', merged_matrix_depth)

        K = np.array([[882.1833, 0, 660.0165],
                      [0, 870.4806, 287.3848],
                      [0, 0, 1]])

        M = np.array([[1, 0, -0.3],
                      [0, 1, 0],
                      [0, 0, 1]])

        coordinate_all = np.zeros((3, 1))

        for i in range(merged_matrix_depth.shape[0]):
            u = merged_matrix_depth[i, 0]
            v = merged_matrix_depth[i, 1]
            pixel_coordinate = np.array([u, v, 1])
            world_coordinate = merged_matrix_depth[i, 4] * np.linalg.inv(M) @ np.linalg.inv(K) @ pixel_coordinate
            coordinate_all = np.append(coordinate_all, world_coordinate)

        coordinate_all = coordinate_all.reshape((-1, 3))
        coordinate_all = coordinate_all[1:, :]
        print('coordinate_all=', coordinate_all)


        #print('xywh_left',boxes_left.xyxy)
        #print('xywh_right ',boxes_right.xyxy)
        # 处理推理结果，例如绘制边界框等
        annotated_frame_left = results_left[0].plot()  # 绘制边界框
        annotated_frame_right = results_right[0].plot()  # 绘制边界框
        cv2.imshow("YOLOv8 Inference_left", annotated_frame_left)
        cv2.imshow("YOLOv8 Inference_right", annotated_frame_right)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break


if __name__ == '__main__':
    # Load a model
    model = YOLO(
        'E:/surface_resconstruction/deep_learning/ultralytics_main/ultralytics_main/backbone/yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    model.train(data='E:/surface_resconstruction/deep_learning/ultralytics_main/ultralytics_main/cross.yaml',
                epochs=100, imgsz=640)
'''
'''
if __name__ == '__main__':
    # Load a model
    model = YOLO(
        'E:/surface_resconstruction/deep_learning/ultralytics_main/ultralytics_main/backbone/yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    model.train(data='E:/surface_resconstruction/deep_learning/ultralytics_main/ultralytics_main/cross.yaml',
                epochs=100, imgsz=640)
'''