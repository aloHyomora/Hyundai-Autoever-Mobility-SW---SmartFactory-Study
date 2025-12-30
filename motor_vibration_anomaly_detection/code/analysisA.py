# ==============================
# Library imports
# 라이브러리 불러오기
# ==============================
import itertools          # Iteration utilities (조합/반복 유틸)
import time               # Time measurement (시간 측정)

import numpy as np        # Numerical computation (수치 연산)
import pandas as pd       # DataFrame handling (데이터프레임)

import matplotlib.pyplot as plt   # Visualization (시각화)

from scipy.io.arff import loadarff     # ARFF file loader (ARFF 데이터 로딩)

from sklearn.preprocessing import StandardScaler, RobustScaler
# StandardScaler: 평균 0, 분산 1 정규화
# RobustScaler: 이상치에 강한 스케일링

import torch              # PyTorch library (딥러닝 라이브러리)

# ==============================
# Data import
# 데이터 불러오기
# ==============================
file_path = './dataset/'   # Dataset directory (데이터셋 경로)
train_fn = 'FordA_TRAIN.arff'  # Training data file (학습 데이터 파일)
test_fn = 'FordA_TEST.arff'    # Test data file (테스트 데이터 파일)

def read_arff(file_path):
    raw_data, meta = loadarff(file_path)
    cols = [x for x in meta]
    data2d = np.zeros([raw_data.shape[0], len(cols)])

    for i, col in zip(range(len(cols)), cols):
        data2d[:,i] = raw_data[col]
    return data2d

train = read_arff(file_path + train_fn)
test = read_arff(file_path + test_fn)

print("Train data shape:", train.shape)
print("Test data shape:", test.shape)    

# ==============================
# Data split into training, validation, and test sets
# 학습용(Training), 검증용(Validation), 테스트용(Test) 데이터셋 나누기
# ==============================

x_train_temp = train[:, :-1]    # 첫 번째 : 모든 행, 두 번째 : 마지막 column 제외한 나머지 column
y_train_temp = train[:, -1]    # 마지막 column이 Lable 값이 있는 column, 모든 행, 마지막 열만, 즉 정답 열
x_test = test[:, :-1]
y_test = test[:, -1]    # 마지막 column이 Lable 값이 있는 column

normal_x = x_train_temp[y_train_temp == 1]   # x_train_temp 중 정상 데이터
abnormal_x = x_train_temp[y_train_temp == -1]  # x_train_temp 중 이상 데이터
normal_y = y_train_temp[y_train_temp == 1]      # y_train_temp 중 정상 데이터
abnormal_y = y_train_temp[y_train_temp == -1]  # y_train_temp 중 이상 데이터

index_x_normal = int(normal_x.shape[0] * 0.8)   # 정상 데이터를 8:2로 나누기 위한 인덱스
index_y_normal = int(normal_y.shape[0] * 0.8)   # 정상 데이터를 8:2로 나누기 위한 인덱스
index_x_abnormal = int(abnormal_x.shape[0] * 0.8)   # 이상 데이터를 8:2로 나누기 위한 인덱스
index_y_abnormal = int(abnormal_y.shape[0] * 0.8)   # 이상 데이터를 8:2로 나누기 위한 인덱스

x_train = np.concatenate((normal_x[:index_x_normal], abnormal_x[:index_x_abnormal]), axis=0)    # 행 방향(axis=0)으로 합치기, 열 개수는 같아야 함
x_valid = np.concatenate((normal_x[index_x_normal:], abnormal_x[index_x_abnormal:]), axis=0)
y_train = np.concatenate((normal_y[:index_y_normal], abnormal_y[:index_y_abnormal]), axis=0)
y_valid = np.concatenate((normal_y[index_y_normal:], abnormal_y[index_y_abnormal:]), axis=0)

print("\nData shapes after split:")
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_valid shape:", x_valid.shape)
print("y_valid shape:", y_valid.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)