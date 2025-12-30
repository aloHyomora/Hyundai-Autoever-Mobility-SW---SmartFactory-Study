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
    data2d = np.zeros([raw_data.shape[0], len(cols)])   # 빈 2D 배열 생성(raw_data 행 수 x meta 길이의 열 수)

    for i, col in zip(range(len(cols)), cols):  # 열 인덱스 i와 열 이름 col을 함께 반복
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

# ==============================
# Data Visualization
# 데이터 시각화 1: 데이터 불균형(Data Imbalance) 확인
# ==============================

# 클래스 종류 확인: 정상(1), 비정상(-1)
classes = np.unique(np.concatenate((y_train, y_valid, y_test), axis=0))

x = np.arange(len(classes))  # Plot의 x축 개수 구하기
labels = ['Normal', 'Abnormal'] # Plot의 x축 이름 지정

values_train = [(y_train == i).sum() for i in classes]  # Train 데이터의 정상/비정상 개수 세기
values_valid = [(y_valid == i).sum() for i in classes]  # Validation 데이터의 정상/비정상 개수 세기
values_test = [(y_test == i).sum() for i in classes]    # Test 데이터의 정상/비정상 개수 세기

plt.figure(figsize=(8, 4))   # Plot 틀(Figure)의 Size 설정(8x4)

plt.subplot(1, 3, 1)   # Plot 틀(Figure) 내 3개의 subplot 중 첫 번째(왼쪽) 지정
plt.title("Training Data")  # Subplot 제목
plt.bar(x, values_train,  width=0.6, color=["red", "blue"])  # Train 데이터 정상/비정상 막대 그래프
plt.ylim([0,1500])
plt.xticks(x, labels)  # x축 눈금 이름 지정

plt.subplot(1, 3, 2)   # Plot 틀(Figure) 내 3개의 subplot 중 두 번째(가운데) 지정
plt.title("Validation Data")  # Subplot 제목
plt.bar(x, values_valid,  width=0.6, color=["red", "blue"])  # Validation 데이터 정상/비정상 막대 그래프
plt.ylim([0,1500])
plt.xticks(x, labels)  # x축 눈금 이름 지정

plt.subplot(1, 3, 3)   # Plot 틀(Figure) 내 3개의 subplot 중 세 번째(오른쪽) 지정
plt.title("Test Data")  # Subplot 제목
plt.bar(x, values_test,  width=0.6, color=["red", "blue"])  # Test 데이터 정상/비정상 막대 그래프
plt.ylim([0,1500])
plt.xticks(x, labels)  # x축 눈금 이름 지정

plt.tight_layout()  # Subplot 간 간격 자동 조정
img_save_path = './output/image/'
plt.savefig(img_save_path + 'data_distribution.png', dpi=100, bbox_inches='tight')  # Plot 이미지 저장
plt.show()  # Plot 화면에 출력

