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

# ==============================
# Data Visualization
# 데이터 시각화 2-1: 특정 시간에서의 시계열 샘플을 플롯
# ==============================

import random

labels = np.unique(np.concatenate((y_train,y_valid,y_test), axis=0))    # 클래스 종류 확인: 정상(1), 비정상(-1)

plt.figure(figsize=(10, 4))   # Plot 틀(Figure)의 Size 설정(10x4)

for c in labels:
    c_x_train = x_train[y_train == c]   # 클래스 c에 해당하는 x_train 샘플들
    if c == -1: c = c + 1   # 편의 상 Abnormal -> 0으로 변경
    time_t = random.randint(0, c_x_train.shape[0]) # 0~샘플 개수 사이에서 랜덤한 인덱스 선택
    plt.scatter(range(0, 500), c_x_train[time_t], label=f'class = {str(int(c))}', marker='o', s=5)  # 클래스 c의 랜덤 샘플 플롯

plt.legend(loc="lower right")  # 범례 표시
plt.xlabel("Sensor", fontsize=15)
plt.ylabel("Sensor Value", fontsize=15)
plt.savefig(img_save_path + 'ford_data_ts_sample1.png', dpi=100, bbox_inches='tight')  # Plot 이미지 저장
plt.show()  # Plot 화면에 출력

# ==============================
# Data Visualization
# 데이터 시각화 2-2: 특정 시간에서의 시계열 샘플을 플롯
# ==============================

def get_scatter_plot(c):
    time_t = random.randint(0, c_x_train.shape[0]) # 0~샘플 개수 사이에서 랜덤한 인덱스 선택
    plt.scatter(range(0, c_x_train.shape[1]), c_x_train[time_t], marker='o', s=5, c="r" if c == -1 else "b")  # 클래스 c의 랜덤 샘플 플롯
    plt.title("at time: t_{}".format(time_t), fontsize=20)
    plt.xlabel("Sensor", fontsize=14)
    plt.ylabel("Sensor Value", fontsize=14)
    plt.savefig(img_save_path + '{state}.png'.format(state="abnormal" if c == -1 else "normal"), dpi=100, bbox_inches='tight')  # Plot 이미지 저장
    plt.show()  # Plot 화면에 출력

labels = np.unique(np.concatenate((y_train,y_valid,y_test), axis=0))    # 클래스 종류 확인: 정상(1), 비정상(-1)

for c in labels:
    c_x_train = x_train[y_train == c]   # 클래스 c에 해당하는 x_train 샘플들
    if c == -1:
        print("비정상 Label 데이터 수: ", len(c_x_train))
        get_scatter_plot(c)
    else:
        print("정상 Label 데이터 수: ", len(c_x_train))
        get_scatter_plot(c)

# ==============================
# Data Visualization
# 데이터 시각화 3: 1개의 임의의 센서 값을 시계열을 플롯
# ==============================

sensor_number = random.randint(0, 500)  # 0~499 사이에서 랜덤한 센서 번호 선택
plt.figure(figsize=(13, 4))   # Plot 틀(Figure)의 Size 설정(13x4)
plt.title("Sensor number: {}".format(sensor_number), fontsize=20)
plt.plot(x_train[:, sensor_number])
plt.xlabel("Time", fontsize=15)
plt.ylabel("Sensor Value", fontsize=15)
plt.savefig(img_save_path + 'ford_data_ts_sensor{}.png'.format(sensor_number), dpi=100, bbox_inches='tight')  # Plot 이미지 저장
plt.show()  # Plot 화면에 출력

# ==============================
# 데이터 상관관계 분석
# Data Correlation Analysis
# ==============================

import matplotlib.cm as cm  # Colormap handling (컬러맵 처리)
from matplotlib.collections import EllipseCollection  # Ellipse collection for correlation plot (상관관계 플롯용 타원 컬렉션)


df = pd.DataFrame(data=x_train, columns=["sensor_{}".format(label + 1) for label in range(x_train.shape[1])])  # DataFrame 생성, 열 이름 지정
data = df.corr() # 상관관계 행렬 계산(원리: 피어슨 상관계수)

def plot_corr_ellipes(data, ax=None, **kwargs):
    M = np.array(data)
    if not M.ndim == 2: # 2차원 배열인지 확인
        raise ValueError("Data must be 2-dimensional")
    if ax is None:      # ax가 주어지지 않으면 새로 생성
        fig, ax = plt.subplots(1, 1, subplot_kw={'aspect': 'equal'})   # 동일한 비율의 축 설정
        ax.set_xlim(-0.5, M.shape[1] - 0.5) # x축 한계 설정
        ax.set_ylim(-0.5, M.shape[0] - 0.5) # y축 한계 설정

    xy = np.indices(M.shape)[::-1].reshape(2, -1).T  # 좌표 생성 및 재구성

    w = np.ones_like(M).ravel() # 타원 너비 설정
    h = 1 - np.abs(M).ravel()  # 타원 높이 설정
    a = 45 * np.sign(M).ravel() # 타원 각도 설정

    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x',
                           offsets=xy, transOffset=ax.transData,array=M.ravel(), **kwargs)  # 타원 컬렉션 생성
    ax.add_collection(ec)   # 타원 컬렉션 추가

    if isinstance(data, pd.DataFrame):  # DataFrame인 경우 축 눈금 설정
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)
    return ec

fig, ax = plt.subplots(1, 1, figsize=(20, 20))  # Plot 틀(Figure) 및 축(Axis) 생성
cmap = cm.get_cmap('jet', 31)  # Colormap 설정
m = plot_corr_ellipes(data, ax=ax, cmap=cmap)  # 상관관계 타원 플롯 생성
cb = fig.colorbar(m)  # Colorbar 추가
cb.set_label('Correlation coefficient')
plt.title('Correlation between Feature')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.tight_layout()
plt.savefig(img_save_path + 'corr.png', dpi=100, bbox_inches='tight')  # Plot 이미지 저장
plt.show()  # Plot 화면에 출력

# ==============================
# 데이터 정규화
# Data Normalization
# ==============================

from sklearn.preprocessing import StandardScaler, RobustScaler

# Standard Scaler를 적용하고 싶은 경우 아래 Code를 실행
stder = StandardScaler()
stder.fit(x_train)   # 학습 데이터로 Scaler 학습
x_train = stder.transform(x_train)   # 학습 데이터에 Scaler 적용
x_valid = stder.transform(x_valid)   # 검증 데이터에 Scaler 적용

# Robust Scaler를 적용하고 싶은 경우 아래 Code를 실행
# rscaler = RobustScaler()
# rscaler.fit(x_train)   # 학습 데이터로 Scaler 학습
# x_train = rscaler.transform(x_train)   # 학습 데이터에 Scaler 적용
# x_valid = rscaler.transform(x_valid)   # 검증 데이터에 Scaler 적용

# CNN, RNN 모델의 입력 변수는 채널 축 1개 차원을 확장시킨 3D Tensor이다.
# 축 아래 CNN, RNN 모델은 x_train, x_valid, x_test를 입력 변수로 넣지 않고,
# x_train_exp, x_valid_exp, x_test_exp를 입력 변수로 넣는다.
x_train_exp = np.expand_dims(x_train, -1)   # 채널 축
x_valid_exp = np.expand_dims(x_valid, -1)   # 채널 축
x_test_exp = np.expand_dims(x_test, -1)     # 채널 축

print("\nData shapes after normalization and dimension expansion:")
print("x_train_exp shape:", x_train_exp.shape)
print("x_valid_exp shape:", x_valid_exp.shape)
print("x_test_exp shape:", x_test_exp.shape)

# ==============================
# 종속 변수(y)를 양의 값으로 변경
# Change dependent variable (y) to positive values
# ==============================

y_train[y_train == -1] = 0   # y_train의 -1 값을 0으로 변경
y_valid[y_valid == -1] = 0   # y_valid의 -1 값을 0으로 변경
y_test[y_test == -1] = 0     # y_test의 -1 값을 0으로 변경