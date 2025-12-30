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
print(torch.__version__)

# ==============================
# Data import
# 데이터 불러오기
# ==============================
file_path = '/dataset/'   # Dataset directory (데이터셋 경로)
train_fn = 'FordA_TRAIN.arff'  # Training data file (학습 데이터 파일)
test_fn = 'FordA_TEST.arff'    # Test data file (테스트 데이터 파일)