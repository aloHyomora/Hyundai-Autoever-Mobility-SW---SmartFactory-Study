import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# data.xlsx 경로
data_file_path = 'data.xlsx'

def read_data(file_path, sheet_name):
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        sys.exit(1)
    
    try:
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        return data
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        sys.exit(1)

# 기초 통계량 계산 함수
# 기초 통계량 항목: 평균, 중앙값, 최빈값, 분산, 표준편차, 최소값, 최대값, 사분위수

# 동일한 "Date"&"LoT" 기준 그룹화하기
def group_data(data):
    grouped = data.groupby(['Date', 'LoT'])
    return grouped

# 기초 통계량 (동일한 "Date"&"LoT" 기준 그룹화된 데이터)
def grou_data(grouped_data):
    selected_columns = grouped_data[['pH_Standard', 'Temperature_Standard']]
    print(selected_columns.head())
    # "pH_Standard, Temperature_Standard" 컬럼에 대해서 기초 통계량 계산(mean, var, std, min, max, 1st quartile, 3rd quartile)
    # summary = selected_columns.
    return 0

# 전달받은 컬럼에 대해 히스토그램 그리기(pH_Standard_mean histogram or Temperature_Standard_mean histogram)
def plot_histogram(data, column_name):
    plt.figure(figsize=(10, 6))
    plt.hist(data[column_name].dropna(), bins=30, color='blue', alpha=0.7)
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

if __name__ == "__main__":
    data = read_data(data_file_path, "data")
    print("Data Summary:")
    
    grouped_data = data.groupby(['Date', 'LoT'])[['pH_Standard', 'Temperature_Standard']]
    
    # 각 그룹의 데이터 중 mean값만 구해서 print
    summary = grouped_data.mean().unstack(level=-1)
    print(summary.head())
    #print(summary)
    # plot_histogram(summary, ('pH_Standard', 'mean'))
    # plot_histogram(summary, ('Temperature_Standard', 'mean'))

    

    
    