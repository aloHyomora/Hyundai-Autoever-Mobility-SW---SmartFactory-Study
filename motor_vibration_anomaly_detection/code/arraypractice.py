import numpy as np

np.set_printoptions(linewidth=120)  # 출력이 한 줄에 더 잘 보이게

print("===== 1) 1차원 배열(1D) 만들기 =====")
a = np.array([10, 11, 12, 13, 14, 15, 16])
print("a =", a)
print("a.shape =", a.shape)

print("\n===== 2) 1차원 슬라이싱 =====")
print("a[0] =", a[0])                 # 첫 원소
print("a[-1] =", a[-1])               # 마지막 원소
print("a[1:4] =", a[1:4])             # 인덱스 1~3
print("a[:3] =", a[:3])               # 처음 3개
print("a[3:] =", a[3:])               # 3부터 끝까지
print("a[::2] =", a[::2])             # 2칸씩
print("a[::-1] =", a[::-1])           # 뒤집기

print("\n===== 3) 1차원: 조건(불리언) 인덱싱 =====")
mask = a >= 14
print("mask (a>=14) =", mask)
print("a[mask] =", a[mask])

print("\n===== 4) 1차원: fancy indexing(인덱스 리스트로 선택) =====")
idx = [0, 2, 5]
print("idx =", idx)
print("a[idx] =", a[idx])

print("\n===== 5) 1차원: 반복문으로 새 배열 채우기 =====")
b = np.zeros_like(a)  # a와 같은 shape의 0 배열
for i in range(len(a)):
    b[i] = a[i] * 2 + 1
print("b (loop fill) =", b)

print("\n===== 6) 1차원: 벡터화(vectorized)로 새 배열 만들기(반복문 없이) =====")
b2 = a * 2 + 1
print("b2 (vectorized) =", b2)
print("b == b2 ?", np.array_equal(b, b2))

print("\n\n===== 7) 2차원 배열(2D) 만들기 =====")
A = np.arange(1, 13).reshape(3, 4)  # 3행 4열
print("A =\n", A)
print("A.shape =", A.shape)

print("\n===== 8) 2차원 기본 인덱싱/슬라이싱 =====")
print("A[0, 0] =", A[0, 0])          # (0행,0열)
print("A[0, :] =", A[0, :])          # 0행 전체 (1D)
print("A[:, 1] =", A[:, 1])          # 1열 전체 (1D)
print("A[1:3, 1:3] =\n", A[1:3, 1:3]) # 부분 행렬
print("A[:, ::2] =\n", A[:, ::2])    # 열을 2칸씩

print("\n===== 9) 2차원: 행/열 선택 결과 shape 차이 =====")
row0 = A[0, :]     # (4,) 1D
col0 = A[:, 0]     # (3,) 1D
col0_2d = A[:, 0:1]  # (3,1) 2D로 유지
print("row0 =", row0, "shape=", row0.shape)
print("col0 =", col0, "shape=", col0.shape)
print("col0_2d =\n", col0_2d, "shape=", col0_2d.shape)

print("\n===== 10) 2차원: 조건(불리언) 인덱싱 =====")
maskA = A % 2 == 0
print("maskA (A%2==0) =\n", maskA)
print("A[maskA] =", A[maskA])  # 2D가 아니라 '조건에 맞는 원소들만' 1D로 펼쳐져 나옴

print("\n===== 11) 2차원: 반복문으로 새 2D 배열 채우기(원소 단위) =====")
B = np.zeros_like(A)
for r in range(A.shape[0]):
    for c in range(A.shape[1]):
        B[r, c] = A[r, c] * 10
print("B (loop fill) =\n", B)

print("\n===== 12) 2차원: 반복문으로 '열 단위'로 채우기 (data2d[:, i] 느낌) =====")
# C의 각 열 i에 (A의 i번째 열 + i) 같은 값을 넣어보기
C = np.zeros_like(A)
for i in range(A.shape[1]):          # i는 열 인덱스
    C[:, i] = A[:, i] + i            # i번째 열 전체에 한 번에 대입
print("C (column-wise fill) =\n", C)

print("\n===== 13) 2차원: fancy indexing (행/열 일부만 선택) =====")
rows = [0, 2]
cols = [1, 3]
print("A[rows] =\n", A[rows])  # 행 0,2만
print("A[:, cols] =\n", A[:, cols])  # 열 1,3만
print("A[np.ix_(rows, cols)] =\n", A[np.ix_(rows, cols)])  # (0,2)행과 (1,3)열 교차 부분행렬

print("\n===== 14) reshape / ravel / flatten =====")
flat = A.ravel()      # 보통 view(가능하면)로 1D
flat2 = A.flatten()   # 항상 복사본
print("A.ravel() =", flat, "shape=", flat.shape)
print("A.flatten() =", flat2, "shape=", flat2.shape)

print("\nDONE.")