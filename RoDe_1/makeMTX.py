import numpy as np
from scipy.sparse import rand
from scipy.io import mmwrite
import os # <-- os 모듈 추가

# --- 설정 ---
M = 200
N = 200
density = 0.1
# 파일 경로의 디렉토리 부분을 변수로 분리
output_dir = "/home/acorn0415/AAA/RoDe/random_matrix"
# -------------

# 💡 디렉토리가 없으면 생성 (재귀적으로 생성)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"알림: 디렉토리 '{output_dir}'를 생성했습니다.")
# 100개의 mtx 파일 생성
for i in range(100):
    # 희소 행렬 생성
    A = rand(M, N, density=density, format='csr', dtype=np.float32)
    # (선택 사항) 값의 범위를 1~10으로 설정
    A.data = np.random.uniform(1.0, 10.0, size=A.nnz)
    # 행렬을 Matrix Market (MTX) 파일로 저장
    output_filename = os.path.join(output_dir, f"random_200x200_sparse90_{i+1}.mtx")
    mmwrite(output_filename, A)
    try:
        print(f"성공: '{output_filename}' 파일이 생성되었습니다.")
        print(f"크기: {M}x{N}, 0이 아닌 요소(NNZ): {A.nnz}")

    except Exception as e:
        print(f"오류 발생: {e}")

# # 64 * 64 * 0.3 = 1228.8 이므로, scipy가 반올림하여 1229개의 NNZ를 생성합니다.
# A = rand(M, N, density=density, format='csr', dtype=np.float32)

# # (선택 사항) 값의 범위를 1~10으로 설정
# A.data = np.random.uniform(1.0, 10.0, size=A.nnz)

# try:
#     # 행렬을 Matrix Market (MTX) 파일로 저장
#     mmwrite(filename, A)
    
#     print(f"성공: '{filename}' 파일이 생성되었습니다.")
#     print(f"크기: {M}x{N}, 0이 아닌 요소(NNZ): {A.nnz}")
    
#     # 실제 희소성 계산
#     actual_sparsity = 1.0 - (A.nnz / (M * N))
#     print(f"실제 희소성(Sparsity): {actual_sparsity * 100:.2f}%")

# except Exception as e:
#     print(f"오류 발생: {e}")