import os
import csv
import subprocess
import time
import scipy.sparse
import scipy.io
import numpy as np

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 현재 디렉터리와 프로젝트 디렉터리 경로 설정
current_dir = os.path.dirname(__file__)
if not current_dir:
    current_dir = os.getcwd()
project_dir = "/home/acorn0415/AAA/RoDe"  # 변경된 경로

# 랜덤 sparse matrix 생성 파라미터
m, n = 1024, 1024  # 행, 열 크기
density = 0.3  # 희소 행렬 비율

# scipy.sparse로 랜덤 희소 행렬 생성 (COO 포맷)
random_sparse = scipy.sparse.random(m, n, density=density, format='csr', dtype=np.float32)

# 임시 mtx 파일 경로 및 생성
tmp_dir = os.path.join(project_dir, 'tmp')
os.makedirs(tmp_dir, exist_ok=True)
tmp_mtx_path = os.path.join(tmp_dir, 'random_sparse.mtx')

# Matrix Market 형식으로 저장
scipy.io.mmwrite(tmp_mtx_path, random_sparse)

# 결과 CSV 파일 경로 (RoDe 폴더 내부 result 폴더 under Baseline)
file_name = os.path.join(project_dir, 'results', 'spmm', 'random_test_result.csv')
head = ['dataSet','rows_','columns_','nonzeros_','sputnik','Sputnik_gflops','cusparse','cuSPARSE_gflops','rode','ours_gflops', 'diff']
os.makedirs(os.path.dirname(file_name), exist_ok=True)

os.makedirs(os.path.dirname(file_name), exist_ok=True)
with open(file_name, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(head)

# evaluate 실행 명령 및 시간 측정
eval_exec = os.path.join(project_dir, 'build', 'eval', 'eval_spmm_f32_n128')
shell_command = f"{eval_exec} {tmp_mtx_path} >> {file_name}"

start_time = time.time()
print(f"Running test on random sparse matrix: {m}x{n}, density={density}")

# 외부 바이너리 실행
subprocess.run([eval_exec, tmp_mtx_path], timeout=10)

end_time = time.time()
execution_time = end_time - start_time

# 시간 기록
with open("execution_time_base.txt", "a") as file:
    file.write(f"random_sparse-{m}x{n}-{execution_time/60:.2f} minutes\n")

print(f"Test complete. Execution time: {execution_time/60:.2f} minutes")
