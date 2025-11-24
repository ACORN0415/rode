import os
import pandas as pd
import csv
import subprocess
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

# dataset 폴더 내 모든 dataSet 폴더 리스트 획득
dataset_base = os.path.join(project_dir, 'ZZZZZ', 'RoDe', 'dataSet')
dataSet_list = [d for d in os.listdir(dataset_base) if os.path.isdir(os.path.join(dataset_base, d))]

# 결과 CSV 파일 경로 및 헤더
result_csv = os.path.join(project_dir, 'results', 'alldata', 'spmm', 'rode_spmm_f32_n128_streamRode.csv')
header = ['dataSet','cusparse','cuSPARSE_gflops','streamk','streamk_gflops']

# 결과 CSV 파일에 헤더 작성
os.makedirs(os.path.dirname(result_csv), exist_ok=True)
with open(result_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

start_time = time.time()
count = 0

for dataSet in dataSet_list:
    mtx_path = os.path.join(dataset_base, dataSet, f"{dataSet}.mtx")
    if not os.path.isfile(mtx_path):
        print(f"Skipped: {mtx_path} 파일이 존재하지 않습니다.")
        continue

    count += 1
    print(f"Processing dataset {count}: {dataSet}")

    # dataSet명 먼저 쓰기 (append 모드)
    with open(result_csv, 'a', newline='') as f:
        f.write(f"{dataSet},")

    # eval_spmm_f32_n128 실행 및 결과를 result_csv에 append
    eval_exec = os.path.join(project_dir, 'ZZZZZ', 'RoDe', 'build', 'eval', 'eval_spmm_f32_n128')
    cmd = [eval_exec, mtx_path]
    with open(result_csv, 'a') as f:
        subprocess.run(cmd, stdout=f)

end_time = time.time()
exec_minutes = round((end_time - start_time) / 60, 2)

with open(os.path.join(project_dir, "execution_time_base.txt"), "a") as f:
    f.write(f"spmm-128-{exec_minutes} minutes\n")

print("All datasets processed.")
