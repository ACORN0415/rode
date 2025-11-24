import torch
import pandas as pd
import csv
import subprocess
import os
import time

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- 경로 설정 ---
current_dir = os.path.dirname(__file__)
# __file__이 정의되지 않은 환경(예: Jupyter)을 위해 예외 처리
if not current_dir:
    current_dir = os.getcwd()
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
dataset_base_dir = os.path.join(project_dir, "AAA", "RoDe", "dataset")

# --- 입력 파일 자동 생성 로직 ---
input_csv_dir = os.path.join(project_dir, 'result', 'ref')
input_csv_path = os.path.join(input_csv_dir, 'baseline_h100_spmm_128.csv')

# 1. 입력 파일이 존재하는지 확인
if not os.path.exists(input_csv_path):
    print(f"경고: 입력 파일({input_csv_path})을 찾을 수 없습니다.")
    print("시스템의 데이터셋 폴더를 스캔하여 파일을 자동으로 생성합니다...")
    
    categories = ['symmetric', 'unsymmetric', 'square', 'rectangular']
    found_datasets = []

    if not os.path.isdir(dataset_base_dir):
        print(f"오류: 데이터셋 폴더를 찾을 수 없습니다! 경로를 확인하세요: {dataset_base_dir}")
        exit() # 데이터셋 폴더가 없으면 실행 중지

    # 각 카테고리 폴더를 순회하며 데이터셋 이름 수집
    for category in categories:
        category_path = os.path.join(dataset_base_dir, category)
        if os.path.isdir(category_path):
            for dataset_name in os.listdir(category_path):
                if os.path.isdir(os.path.join(category_path, dataset_name)):
                    found_datasets.append(dataset_name)
    
    if found_datasets:
        os.makedirs(input_csv_dir, exist_ok=True) # 출력 폴더 생성
        # CSV 파일 작성
        with open(input_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['dataSet']) # 헤더
            for name in sorted(found_datasets):
                writer.writerow([name])
        print(f"성공: {len(found_datasets)}개의 데이터셋을 찾아 입력 파일을 생성했습니다.")
    else:
        print("오류: 데이터셋을 하나도 찾지 못하여 실행을 중지합니다.")
        exit()

# 2. 데이터셋 목록 로드
df = pd.read_csv(input_csv_path)

# --- 출력 파일 설정 ---
file_name = os.path.join(project_dir, 'AAA', 'results', 'residual_spmm_f32_n128.csv')
head = ['dataSet', 'category','sputnik','Sputnik_gflops','cusparse','cuSPARSE_gflops','rode_stream','RoDe_gflops']

# ==================================================================
#               ★★★ 바로 이 부분이 핵심입니다 ★★★
# ==================================================================
# 출력 폴더가 없으면 자동으로 생성합니다.
output_dir = os.path.dirname(file_name)
os.makedirs(output_dir, exist_ok=True)
# ==================================================================
with open(file_name, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(head)

# --- 메인 로직 ---
count = 0
start_time = time.time()

for index, row in df.iterrows():
    count += 1
    dataset_name = row['dataSet']
    
    categories = ['symmetric', 'unsymmetric', 'square', 'rectangular']
    found_path = None
    found_category = None

    for category in categories:
        potential_path = os.path.join(dataset_base_dir, category, dataset_name, f"{dataset_name}.mtx")
        if os.path.exists(potential_path):
            found_path = potential_path
            found_category = category
            break

    if found_path:
        print(f"Processing ({count}/{len(df)}): {dataset_name} (found in '{found_category}')")
        with open(file_name, 'a', newline='') as csvfile:
            csvfile.write(f"{dataset_name},{found_category},")
        shell_command = f"{project_dir}/AAA/RoDe/build/eval/eval_spmm_f32_n128 {found_path} >> {file_name}"
        print(f"DEBUG: Executing -> {shell_command}\n")
        subprocess.run(shell_command, shell=True)
    else:
        print(f"Warning: .mtx file for '{dataset_name}' not found in any category. Skipping.")
        with open(file_name, 'a', newline='') as csvfile:
            csvfile.write(f"{dataset_name},NOT_FOUND,,,,,,,,,NOT_FOUND\n")

end_time = time.time()
execution_time = end_time - start_time

# --- 실행 시간 기록 ---
dimN = 128
with open("execution_time_base.txt", "a") as file:
    file.write(f"spmm-{dimN}-{execution_time/60:.2f} minutes\n")

print(f"\nTotal execution time: {execution_time/60:.2f} minutes")