#!/bin/bash

# 프로파일링할 목표 커널 이름
TARGET_KERNEL="RoDeComputeKernel1"

# 이 스크립트로 전달된 모든 인자 (예: --dataset Dubcova2 ...)
ALL_ARGS=("$@")

# 원래 실행해야 할 프로그램 경로
# 이 경로는 실제 build 디렉토리의 실행 파일을 가리켜야 합니다.
EXECUTABLE_PATH="../build/eval/eval_spmm_f32_n128" 

# 먼저, list-kernels로 어떤 커널이 실행될지 확인
KERNEL_LIST=$(ncu --list-kernels --target-processes all $EXECUTABLE_PATH "${ALL_ARGS[@]}" 2>&1)

# grep으로 목표 커널이 있는지 확인
if echo "$KERNEL_LIST" | grep -q "$TARGET_KERNEL"; then
    echo "++++ Found target kernel '$TARGET_KERNEL'. Starting NCU profiling... ++++"
    
    # 목표 커널이 있으므로, ncu로 실제 프로파일링 실행
    ncu --kernel-name "$TARGET_KERNEL" --metrics achieved_occupancy --target-processes all $EXECUTABLE_PATH "${ALL_ARGS[@]}"

else
    echo "---- Target kernel '$TARGET_KERNEL' not found for these args. Running normally... ----"
    
    # 목표 커널이 없으므로, 프로파일링 없이 그냥 프로그램만 실행
    $EXECUTABLE_PATH "${ALL_ARGS[@]}"
fi