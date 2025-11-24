#!/bin/bash

# ../dataset 폴더 생성 및 이동
mkdir ../dataset
cd ../dataset

# --- 1. Symmetric (대칭 행렬) 처리 ---
echo "Processing Symmetric matrices..."
mkdir symmetric
cd symmetric
wget http://www.cise.ufl.edu/research/sparse/MM/FIDAP/ex5.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Oberwolfach/LF10.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Oberwolfach/LFAT5.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/HB/bcsstk01.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Pothen/mesh1em1.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Pajek/GD97_b.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Bai/bfwb62.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/HB/bcsstk03.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/HB/plat362.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Boeing/bcsstk34.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Nasa/nasa2910.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/ND/nd3k.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/GHS_indef/ncvxqp9.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Schenk_IBMNA/c-67.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/UTEP/Dubcova2.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Koutsovasilis/F2.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/MaxPlanck/shallow_water1.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/GHS_indef/ncvxqp7.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/GHS_indef/c-59.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Newman/cond-mat-2005.tar.gz

echo "Extracting and cleaning up symmetric matrices..."
# 현재 폴더의 모든 tar.gz 파일에 대해 압축 해제 실행
for f in *.tar.gz; do tar xvf "$f"; done
# 압축 해제가 끝난 원본 .tar.gz 파일 삭제
rm *.tar.gz
cd ..

# --- 2. Unsymmetric (비대칭 행렬) 처리 ---
echo "Processing Unsymmetric matrices..."
mkdir unsymmetric
cd ./unsymmetric
wget http://www.cise.ufl.edu/research/sparse/MM/vanHeukelum/cage5.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Rommes/ww_36_pmec_36.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Bai/bfwa62.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Bai/tols90.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Bai/tub100.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Morandini/robot.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/FIDAP/ex1.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/HB/west0381.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Sandia/fpga_dcop_43.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Sandia/adder_dcop_58.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Sandia/adder_dcop_47.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Zitney/hydr1.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/HB/lns_3937.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Fluorem/GT01R.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Hollinger/mark3jac020sc.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Oberwolfach/flowmeter5.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Bai/cryg10000.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Mallya/lhr11c.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Hollinger/g7jac040sc.tar.gz

echo "Extracting and cleaning up unsymmetric matrices..."
for f in *.tar.gz; do tar xvf "$f"; done
rm *.tar.gz
cd ..

# --- 3. Square (정방 행렬) 처리 ---
echo "Processing Square matrices..."
mkdir square
cd ./square
wget http://www.cise.ufl.edu/research/sparse/MM/Pajek/Tina_AskCog.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Pajek/Ragusa16.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Pajek/GD01_c.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/HB/bcspwr01.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Pajek/GD99_b.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/HB/bcsstm02.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/HB/west0156.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Sandia/oscil_dcop_49.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/HB/hor_131.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Bai/rbsa480.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Gset/G14.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Sandia/fpga_trans_02.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/HB/pores_2.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/HB/bcsstm12.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Sandia/adder_dcop_17.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Bai/rdb2048_noL.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Nasa/nasa2146.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/PARSEC/SiNa.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Nemeth/nemeth15.tar.gz

echo "Extracting and cleaning up square matrices..."
for f in *.tar.gz; do tar xvf "$f"; done
rm *.tar.gz
cd ..

# --- 4. Rectangular (직사각 행렬) 처리 ---
echo "Processing Rectangular matrices..."
mkdir rectangular
cd rectangular
wget http://www.cise.ufl.edu/research/sparse/MM/Bai/dw2048.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/FIDAP/ex20.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Norris/heart3.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/LPnetlib/lp_osa_14.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Meszaros/l30.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/LPnetlib/lp_cre_c.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Meszaros/delf.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Meszaros/dano3mip.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Meszaros/deter4.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Meszaros/large.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Meszaros/deter5.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Meszaros/cq5.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Meszaros/r05.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Meszaros/stat96v1.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Meszaros/nl.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Meszaros/nemsemm2.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Meszaros/deter3.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/MathWorks/Pd_rhs.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Meszaros/scsd8-2r.tar.gz
wget http://www.cise.ufl.edu/research/sparse/MM/Meszaros/cq9.tar.gz

echo "Extracting and cleaning up rectangular matrices..."
for f in *.tar.gz; do tar xvf "$f"; done
rm *.tar.gz
cd ..


# --------------------------------------------------------------------
# 아래의 .mtx 파일 변환 로직은 기존 스크립트와 동일하게 유지됩니다.
# --------------------------------------------------------------------
#cp ../conv.c .
gcc -O3 -o conv conv.c

for i in `ls -d */`
do
cd ${i}
ii=${i/\//}
mv ${ii}.mtx ${ii}.mt0
../conv ${ii}.mt0 ${ii}.mtx 
rm ${ii}.mt0
cd ..
done

rm conv
rm conv.c
cd ..