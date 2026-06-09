

bash ex0.sh
bash ex0_b0.sh 2>&1 | tee log/eval_b0.log
bash ex1_b1.sh 2>&1 | tee log/eval_b1.log
bash ex1_p1.sh 2>&1 | tee log/eval_p1.log
bash ex2_fr.sh 2>&1 | tee log/eval_fr.log
bash ex3_sin.sh 2>&1 | tee log/eval_sin.log
bash ex3_cos.sh 2>&1 | tee log/eval_cos.log
bash ex4_1e6.sh 2>&1 | tee log/eval_1e6.log
bash ex4_1e7.sh 2>&1 | tee log/eval_1e7.log
bash ex4_1e8.sh 2>&1 | tee log/eval_1e8.log



python exlog.py




