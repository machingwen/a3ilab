

# Neuron-Activity-Aware Fine-Tuning for Large Language Models: Enhancing the Sparsity–Performance Trade-of



## overview

All experiments can be executed using the following logging format:

```
bash <script>.sh 2>&1 | tee log/<script>.log
```

This ensures both terminal output and logs are saved.

## Pre-run / Initialization

This step must be executed before running any experiments:

```
bash ex0.sh 
```
## Activity-Aware L1 Regularization

### Our Proposed Activity-Aware L1 Regularization 
```
bash np.sh 2>&1  | grep -v "Running loglikelihood requests" | tee log/eval_np.log
```
### Conventional L1 Regularization (old L1 Regularization) 
```
bash ex4_l1.sh 2>&1| grep -v "Running loglikelihood requests" | tee log/eval_l1.log
```
### Baseline ( fine-tuned without activation L1 regularization) 
```
bash ex0_b0.sh 2>&1 | grep -v "Running loglikelihood requests" | tee log/eval_b0.log
```


## Probabilistic Dropout 

### Experiment 1: 

old Baseline(no dropout)

```
bash ex0_b0.sh 2>&1 | tee log/eval_b0.log
```


Baseline(normal dropout)
```
bash ex1_b1.sh 2>&1 | tee log/ex1_b1.log
```

Probabilistic dropout (cdf, linear)

```
bash ex1_p1.sh 2>&1 | tee log/ex1_p1.log
```

### Experiment 2: 

Cdf

```
bash ex1_p1.sh 2>&1 | tee log/ex1_p1.log
```

Activity-Freq
```
bash ex2_fr.sh 2>&1 | tee log/eval_fr.log
```

### Experiment 3: 

linear
```
bash ex1_p1.sh 2>&1 | tee log/ex1_p1.log
```

sin and cos

```
bash ex3_sin.sh 2>&1 | tee log/eval_sin.log
bash ex3_cos.sh 2>&1 | tee log/eval_cos.log
```


###  show easy read result (after running all experiments)
```
python exlog.py
```





