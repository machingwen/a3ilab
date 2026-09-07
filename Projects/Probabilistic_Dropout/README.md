

# Neuron-Activity-Aware Fine-Tuning for Large Language Models: Enhancing the Sparsity–Performance Trade-off



## Overview

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

### Training with Our Proposed Activity-Aware L1 Regularization 
```
bash np.sh 2>&1  | grep -v "Running loglikelihood requests" | tee log/eval_np.log
```
### Training with Conventional L1 Regularization (old L1 Regularization) 
```
bash ex4_l1.sh 2>&1| grep -v "Running loglikelihood requests" | tee log/eval_l1.log
```
### Training with Baseline (fine-tuned without activation L1 regularization) 
```
bash ex0_b0.sh 2>&1 | grep -v "Running loglikelihood requests" | tee log/eval_b0.log
```

### Evaluate Existing Models about L1-regularized 

Evaluate the baseline and the two L1-regularized models:

```
mkdir -p log

# Baseline without activation L1 regularization
bash eval_np.sh no_l1 2>&1 | tee log/eval_b0.log

# Conventional activation L1 regularization
bash eval_np.sh old_l1 2>&1 | tee log/eval_l1.log

# Our Proposed Activity-Aware L1 Regularization
bash eval_np.sh my_l1 2>&1 | tee log/eval_np.log

# Summarize the evaluation results
python exlog_np.py
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


### Evaluate Existing Models about Probabilistic Dropout



```
mkdir -p log

# Probabilistic Dropout with CDF-based linear mapping on SIQA
bash eval.sh ex1_p1 2>&1 | tee log/eval_p1.log

# Uniform Dropout baseline on SIQA
bash eval.sh ex1_b1 2>&1 | tee log/eval_b1.log

# Probabilistic Dropout with CDF-based linear mapping on PIQA
bash eval_piqa.sh ex1_p1 2>&1 | tee log/eval_p1_piqa.log

# Uniform Dropout baseline on PIQA
bash eval_piqa.sh ex1_b1 2>&1 | tee log/eval_b1_piqa.log

# Summarize the evaluation results
python exlog.py
```




