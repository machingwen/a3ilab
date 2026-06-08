

# my project



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

## Experiment 1: 

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

## Experiment 2: 

Cdf

```
bash ex1_p1.sh 2>&1 | tee log/ex1_p1.log
```

Activity-Freq
```
bash ex2_fr.sh 2>&1 | tee log/eval_fr.log
```

## Experiment 3: 

linear
```
bash ex1_p1.sh 2>&1 | tee log/ex1_p1.log
```

sin and cos

```
bash ex3_sin.sh 2>&1 | tee log/eval_sin.log
bash ex3_cos.sh 2>&1 | tee log/eval_cos.log
```

## Experiment 4: 

l1 method
```
bash ex4_1e6.sh 2>&1 | tee log/eval_1e6.log
bash ex4_1e7.sh 2>&1 | tee log/eval_1e7.log
bash ex4_1e8.sh 2>&1 | tee log/eval_1e8.log
```

## show easy read result (after running all experiments)


```
python exlog.py
```

## run all codes with one script

```
bash runall.sh
```




