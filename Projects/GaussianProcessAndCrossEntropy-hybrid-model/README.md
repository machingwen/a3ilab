---
title: PCB-Defect-Classfication

---

# PCB-Defect-Classfication
## Abstract
 **Method 1 : A modified hybrid expert, incorporating both multi-defect-type and multi-component-type classifier:**

In expert1,  we implement binary-class (good/bad) MLP classifier into 4-class(good/shift/broke/short) MLP classifier ,such that we can classify multi-defect types.
In Expert 2, we only use good samples for component classification, using the uncertainty of the gaussian process classifier for each component type to determine true/false/indeterminate.


 **Method 2 :One-vs-all classification framwork,3 hybrid experts (each expert incorporating both single-defect-type and multi-component-type classifier):**
 
Based on the concept of one-vs-all, we train 3 hybrid experts for guided defect detector (shift/not shift, broke/not broke, short/not short) and merge the 3 to get the classification result of defect types.

## Requirement
```
# Python 3.8: https://tech.serhatteker.com/post/2019-12/upgrade-python38-on-ubuntu/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -U scikit-learn
pip install tensorboard tensorboardX selenium twder beautifulsoup4 seaborn thop tqdm pytorch_metric_learning openpyxl natsort tensorboard_logger opencv-python pandas seaborn 
pip install numpy==1.23.5
pip install bayesian-optimization==1.4.2
pip install pycave==3.1.3
pip install ipykernel --upgrade
python3 -m ipykernel install --user
pip install gpytorch==1.8.1
pip install pytorch-ignite
pip uninstall horovod -y
HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]
pip uninstall pandas 
pip install pandas==1.5.3
cd ./clustimage_phison

pip install -r requirements.txt 
```

## Training 

- **Stage 1 : Hybrid Expert model training:**
```
python train_CEDUE.py --output_dir /root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/CEdue --seed $seed
```
- **Stage 2 : Stage2 model training**
```
python train_CEDUE_stage2.py --output_dir /root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/CEdue_s2 --checkpoint_path $ckpt --seed $seed
```
- **Plotting t-SNE for Hybrid Expert Model and GMM Model**
```
python tsne_CEDUE.py --embedding_layer shared_embedding --checkpoint_path $ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed
```
- **Bayesian optimization**
```
python bay_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed 
```
- **Inference**
```
python uncertainty_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed

python inference_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed
```

## Hybrid Expert model training script
- **method1:**
```shell
# stage 1
for seed in 2 211 1212 #1 2 3 #2 211 1212
do 
    CUDA_VISIBLE_DEVICES=0 python train_CEDUE.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/fourcls/$seed --seed $seed --dataset PHISON_fourcls
done

# stage 2
for seed in 2 211 1212 #1 2 3 #2 211 1212
do 
    ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/fourcls/$seed/ckpt.pt
    
    CUDA_VISIBLE_DEVICES=0 python train_CEDUE_stage2_gp.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/fourcls/$seed --checkpoint_path $ckpt --seed $seed --dataset PHISON_fourcls
    CUDA_VISIBLE_DEVICES=0 python train_CEDUE_stage2_mlp.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/MLP/fourcls/$seed --checkpoint_path $ckpt --seed $seed --dataset PHISON_fourcls
done
```
- **method2:**
```shell
# stage 1
for seed in 2 211 1212 #1 2 3 #2 211 1212
do 
    CUDA_VISIBLE_DEVICES=0 python train_CEDUE.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/shift/$seed --seed $seed --dataset PHISON_shift
    CUDA_VISIBLE_DEVICES=0 python train_CEDUE.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/broke/$seed --seed $seed --dataset PHISON_broke
    CUDA_VISIBLE_DEVICES=0 python train_CEDUE.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/short/$seed --seed $seed --dataset PHISON_short
done

# stage 2
for seed in 2 211 1212 #1 2 3 #2 211 1212
do 
    ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/shift/$seed/ckpt.pt
    
    CUDA_VISIBLE_DEVICES=0 python train_CEDUE_stage2_gp.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/shift/$seed --checkpoint_path $ckpt --seed $seed --dataset PHISON_shift
    CUDA_VISIBLE_DEVICES=0 python train_CEDUE_stage2_mlp.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/MLP/shift/$seed --checkpoint_path $ckpt --seed $seed --dataset PHISON_shift
done

for seed in 2 211 1212 #1 2 3 #2 211 1212
do 
    ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/broke/$seed/ckpt.pt
    
    CUDA_VISIBLE_DEVICES=0 python train_CEDUE_stage2_gp.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/broke/$seed --checkpoint_path $ckpt --seed $seed --dataset PHISON_broke
    CUDA_VISIBLE_DEVICES=0 python train_CEDUE_stage2_mlp.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/MLP/broke/$seed --checkpoint_path $ckpt --seed $seed --dataset PHISON_broke
done

for seed in 2 211 1212 #1 2 3 #2 211 1212
do 
    ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/short/$seed/ckpt.pt
    
    CUDA_VISIBLE_DEVICES=0 python train_CEDUE_stage2_gp.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/short/$seed --checkpoint_path $ckpt --seed $seed --dataset PHISON_short
    CUDA_VISIBLE_DEVICES=0 python train_CEDUE_stage2_mlp.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/MLP/short/$seed --checkpoint_path $ckpt --seed $seed --dataset PHISON_short
done
```
## Bayesian optimization script
- **method1:**
```shell
for seed in 2 211 1212 #1 2 3 #211 1212 2
do 
    ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/four_cls/$seed/ckpt.pt
    mlp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/MLP/four_cls/$seed/ckpt.pt
    gp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/four_cls/$seed/ckpt.pt
    
    CUDA_VISIBLE_DEVICES=0 python bay_CEdue_stage2_method1.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_fourcls_rd_$seed --dataset PHISON_fourcls
done
```
- **method2:**
```shell
for seed in 2 211 1212 #1 2 3 #211 1212 2
do 
    ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/shift/$seed/ckpt.pt
    mlp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/MLP/shift/$seed/ckpt.pt
    gp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/shift/$seed/ckpt.pt
    
    CUDA_VISIBLE_DEVICES=0 python bay_CEdue_stage2_method2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_shift_rd_$seed --dataset PHISON_shift
done
for seed in 2 211 1212 #1 2 3 #211 1212 2
do 
    ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/broke/$seed/ckpt.pt
    mlp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/MLP/broke/$seed/ckpt.pt
    gp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/broke/$seed/ckpt.pt
    
    CUDA_VISIBLE_DEVICES=0 python bay_CEdue_stage2_method2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_broke_rd_$seed --dataset PHISON_broke
done
for seed in 2 211 1212 #1 2 3 #211 1212 2
do 
    ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/short/$seed/ckpt.pt
    mlp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/MLP/short/$seed/ckpt.pt
    gp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/short/$seed/ckpt.pt
    
    CUDA_VISIBLE_DEVICES=0 python bay_CEdue_stage2_method2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_short_rd_$seed --dataset PHISON_short
done
```
## Inference script
- **method1:**
```shell
for seed in 2 211 1212 #1 2 3 #211 1212 2
do 
    ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/four_cls/$seed/ckpt.pt
    mlp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/MLP/four_cls/$seed/ckpt.pt
    gp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/four_cls/$seed/ckpt.pt
    
    CUDA_VISIBLE_DEVICES=0 python tsne_CEdue.py --embedding_layer shared_embedding --checkpoint_path $ckpt_fourcls --random_seed $seed --output_inference_dir CEdue_rd_$seed
    
    CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_fourcls_rd_$seed --dataset PHISON_fourcls

    CUDA_VISIBLE_DEVICES=0 python inference_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --mlp_checkpoint_path $mlp_ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_fourcls_rd_$seed --dataset PHISON_fourcls

done
```
- **method2:**
```shell
for seed in 2 211 1212 #1 2 3 #211 1212 2
do 
    ckpt_shift=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/shift/$seed/ckpt.pt
    gp_ckpt_shift=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/shift/$seed/ckpt.pt
    ckpt_broke=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/broke/$seed/ckpt.pt
    gp_ckpt_broke=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/broke/$seed/ckpt.pt
    ckpt_short=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/short/$seed/ckpt.pt
    gp_ckpt_short=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/short/$seed/ckpt.pt
    mlp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/MLP/short/$seed/ckpt.pt
    
    
    CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE_stage2.py --relabel --checkpoint_path $ckpt_shift --gp_checkpoint_path $gp_ckpt_shift --random_seed $seed --output_inference_dir CEdue_s2_shift_rd_$seed --dataset PHISON_shift
    CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE_stage2.py --relabel --checkpoint_path $ckpt_broke --gp_checkpoint_path $gp_ckpt_broke --random_seed $seed --output_inference_dir CEdue_s2_broke_rd_$seed --dataset PHISON_broke
    CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE_stage2.py --relabel --checkpoint_path $ckpt_short --gp_checkpoint_path $gp_ckpt_short --random_seed $seed --output_inference_dir CEdue_s2_short_rd_$seed --dataset PHISON_short

    CUDA_VISIBLE_DEVICES=0 python inference_CEDUE_stage2_method2.py --relabel --checkpoint_path_shift $ckpt_shift --gp_checkpoint_path_shift $gp_ckpt_shift --checkpoint_path_broke $ckpt_broke --gp_checkpoint_path_broke $gp_ckpt_broke --checkpoint_path_short $ckpt_short --gp_checkpoint_path_short $gp_ckpt_short --mlp_checkpoint_path $mlp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_method2_fourcls_rd_$seed --dataset PHISON_fourcls

    CUDA_VISIBLE_DEVICES=0 python tsne_CEdue.py --embedding_layer shared_embedding --checkpoint_path $ckpt_shift --random_seed $seed --output_inference_dir CEdue_rd_$seed
    
    CUDA_VISIBLE_DEVICES=0 python tsne_CEdue.py --embedding_layer shared_embedding --checkpoint_path $ckpt_broke --random_seed $seed --output_inference_dir CEdue_rd_$seed
    
    CUDA_VISIBLE_DEVICES=0 python tsne_CEdue.py --embedding_layer shared_embedding --checkpoint_path $ckpt_short --random_seed $seed --output_inference_dir CEdue_rd_$seed

    
done
```

