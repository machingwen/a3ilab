# PCB-Defect-Detection

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
```shell
# data regroup
CUDA_VISIBLE_DEVICES=0 python train_data_regroup.py --output_dir /root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/phison_data_regroup --seed 1212

# stage 1
for seed in 2 211 1212
do 
    CUDA_VISIBLE_DEVICES=0 python train_CEDUE.py --output_dir /root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/CEdue --seed $seed
done
# stage 2
for seed in 2 211 1212
do 
    ckpt=/root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/CEdue/$seed/ckpt.pt
    CUDA_VISIBLE_DEVICES=0 python train_CEDUE_stage2.py --output_dir /root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/CEdue_s2 --checkpoint_path $ckpt --seed $seed
done
```
## Bayesian optimization script
```shell
for seed in 2 211 1212 
do 
    ckpt=/root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/CEdue/$seed/ckpt.pt
    gp_ckpt=/root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/stage2_model/CEdue_s2/$seed/gp_ckpt.pt
    
    CUDA_VISIBLE_DEVICES=0 python bay_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed 
done
```
## Inference script
```shell
for seed in 2 211 1212 
do 
    ckpt=/root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/CEdue/$seed/ckpt.pt
    
    CUDA_VISIBLE_DEVICES=0 python tsne_CEdue.py --embedding_layer shared_embedding --checkpoint_path $ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed
    
    # CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed

    # CUDA_VISIBLE_DEVICES=0 python inference_CEDUE.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed
    
done

for seed in 2 211 1212
do 
    ckpt=/root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/CEdue/$seed/ckpt.pt
    gp_ckpt=/root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/stage2_model/CEdue_s2/$seed/gp_ckpt.pt
    
    CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed

    CUDA_VISIBLE_DEVICES=0 python inference_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed
    
done
```

