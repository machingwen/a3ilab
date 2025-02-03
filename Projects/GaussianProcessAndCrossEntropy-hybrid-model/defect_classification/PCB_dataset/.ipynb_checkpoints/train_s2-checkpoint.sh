# data regroup

# CUDA_VISIBLE_DEVICES=0 python train_data_regroup.py --output_dir /root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/phison_data_regroup --seed 1212

# for seed in 238 #2 211 1212
# do 
#     CUDA_VISIBLE_DEVICES=0 python train_CE_six.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/Baseline --seed $seed
#     # CUDA_VISIBLE_DEVICES=0 python train_CEDUE.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/GP --seed $seed
# done



for seed in 2 211 1212
do 
    ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/$seed/ckpt.pt
    CUDA_VISIBLE_DEVICES=0 python train_CEDUE_stage2_gp.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/$seed --checkpoint_path $ckpt --seed $seed
    CUDA_VISIBLE_DEVICES=0 python train_CEDUE_stage2_mlp.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/MLP/$seed --checkpoint_path $ckpt --seed $seed
done
