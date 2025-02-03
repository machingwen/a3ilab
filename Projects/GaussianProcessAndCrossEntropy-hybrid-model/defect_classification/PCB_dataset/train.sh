# data regroup
# CUDA_VISIBLE_DEVICES=0 python train_data_regroup.py --output_dir /root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/phison_data_regroup --seed 1212

for seed in 2 211 1212 #1 2 3 #2 211 1212
do 
    CUDA_VISIBLE_DEVICES=0 python train_CE.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/threecls/$seed --seed $seed --dataset PHISON_threecls
#     CUDA_VISIBLE_DEVICES=0 python train_CEDUE.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/good/$seed --seed $seed --dataset PHISON_good
done



# for seed in 2 211 1212 #1 2 3 #2 211 1212
# do 
#     ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/good/$seed/ckpt.pt
#     CUDA_VISIBLE_DEVICES=0 python train_CEDUE_stage2_gp.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/good/$seed --checkpoint_path $ckpt --seed $seed --dataset PHISON_good
#     CUDA_VISIBLE_DEVICES=0 python train_CEDUE_stage2_mlp.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/MLP/good/$seed --checkpoint_path $ckpt --seed $seed --dataset PHISON_good
# done

# for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 #2 211 1212
# do 
#     CUDA_VISIBLE_DEVICES=0 python train_CE_six.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/Baseline/$seed --seed $seed
#     # CUDA_VISIBLE_DEVICES=0 python train_CEDUE.py --output_dir /root/notebooks/nfs/work/Kelly.Lin/GP --seed $seed
# done