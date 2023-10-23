###  CEDUE  ###
for seed in 2 211 1212
do 
    ckpt=/root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/CEdue/$seed/ckpt.pt
    
    CUDA_VISIBLE_DEVICES=0 python tsne_CEDUE.py --embedding_layer shared_embedding --checkpoint_path $ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed
    
    # CUDA_VISIBLE_DEVICES=0 python bay_CEdue.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed 
    
    # CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed

    # CUDA_VISIBLE_DEVICES=0 python inference_CEDUE.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed
    
done

for seed in 2 211 1212
do 
    ckpt=/root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/CEdue/$seed/ckpt.pt
    gp_ckpt=/root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/stage2_model/CEdue_s2/$seed/gp_ckpt.pt
    
    
    CUDA_VISIBLE_DEVICES=0 python bay_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed 
    
    CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed

    CUDA_VISIBLE_DEVICES=0 python inference_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed
    
done