
for seed in 1 1212 42
do 

    ckpt=/root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/fruit_8/$seed/ckpt.pt
    
    CUDA_VISIBLE_DEVICES=0 python tsne_CEdue_fruit.py --embedding_layer shared_embedding --checkpoint_path $ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed
    
#     CUDA_VISIBLE_DEVICES=0 python bay_fruit.py --relabel --checkpoint_path $ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed
    
#     CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE_fruit.py --relabel --checkpoint_path $ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed

#     CUDA_VISIBLE_DEVICES=0 python inference_CEDUE_fruit.py --relabel --checkpoint_path $ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed
    
done

for seed in 1 1212 42
do 
    ckpt=/root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/otherdataset/final/Fruit_8/fruit_8/$seed/ckpt.pt
    gp_ckpt=/root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/stage2_model/fruit_8_stage2/$seed/gp_ckpt.pt
    
    CUDA_VISIBLE_DEVICES=0 python bay_fruit_s2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed
    
    CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE_fruit_s2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed

    CUDA_VISIBLE_DEVICES=0 python inference_CEDUE_fruit_s2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed
    
done
