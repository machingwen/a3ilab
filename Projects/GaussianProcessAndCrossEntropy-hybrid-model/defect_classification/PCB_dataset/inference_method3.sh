# ##  CEDUE  ###
# for seed in 2 211 1212
# do 
#     ckpt=/root/notebooks/nfs/work/Kelly.Lin/GP/$seed/ckpt.pt

#     CUDA_VISIBLE_DEVICES=0 python tsne_CEdue.py --embedding_layer shared_embedding --checkpoint_path $ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed

# #     # CUDA_VISIBLE_DEVICES=0 python bay_CEdue.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed 

# #     # CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed

# #     # CUDA_VISIBLE_DEVICES=0 python inference_CEDUE.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_rd_$seed

# done

# for seed in 211 1212 2
# do 
#     ckpt=/root/notebooks/nfs/work/Kelly.Lin/GP/$seed/ckpt.pt
#     gp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/GP_s2/$seed/ckpt.pt


#     #CUDA_VISIBLE_DEVICES=0 python bay_CEdue_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed 

#     #CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed

#     CUDA_VISIBLE_DEVICES=0 python inference_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed

# done

# for seed in 211 1212 2
# do 
#     ckpt1=/root/notebooks/nfs/work/Kelly.Lin/GP/old/$seed/ckpt.pt
#     ckpt2=/root/notebooks/nfs/work/Kelly.Lin/GP/new/$seed/ckpt.pt
#     gp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/GP_s2/$seed/ckpt.pt


#     #CUDA_VISIBLE_DEVICES=0 python bay_CEdue_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed 

#     #CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE_stage2.py --relabel --checkpoint_path $ckpt --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed

#     CUDA_VISIBLE_DEVICES=0 python inference.py --relabel --checkpoint_path1 $ckpt1 --checkpoint_path2 $ckpt2 --gp_checkpoint_path $gp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_rd_$seed

# done

for seed in 211 1212 #1 2 3 #211 1212 2
do 
    
    ckpt_good=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/good/$seed/ckpt.pt
    gp_ckpt_good=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/good/$seed/ckpt.pt
    ckpt_threecls=/root/notebooks/nfs/work/Kelly.Lin/model/threecls/$seed/ckpt.pt
    ckpt_shift=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/shift/$seed/ckpt.pt
    gp_ckpt_shift=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/shift/$seed/ckpt.pt
    ckpt_broke=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/broke/$seed/ckpt.pt
    gp_ckpt_broke=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/broke/$seed/ckpt.pt
    ckpt_short=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/short/$seed/ckpt.pt
    gp_ckpt_short=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/short/$seed/ckpt.pt
    mlp_ckpt=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/MLP/short/$seed/ckpt.pt
    ckpt_fourcls=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s1/four_cls/$seed/ckpt.pt
    gp_ckpt_fourcls=/root/notebooks/nfs/work/Kelly.Lin/model/GP_s2/GP/four_cls/$seed/ckpt.pt
    ckpt_baseline=/root/notebooks/nfs/work/Kelly.Lin/model/baseline/$seed/ckpt.pt
    
    # CUDA_VISIBLE_DEVICES=0 python tsne_CE_six.py --embedding_layer shared_embedding --checkpoint_path $ckpt_baseline --random_seed $seed --output_inference_dir CEdue_s2_short_rd_$seed #  --dataset PHISON
    
    # CUDA_VISIBLE_DEVICES=0 python bay_CEdue_stage2.py --relabel --checkpoint_path $ckpt_good --gp_checkpoint_path $gp_ckpt_good --random_seed $seed --output_inference_dir CEdue_s2_good_rd_$seed --dataset PHISON_good
    
    CUDA_VISIBLE_DEVICES=0 python uncertainty_CEDUE_stage2.py --relabel --checkpoint_path $ckpt_good --gp_checkpoint_path $gp_ckpt_good --random_seed $seed --output_inference_dir CEdue_s2_good_rd_$seed --dataset PHISON_good

    # CUDA_VISIBLE_DEVICES=0 python inference_CEDUE_stage2.py --relabel --checkpoint_path_shift $ckpt_shift --gp_checkpoint_path_shift $gp_ckpt_shift --checkpoint_path_broke $ckpt_broke --gp_checkpoint_path_broke $gp_ckpt_broke --checkpoint_path_short $ckpt_short --gp_checkpoint_path_short $gp_ckpt_short --mlp_checkpoint_path $mlp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_fourcls_rd_$seed --dataset PHISON_fourcls
    python inference_CEDUE_method3.py --relabel --checkpoint_path_good $ckpt_good --gp_checkpoint_path_good $gp_ckpt_good --checkpoint_path_threecls $ckpt_threecls --mlp_checkpoint_path $mlp_ckpt --random_seed $seed --output_inference_dir CEdue_s2_method3_rd_$seed --dataset PHISON_fourcls

    # CUDA_VISIBLE_DEVICES=0 python tsne_CEdue.py --embedding_layer shared_embedding --checkpoint_path $ckpt_fourcls --random_seed $seed --output_inference_dir CEdue_rd_$seed

done
