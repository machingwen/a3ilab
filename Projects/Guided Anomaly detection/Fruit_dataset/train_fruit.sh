

for seed in 1 1212 42
do 

    CUDA_VISIBLE_DEVICES=0 python train_CEdue_fruit.py --output_dir /root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/fruit_8 --seed $seed
    
done


for seed in 1 1212 42
do 
    ckpt=/root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/otherdataset/final/Fruit_8/fruit_8/$seed/ckpt.pt
    
    CUDA_VISIBLE_DEVICES=0 python train_CEdue_fruit_stage2.py --output_dir /root/notebooks/nfs/work/jason.chen/DUE/regroup_output_model/fruit_8_stage2 --checkpoint_path $ckpt --seed $seed
    
done

