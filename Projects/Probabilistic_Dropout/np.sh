


python new_gemma.py -load yes -train yes -recount yes -check yes -relu yes -data siqa_short -dropout_method fr -input input_model -output my_new_model  -lr 1e-5 -l1_lr 5e-9

python new_gemma.py -load yes -train yes -recount yes -check yes -relu yes -data siqa_md -dropout_method fr -input my_new_model -output my_new_model  -lr 1e-5 -l1_lr 1e-8

python eval_gemma.py --dropout yes --threshold 0.001  --sd 0 --original no --recount yes --model my_new_model --recount_data data_siqa
lm_eval --model hf \
    --model_args  pretrained=./output/dropout_model/final_model\
    --tasks social_iqa\
    --device cuda \
    --batch_size 8

python eval_gemma.py --dropout yes --threshold 0.003  --sd 0 --original no --recount no --model my_new_model --recount_data data_siqa
lm_eval --model hf \
    --model_args  pretrained=./output/dropout_model/final_model\
    --tasks social_iqa\
    --device cuda \
    --batch_size 8

python eval_gemma.py --dropout yes --threshold 0.005  --sd 0 --original no --recount no --model my_new_model --recount_data data_siqa
lm_eval --model hf \
    --model_args  pretrained=./output/dropout_model/final_model\
    --tasks social_iqa\
    --device cuda \
    --batch_size 8

python eval_gemma.py --dropout yes --threshold 0.007  --sd 0 --original no --recount no --model my_new_model --recount_data data_siqa
lm_eval --model hf \
    --model_args  pretrained=./output/dropout_model/final_model\
    --tasks social_iqa\
    --device cuda \
    --batch_size 8 
python eval_gemma.py --dropout yes --threshold 0.01  --sd 0 --original no --recount no --model my_new_model --recount_data data_siqa 
lm_eval --model hf \
    --model_args  pretrained=./output/dropout_model/final_model\
    --tasks social_iqa \
    --device cuda \
    --batch_size 8

python eval_gemma.py --dropout yes --threshold 0.015  --sd 0 --original no --recount no --model my_new_model --recount_data data_siqa 
lm_eval --model hf \
    --model_args  pretrained=./output/dropout_model/final_model\
    --tasks social_iqa \
    --device cuda \
    --batch_size 8



