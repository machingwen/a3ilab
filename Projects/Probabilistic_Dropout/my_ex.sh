










python main_gemma.py -load no -recount no  -check no -relu yes -data alpaca -output input_model -lr 3e-5


python main_gemma.py -load yes -recount no  -check no -relu yes -data siqa -output input_model -lr 3e-5



python main_gemma.py -load yes -train yes -recount no -check yes -relu yes -data siqa_short -dropout_method linear2 -input input_model -output my_new_model  -lr 1e-5 -normal_dropout 35


python eval_gemma.py --dropout yes --threshold 0.005  --sd 0 --original no --recount yes --model my_new_model --recount_data data_siqa
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


python eval_gemma.py --dropout yes --threshold 0.02  --sd 0 --original no --recount no --model my_new_model --recount_data data_siqa
lm_eval --model hf \
    --model_args  pretrained=./output/dropout_model/final_model\
    --tasks social_iqa\
    --device cuda \
    --batch_size 8
    
python eval_gemma.py --dropout yes --threshold 0.03  --sd 0 --original no --recount no --model my_new_model --recount_data data_siqa
lm_eval --model hf \
    --model_args  pretrained=./output/dropout_model/final_model\
    --tasks social_iqa \
    --device cuda \
    --batch_size 8
    

    
    
    
