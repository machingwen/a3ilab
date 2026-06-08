



python eval_gemma.py --dropout yes --threshold 0.005  --sd 0 --original no --recount yes --model input_model --recount_data data_siqa
lm_eval --model hf \
    --model_args  pretrained=./output/dropout_model/final_model\
    --tasks social_iqa\
    --device cuda \
    --batch_size 8

    
python eval_gemma.py --dropout yes --threshold 0.01  --sd 0 --original no --recount no --model input_model --recount_data data_siqa
lm_eval --model hf \
    --model_args  pretrained=./output/dropout_model/final_model\
    --tasks social_iqa \
    --device cuda \
    --batch_size 8


python eval_gemma.py --dropout yes --threshold 0.02  --sd 0 --original no --recount no --model input_model --recount_data data_siqa
lm_eval --model hf \
    --model_args  pretrained=./output/dropout_model/final_model\
    --tasks social_iqa\
    --device cuda \
    --batch_size 8
    
python eval_gemma.py --dropout yes --threshold 0.03  --sd 0 --original no --recount no --model input_model --recount_data data_siqa
lm_eval --model hf \
    --model_args  pretrained=./output/dropout_model/final_model\
    --tasks social_iqa \
    --device cuda \
    --batch_size 8
    

