



python main_gemma.py -load no -recount no  -check no -relu yes -data alpaca -output input_model -lr 3e-5
python main_gemma.py -load yes -recount no  -check no -relu yes -data siqa -output input_model -lr 3e-5

python main_gemma.py -load yes -train yes -recount yes -check no -relu yes -data siqa_short -dropout_method no -input input_model -output my_new_model  -lr 1e-5 -normal_dropout 0

