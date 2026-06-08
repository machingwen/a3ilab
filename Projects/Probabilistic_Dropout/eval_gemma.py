
from transformers import (
    DataCollatorForSeq2Seq,
    DefaultDataCollator
)
from datasets import load_dataset, load_from_disk
from huggingface_hub import login
from transformers import GPT2Tokenizer, GPT2Model,AutoConfig,LlamaForCausalLM,DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,Trainer,TrainingArguments,AutoModelForCausalLM

#import wandb
import torch.optim as optim
import numpy as np
import torch


'''
inactivate neuron= 530512
inactivate percent= 67.362



'''
def show_num(model):
	Total_params = 0
	Trainable_params = 0
	NonTrainable_params = 0
	for param in model.parameters():
		mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
		Total_params += mulValue  # 总参数量
		if param.requires_grad:
			Trainable_params += mulValue  # 可训练参数量
	Total_params=Total_params/1000000
	Trainable_params=Trainable_params/1000000
	print(f'Total params: {Total_params} M')
	print(f'Trainable params: {Trainable_params} M')
	








def get_reply(input_text):
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(input_ids=inputs.input_ids.to('cuda'),
	    	attention_mask=inputs.attention_mask.to('cuda'),
		max_length=384,
		num_return_sequences=1,
		pad_token_id=tokenizer.eos_token_id)  # Ensure to use the appropriate pad token id)
		
        generated_text = tokenizer.batch_decode(outputs)[0]

        return generated_text
                  



from deepspeed.runtime.activation_checkpointing import checkpointing
	
#model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-base")  #bigscience/mt0-xxl
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


	

if 1==0:
	tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b", device_map="cuda")
	model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", device_map="cuda",torch_dtype=torch.float16)	
	
if 1==0:# my pretrain model after alpaca_clean 5 epoch
	path='facebook/opt-2.7b'
	tokenizer = AutoTokenizer.from_pretrained(path)
	model = AutoModelForCausalLM.from_pretrained("output/opt_model/final_model",torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)  #final_model   #checkpoint-500
	
	#model = AutoModelForCausalLM.from_pretrained("output/base_model/final_model",torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)  #final_model   #checkpoint-500


#tokenizer.pad_token = "▁<EOT>"
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#tokenizer.pad_token = tokenizer.eos_token

import time
import random

total=0
neuron=1
inactivate=0
inactivate2=0
neuron2=1
first=True

num=0
ac=np.zeros((33, 17384)) #紀錄神經元活躍次數
atotal=np.zeros((33, 17384)) # 紀錄神經元被統計次數

cp1=np.ones((33, 17384))

copy_neuron= {}

level=np.zeros((202))
an=0
num=0

from accelerate.utils import DistributedType

hook = []

def check_end(model,tokenizer):
	global total, neuron, inactivate, neuron2, inactivate2, first, num, ac, atotal, level, ts, sd, change, an, hook, recount
	level=np.zeros((202))
	def remove_all_layers():
		global hook
		for k in hook:
			k.remove()
	def print_all_layers(name,module,deep, prefix=""): #統計神經元活躍度，替layer加入 hook_fn
		n=0
		global hook
		for name, module in module.named_children():
			n+=1
			print_all_layers(name,module,deep+1)

		if 1==1 and n==0 and hasattr(module, 'weight') and isinstance(module, torch.nn.Linear):
			hh=module.register_forward_hook(hook_fn)
			hook.append(hh)

	def cancel_all_layers(name,module,deep, prefix=""):
		n=0
		global hook
		for name, module in module.named_children():
			n+=1
			cancel_all_layers(name,module,deep+1)

		if 1==1 and n==0 and hasattr(module, 'weight') and isinstance(module, torch.nn.Linear) and ('up_proj' in name):
			#print('Get!!  deep=',deep,' module=',module ,' name=',name )
			hh=module.register_forward_hook(hook_fn_cancel)
			hook.append(hh)	
		elif 1==1 and n==0 and hasattr(module, 'weight') and isinstance(module, torch.nn.Linear):
			#print('Get!!  deep=',deep,' module=',module ,' name=',name )
			hh=module.register_forward_hook(hook_fn_cancel_base)
			hook.append(hh)		
				

	def train_all_layers(name,module,deep, prefix=""): #訓練時對冷神經元做出處理，加入hook_fn_train
		n=0
		global hook
		for name, module in module.named_children():
			n+=1
			train_all_layers(name,module,deep+1)

		if 1==0 and n==0 and hasattr(module, 'weight') and isinstance(module, torch.nn.Linear) and ('down_proj' in name):
			#print('count!!  deep=',deep,' module=',module ,' name=',name )
			#hh=module.register_forward_hook(hook_fn_train)
			hh=module.register_forward_pre_hook(hook_fn_train)
			hook.append(hh)	
		if 1==0 and n==0 and hasattr(module, 'weight') and isinstance(module, torch.nn.Linear) and ('up_proj' in name):
			hh=module.register_forward_pre_hook(hook_fn_up)
			hook.append(hh)	
		if 1==0 and n==0 and hasattr(module, 'weight') and isinstance(module, torch.nn.Linear) and ('gate_proj' in name):
			hh=module.register_forward_pre_hook(hook_fn_gate)
			hook.append(hh)	
	def hook_fn_train(module, input): # train過程中，隨機dropout冷神經元，須配合  hook_fn_count
		global total, num, ac, atotal, cp1
		#fc2	 
		new_input = input[0].clone()  # 複製輸入
		#print('fn_train:',new_input.shape,' ',new_input)
		#new_input[new_input < 0] = 0		
		new_input = (new_input,)  # 
		return  new_input
	
	def hook_fn_up(module, input): # train過程中，隨機dropout冷神經元，須配合  hook_fn_count
		
		new_input = input[0].clone()  # 複製輸入
		print('fn_up:',new_input.shape,' ',new_input)
		#new_input[new_input < 0] = 0		
		
	
	def hook_fn_gate(module, input): # train過程中，隨機dropout冷神經元，須配合  hook_fn_count
		new_input = input[0].clone()  # 複製輸入
		print('fn_gate:',new_input.shape,' ',new_input)
		#new_input[new_input < 0] = 0	
		
	def hook_fn(module, input, output):  #統計神經元活度度的 hook	
		global total, neuron, inactivate, first, num, ac, atotal, level, an,copy_neuron, ts, sd
		if 1==0 or (module.weight.shape[0]==2048  and module.weight.shape[1]==16384 ):# 適用於opt2.7b 的 fc1 layer
			total+=1
			if total>18: # model的最後一個此種layer
				num+=1
				total=1
			#print('\nhook_fn num:',num,' total:',total,"input_shape: ",input[0].shape)
			x = input[0]
			T = x.shape[1]
			for t in range(input[0].shape[0]):#遍歷layer中的每個batch
				for z in range(input[0].shape[2]):#遍歷layer中的每個神經元
					er=0
					col = x[t, :, z]  # [T]

					er = (col > sd).sum().item()

					an += 1

					atotal[total][z] += T
					ac[total][z] += er
					
					"""
					for i in range(0, input[0].shape[1]): #遍歷inference時的每個輸入token
						if input[0][t][i][z]>sd:  #-module.bias[z]:
							er+=1
							#break		
					an+=1 #第幾次統計
					atotal[total][z]=atotal[total][z]+input[0].shape[1]  #total是第幾層 z是該曾第幾神經元
					ac[total][z]=ac[total][z]+er
					"""
								
		
		
	def hook_fn_cancel(module, input, output):
		global total, neuron, inactivate, neuron2, inactivate2, first, num, ac, atotal, level, change, ts
		t=0
		t2=0						
		if module.weight.shape[0]==16384 and module.weight.shape[1]==2048 :

			neuron2+=module.weight.shape[0]	
			#print('hook_fn_cancel up total=',total,' input:',input[0].shape,' ',ac[total][0],' ',atotal[total][0],' g=',ac[total][0]/atotal[total][0])
			
			ac_t = ac[total]
			atotal_t = atotal[total]
			w0=module.weight.shape[0]
			for z in  range(w0): #遍歷layer中的每個神經元	-----
				
				#g=ac[total][z]/atotal[total][z]  #該神元的活躍度
				
				g= ac_t[z] / atotal_t[z]
				#print('g=',g,' ',(g*1000+1),' total=',total,' z=',z,' ',ac[total][z],' ',atotal[total][z])
				if g<0.001:
					level[0]+=1              #紀錄有多少神經元在某個活躍度的區間
				elif g>=0.9:
					level[181]+=1
				elif g>=0.1:
					level[100+(int)((g-0.1)/0.01)]+=1
				else:
					level[(int)(g*1000)+1]+=1					
				if g<=ts and 1==1:                     #若活躍度小於個闕值值
					t+=module.weight.shape[1]
					t2+=1
					module.weight.data[z, :]=0 #關閉該神經元(dropout)
		
		neuron+=module.weight.shape[0]*module.weight.shape[1]				
		inactivate+=t
		inactivate2+=t2
		
	def hook_fn_cancel_base(module, input, output):
		global total, neuron, inactivate, neuron2, inactivate2, first, num, ac, atotal, level, change, ts
		t=0
		t2=0	
		#print('hook_fn_cancel base total=',total,' input:',input[0].shape,' w:',module.weight.shape)					
		if module.weight.shape[0]==16384 and module.weight.shape[1]==2048 :
			total+=1
			if total>18:	
				num+=1
				total=1				
			neuron2+=module.weight.shape[0]	
			#print('hook_fn_cancel gate total=',total,' input:',input[0].shape,' ',ac[total][0],' ',atotal[total][0],' g=',ac[total][0]/atotal[total][0])	
			for z in  range(module.weight.shape[0]): #遍歷layer中的每個神經元	-----
				g=ac[total][z]/atotal[total][z]  #該神元的活躍度
				if g<0.001:
					level[0]+=1              #紀錄有多少神經元在某個活躍度的區間
				elif g>=0.9:
					level[181]+=1
				elif g>=0.1:
					level[100+(int)((g-0.1)/0.01)]+=1
				else:
					level[(int)(g*1000)+1]+=1					
				if g<=ts and 1==1:                     #若活躍度小於個闕值值
					t+=module.weight.shape[1]
					t2+=1
					module.weight.data[z, :]=0 #關閉該神經元(dropout)
		elif module.weight.shape[0]==2048 and module.weight.shape[1]==16384:
			neuron2+=module.weight.shape[1]	
		
			#print('hook_fn_cancel total=',total,' input:',input[0].shape,' ',ac[total][0],' ',atotal[total][0],' g=',ac[total][0]/atotal[total][0])	
			for z in  range(module.weight.shape[1]): #遍歷layer中的每個神經元	-----			
				g=ac[total][z]/atotal[total][z]  #該神元的活躍度
				
				if g<0.001:
					level[0]+=1              #紀錄有多少神經元在某個活躍度的區間
				elif g>=0.9:
					level[181]+=1
				elif g>=0.1:
					level[100+(int)((g-0.1)/0.01)]+=1
				else:
					level[(int)(g*1000)+1]+=1					
				if g<=ts and 1==1:                     #若活躍度小於個闕值值
					t+=module.weight.shape[0]
					t2+=1
					module.weight.data[:,z]=0 #關閉該神經元(dropout)
		neuron+=module.weight.shape[0]*module.weight.shape[1]				
		inactivate+=t
		inactivate2+=t2			
	global recount_data	
	if recount=='yes': # 
		print('recount with ',recount_data)
		
		rdata=recount_data+".txt"
		with open(rdata, "r") as f:
			input_text_train = f.readlines()  # Reads all lines into a list

		ac=np.zeros((33, 17384))
		atotal=np.zeros((33, 17384))
		an=0
		for name, module in model.named_children(): #加入統計經元活度的hook
			print_all_layers(name,module,0)
		i=0
		for t in input_text_train:
			print('t=',i,'/',len(input_text_train))#,'  text=',t
			i+=1
			if i>9:#5
				break
			inputs_train = tokenizer(t, return_tensors="pt").to('cuda')
			output = model(input_ids=inputs_train.input_ids.to('cuda'), attention_mask=inputs_train.attention_mask.to('cuda')).logits	 #實際開始統計經元活度，透過inference		
		np.save("eval_gemma_ac.npy", ac)
		np.save("eval_gemma_atotal.npy", atotal)
		print('save ac, ac=')
		remove_all_layers()
		#print('save atotal, atotal=',atotal)
			
	else: # load
		print('---load and not  ---')
		ac = np.load("eval_gemma_ac.npy")
		atotal = np.load("eval_gemma_atotal.npy")
	
		#print(ac)
		#print(atotal)
		
		
	input_ids0 = torch.tensor([[1694]])	
	attention_mask0 = torch.tensor([[1]])	
	if 1==1: # add dropout
		total=0		
		change=True
		for name, module in model.named_children():
			cancel_all_layers(name,module,0)		
		output = model(input_ids=input_ids0.to('cuda'), attention_mask=attention_mask0.to('cuda')).logits
	
	remove_all_layers()	
	hook=[]
	if 1==1: # add dropout  訓練時對冷神經元處理
		total=0
		for name, module in model.named_children():
			#count_all_layers(name,module,0)
			train_all_layers(name,module,0)			
		output = model(input_ids=input_ids0.to('cuda'), attention_mask=attention_mask0.to('cuda')).logits
		#remove_all_layers()	

	
	
	
	
	p=100*inactivate/neuron
	p2=100*inactivate2/(neuron2)
	p_r=100*inactivate/(256000*2048*2+neuron)
	if 1==1 :
		
		print("\nlevel_count:")
		n=0
		c=0
		m=0
		c2=0
		cdf=np.zeros((202))
		for z in  range(182):
			n+=level[z]
		
		for z in  range(181,-1,-1):
			if z<101:
				c2+=level[z]*z/1000
			else:
				c2+=level[z]*(0.1+(z-100)/100)
			cdf[z]=c2
			#print('z=',z,' level[z]=',level[z],' cdf=',cdf[z])
		m=c2
		x0=[]
		y0=[]
		for z in  range(181):	
			if (z%10==0 and z<=100) or(z==103 or z==107): #or (z>=101 and z%10==0) 
				pp=" percent:"+(str)(round(100-c*100/n,2))+"% ("+(str)(round(c*100/n,2))+"%)"
				print('level ',z," ",(z/1000),pp,"  cdf=",round(cdf[z]*100/m,2),"%")#,level[z]
			if z==100 or z==110 or z==50 or z==30 or z==20 or z==10:
				x0.append(round(100-c*100/n,2))
				y0.append(round(cdf[z]*100/m,2))
				
			c+=level[z]
				
		print('total neuron=',neuron,' inactivate neuron=',inactivate," real total neron:",(256000*2048*2+neuron))
		#print()
		#print('inactivate percent=', "{:.3f}".format(p)," : ", "{:.3f}".format(p_r))
		print('inactivate2 percent=', "{:.3f}".format(p2))
		x0.reverse()
		y0.reverse()
		print('x0=',x0)
		print('y0=',y0)
		
		
	return model
def tokenize4_swag (batch):
    max_l=512
    q1=batch["startphrase"]+' '
    d=["","","",""]
    d[0]=batch["ending0"]
    d[1]=batch["ending1"]
    d[2]=batch["ending2"]
    d[3]=batch["ending3"]   
    q2=d[batch["label"]]
    q=q1+q2
   # print(q)
    tokenized_inputs = tokenizer(q,padding='max_length', max_length=max_l, truncation=True)    
    tokenized_labels = tokenizer(q,padding='max_length', max_length=max_l, truncation=True) 
    tokenized_labels = tokenized_labels['input_ids']
   
    tokenized_labels.append(tokenizer.eos_token_id)    
    tokenized_inputs["input_ids"].append(tokenizer.eos_token_id)
    tokenized_inputs["attention_mask"].append(1) 
    
    k= {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': tokenized_labels,
        'label': tokenized_labels
    }
    return k	
def eval_model(model,model_name,output_path):

    def tokenize4(batch):
	    max_l=512
	    #q1= batch["text"] #+"\n" 
	    q1= batch["instruction"] +' '+batch["input"]+' '+batch["output"]
	    tokenized_inputs = tokenizer(q1,padding='max_length', max_length=max_l, truncation=True)    
	    tokenized_labels = tokenizer(q1,padding='max_length', max_length=max_l, truncation=True) 
	    tokenized_labels = tokenized_labels['input_ids']
	     
	    tokenized_labels.append(tokenizer.eos_token_id)    
	    tokenized_inputs["input_ids"].append(tokenizer.eos_token_id)
	    tokenized_inputs["attention_mask"].append(1) 
	    
	    k= {
		'input_ids': tokenized_inputs['input_ids'],
		'attention_mask': tokenized_inputs['attention_mask'],
		'labels': tokenized_labels
	    }
	    return k
	    
    eval_data=''
    
    if 1==0:
    	train_data=load_dataset("yahma/alpaca-cleaned",split="train[:40%]")
    	test_data=load_dataset("yahma/alpaca-cleaned",split="train[98%:]")
    	tokenized_train_data = train_data.map(tokenize4)
    	tokenized_test_data = test_data.map(tokenize4)
    else:

    	test_data=load_dataset("allenai/swag",split="train[:64]")
    	tokenized_test_data = test_data.map(tokenize4_swag)
    model.eval()  # 設定為推理模式 
   
    data_collator =   DataCollatorForSeq2Seq(tokenizer, model=model, 
    	padding=True, label_pad_token_id=-100)
    
 
    args = TrainingArguments(
        output_dir='eval',
        per_device_eval_batch_size=4,  # Reduced batch size
        eval_strategy="steps",
        eval_steps=5,
        gradient_accumulation_steps=4,  # Increased accumulation steps
        num_train_epochs=1,
        weight_decay=1e-3,#1e3
        #max_steps=100,
        warmup_steps=2,  # Increased warmup steps
        lr_scheduler_type="cosine",
        #lr_scheduler_type="constant",
        learning_rate=3e-5,  # 6e-5 baseline 3e-5 Adjusted learning rate
        #save_steps=50000,  # Adjusted save steps
        #save_strategy="no",
        save_total_limit=0,
        #deepspeed="deepspeed2.json",  # Path to DeepSpeed config file
        report_to="none",
        #do_train=False,  
        bf16=True
    )
    #args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_test_data, #tokenized_data["train"],
        eval_dataset=tokenized_test_data, #tokenized_data["test"],
    )
   
    print('model=',model_name)
    
    result=trainer.evaluate()  # 執行評估
    print(result,'\n\n')
    print('\nstart save at ',output_path,'\n\n')
    trainer.save_model(f"output/{output_path}/final_model")
    return model
    
def print_all_layers(module, prefix=""):
    for name, child in module.named_children():
        # Print the layer with its hierarchy
        #print(f"{prefix}{name}: {child.__class__.__name__}")
        # Recursively process child modules
        print_all_layers(child, prefix=prefix + "  ")
import time
import random
# Call the function on the model
#print_all_layers(model)
total=0
neuron=0
inactivat=0
first=True
cp=[0]*28192
num=0


an=0

def inference(model,input_text):
	inputs = tokenizer(input_text, return_tensors="pt")
	model.eval()
	#print("\n\n-------------------------\n")
	#print("input:\n")
	#print(input_text)
	with torch.no_grad():	
	    outputs = model.generate(input_ids=inputs.input_ids.to('cuda'),
	    	attention_mask=inputs.attention_mask.to('cuda'),
		max_length=inputs.input_ids.shape[1]+20,
		#temperature=0,
		num_return_sequences=1,
		pad_token_id=tokenizer.eos_token_id)  # Ensure to use the appropriate pad token id)

	generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
	#print("\n\noutput:\n")
	#print(generated_text)
	#print("\n\n-------------------------\n")
	return generated_text
	
	

from transformers.utils import logging
import argparse

'''
import lm_eval
from lm_eval.utils import setup_logging
from lm_eval.api.model import LM

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from  lm_eval.api.instance  import Instance
from lm_eval.models.huggingface import HFLM
@register_model("my_in_memory_model")
class MyLM(HFLM):
    def __init__(self, model_instance, tokenizer_instance, device="cpu", batch_size=1):
        # HFLM 的 __init__ 需要 hf_model 和 hf_tokenizer
        # 它會自動處理 device 和 batch_size
        # 注意：trust_remote_code=True 應該在 from_pretrained 時處理
        super().__init__(
            pretrained=None, # 我們直接傳入實例，所以這裡填None
            tokenizer=None,  # 我們直接傳入實例，所以這裡填None
            device=device,
            batch_size=batch_size,
        )
        self._model = model_instance # 將傳入的模型實例賦值給 HFLM 內部使用的 _model
        self.tokenizer = tokenizer_instance # 將傳入的 tokenizer 實例賦值給 HFLM 內部使用的 tokenizer
        self._model.eval() # 確保模型處於評估模式

'''
# 實例化您的自定義模型



#setup_logging("DEBUG") # optional, but recommended; or you can set up logging yourself
input_text="the Anglo-Saxon kingdom of Wessex emerged as the dominant English"
input_text="Kyiv and Moscow are not known to have held direct talks at"

if __name__ == "__main__":
    global ts, sd, recount, recount_data
    parser = argparse.ArgumentParser()

    # 定義參數
    parser.add_argument("-m", "--model", type=str, default="gemma_model")
    parser.add_argument("-p", "--output", type=str, default="dropout_model")
    parser.add_argument("-o", "--original", type=str, default="no", help="original model or not")
    parser.add_argument("-d", "--dropout", type=str, default="yes", help="dropout")
    parser.add_argument("-t", "--threshold", type=float, default=0.01, help="float g")
    parser.add_argument("-r", "--recount", type=str, default="yes", help="recount")
    parser.add_argument("-c", "--recount_data", type=str, default="data_siqa", help="recount dataset")
    
    parser.add_argument("-s", "--sd", type=float, default=0.3, help="relu standard")


    # 解析參數
   
   
    args = parser.parse_args()
    model_name=args.model
    dr=args.dropout
    ts=args.threshold
    sd=args.sd
    om=args.original
    recount=args.recount
    output_p = args.output
    recount_data=args.recount_data
    print(f"dropout: {args.dropout}")
    print(f"dropout threshold: {args.threshold}")
    print(f"relu standard: {args.sd} ")
    print(f"recount: {recount} ")
   
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    tokenizer.pad_token = '[PAD]'
    if om=='yes':
    	print('google/gemma-2b')
    	model =  AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="cuda", torch_dtype=torch.bfloat16)
    else:
    	print('load my model ', model_name)
    	model = AutoModelForCausalLM.from_pretrained("output/"+model_name+"/final_model",torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)  #final_model   #checkpoint-500
    
    real_output=inference(model,input_text)
	
    # 呼叫主函數
    #print(model)
    show_num(model)
    if dr=='yes':
    	model=check_end(model, tokenizer) 
    
    input_text="There are two spelling errors in the sentence. The corrected sentence"
    #print('input:\n', input_text)
    #print('real_output:\n', real_output)
    #out=inference(model,input_text)
    #print('output:\n', out)
    
    input_text="Some people don't understand the internet's"
    #print('input:\n', input_text)
    #print('real_output:\n', real_output)
   # out=inference(model,input_text)
  #  print('output:\n', out)

    input_text="A number of casualties were confirmed, although the exact number of wounded was"
   # print('input:\n', input_text)
   # out=inference(model,input_text)
   # print('output:\n', out)    
    
    input_text="After the fall of the Roman Empire, the area was first invaded"
    #print('input:\n', input_text)
    #out=inference(model,input_text)
    #print('output:\n', out)    
    
    #print('\n')	



    model=eval_model(model,model_name, output_p)
    

    if 1==0:
	    print('\n-- lm_eval --\n')
	    my_model = MyLM(model, tokenizer, device="cpu") # 這裡使用CPU以簡化，可改為cuda:0

	    #my_in_memory_lm_instance = MyInMemoryLM(model, tokenizer, device="cpu") # 這裡使用CPU以簡化，可改為cuda:0
	    
	    results = lm_eval.simple_evaluate(
	    model=my_model, # 注意這裡不是hf，而是你註冊的模型名稱
	    tasks=["hellaswag"], # 替換為您想評估的任務
	    batch_size=1,
	    device="cpu", # 與模型實例的device一致
	    limit=10 # 限制評估樣本數，方便測試
	    )
	    print(results)


    print('\n----------------------------------------end ------------------------------\n')
	
