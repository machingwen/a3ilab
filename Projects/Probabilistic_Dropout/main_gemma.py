
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

import gc
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

       

def get_reply(input_text): #與此程式無關的函示
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
	path="microsoft/phi-2"
	
	tokenizer = AutoTokenizer.from_pretrained(path)
	model = AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16,device_map="cuda", trust_remote_code=True ,attn_implementation="flash_attention_2")
	
if 1==0:
	model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16, device_map="cuda")
	tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
	#tokenizer.pad_token = tokenizer.eos_token
	tokenizer.pad_token = '[PAD]'
if 1==0:
	model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B",device_map="cuda")
	tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
	
if 1==0:
	tokenizer = AutoTokenizer.from_pretrained("SparseLLM/ReluLLaMA-7B", device_map="cuda")
	model = AutoModelForCausalLM.from_pretrained("SparseLLM/ReluLLaMA-7B", device_map="cuda",torch_dtype=torch.float16)
	
	
	#model = AutoModelForCausalLM.from_pretrained("output/base_model/final_model",torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)  #final_model   #checkpoint-500

	

#tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#tokenizer.pad_token = tokenizer.eos_token
		


#tokenizer.pad_token = tokenizer.unk_token
#model.resize_token_embeddings(len(tokenizer))
def tokenize4_siqa (batch):
    max_l=512
    q1=batch["context"]+' '+batch["question"]+' '
    d=["","","",""]
    d[1]=batch["answerA"]
    d[2]=batch["answerB"]
    d[3]=batch["answerC"]
    g=(int)(batch["label"])
    q2=d[g]
    q=q1+q2
    #print(q)
    tokenized_inputs = tokenizer(q,padding='max_length', max_length=max_l, truncation=True)    
    tokenized_labels = tokenizer(q,padding='max_length', max_length=max_l, truncation=True) 
    tokenized_labels = tokenized_labels['input_ids']
    if 1==1: # if not train question
	    user_inputs = tokenizer(q1, max_length=max_l, truncation=True)    
	    user_len=len(user_inputs['input_ids'])
	    tokenized_labels=[-100] * user_len + tokenized_labels[user_len:]  
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
def tokenize4_wiki(batch):
    max_l=512
    q1= batch["text"] #+"\n"
    #q2= batch["completion"]   
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
def tokenize4_qa(batch):
    max_l=512
    question_text = batch['question']

    choices = batch['choices']
    choice_labels = choices['label']
    choice_texts = choices['text']
    

    choices_str = ""
    for label, text in zip(choice_labels, choice_texts):
        choices_str += f"{label}:{text}, "
    
    # 提取正確答案
    answer_key = batch['answerKey']
   # print('answer_key: ',answer_key)
    #print('st:',choices_str)
    # 結合所有部分，形成最終的格式化字串
    # 注意：這裡使用 f-string 來處理空白和換行，以確保格式正確。
    if answer_key!="" and 1==1:
    	formatted_string = f"Question: {question_text} choices:{choices_str}. Answer:{answer_key}:{choice_texts[choice_labels.index(answer_key)]}."
    else:
    	formatted_string = f"Question: {question_text}{choices_str}."
    	#answer_key="E"

    #print(formatted_string,'\n')
    q1=formatted_string
    #q2=answer_key   #batch["completion"]   
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
def tokenize4_piqa(batch):
    max_l=512
    q1=batch["goal"]+' '
    d=["",""]
    d[0]=batch["sol1"]
    d[1]=batch["sol2"]
    
    q2=d[batch["label"]]
    q=q1+q2
   # print(q)
    #q2= batch["completion"]   
    tokenized_inputs = tokenizer(q1,padding='max_length', max_length=max_l, truncation=True)    
    tokenized_labels = tokenizer(q1,padding='max_length', max_length=max_l, truncation=True) 
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
def tokenize4(batch):
    max_l=512
    q1= batch["instruction"] +' '+batch["input"]+' '+batch["output"]
    #q2= batch["completion"]   
    tokenized_inputs = tokenizer(q1,padding='max_length', max_length=max_l, truncation=True)    
    tokenized_labels = tokenizer(q1,padding='max_length', max_length=max_l, truncation=True) 
    tokenized_labels = tokenized_labels['input_ids']
    if 1==0: # if not train question
	    user_inputs = tokenizer(q1, max_length=max_l, truncation=True)    
	    user_len=len(user_inputs['input_ids'])
	    tokenized_labels=[-100] * user_len + tokenized_labels[user_len:]
	            
    tokenized_labels.append(tokenizer.eos_token_id)    
    tokenized_inputs["input_ids"].append(tokenizer.eos_token_id)
    tokenized_inputs["attention_mask"].append(1) 
    
    k= {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': tokenized_labels
    }
    return k
    
def tokenize4_gsm8k(batch):
    max_l=512
    q1= batch["question"] +' '
    q2= batch["answer"]
    tokenized_inputs = tokenizer(q1+q2,padding='max_length', max_length=max_l, truncation=True)    
    tokenized_labels = tokenizer(q1+q2,padding='max_length', max_length=max_l, truncation=True) 
    tokenized_labels = tokenized_labels['input_ids']
    if 1==0: # if not train question
	    user_inputs = tokenizer(q1, max_length=max_l, truncation=True)    
	    user_len=len(user_inputs['input_ids'])
	    tokenized_labels=[-100] * user_len + tokenized_labels[user_len:]
	            
    tokenized_labels.append(tokenizer.eos_token_id)    
    tokenized_inputs["input_ids"].append(tokenizer.eos_token_id)
    tokenized_inputs["attention_mask"].append(1) 
    
    k= {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': tokenized_labels
    }
    return k
    
    
       
def tokenize6(batch):

    max_l=512
    q1= batch["prompt"] +"\n"
    q2= batch["completion"]   
    tokenized_inputs = tokenizer(q1+q2,padding='max_length', max_length=max_l, truncation=True)    
    tokenized_labels = tokenizer(q1+q2,padding='max_length', max_length=max_l, truncation=True) 
    tokenized_labels = tokenized_labels['input_ids']

    if 1==0: # if not train question
	    user_inputs = tokenizer(q2, max_length=max_l, truncation=True)    
	    user_len=max_l-1-len(user_inputs['input_ids'])
	    tokenized_labels=[-100] * user_len + tokenized_labels[user_len:]

    tokenized_labels.append(tokenizer.eos_token_id)    
    tokenized_inputs["input_ids"].append(tokenizer.eos_token_id)
    tokenized_inputs["attention_mask"].append(1) 
    
    k= {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': tokenized_labels
    }
    return k
      
        
from accelerate.utils import DistributedType
import math
import time
import random

total=0
neuron=0
inactivat=0
first=True

num=0
ac=np.zeros((33, 17384)) #紀錄神經元活躍次數
atotal=np.zeros((33, 17384)) # 紀錄神經元被統計次數

cp1=np.ones((33, 16384))
cp_up=np.ones((33, 16384))
cp_gate=np.ones((33, 16384))
copy_neuron= {}

level=np.zeros((102))
an=0
num=0
hook = []
tmp_weight=0
all_weight=0
def check(model,input_text,input_text_train,tokenizer,print0=True): #主要函式，掛hook並實際處理用
	
	global hook, total, neuron, inactivate,neuron2,inactivate2, first, num, ac, atotal, tmp_ac, tmp_atotal, an, cp1,copy_neuron, level, all_weight, tmp_weight, dropout_method

	global weight_fix, bias_fix	
	
	total=0
	neuron=1
	inactivate=0
	inactivate2=0 
	neuron2=1
	def ch_old(g):
		h=(0.05-g)*25*100
		if h<0.01:
			h=0.01
		return h	
	def ch_old_old(g):
		h=(0.04-g)*20*100
		if h<0.01:
			h=0.01
		return h

	def ch_fix(g):
		if g>26:
			return 0.01
		#h=(71-g)*1.35
		return 100	
		

	def ch_small(g): #
		if g>70:
			return 0.01
		#h=(81-g)*1.3
		h=(71-g)*1.35
		return h

	def ch_56_new(g):
		if g>85:
			return 0.01
		h=(85-g)*1.8
		return h

	
	def ch_cos(g):
		if g>70:
			return 0.02
		
		#h=math.cos(g/200*3.14159)/3+0.05
		
		#h=math.cos((g+15)/200*3.14159)+0.03 # 50
		#h=math.cos((g+20)/200*3.14159)*0.8+0.02# 35
		h=math.cos((g+18)/200*3.14159)+0.02# 
		
		
		#print('ch g=',g,' gg=',gg,' h=',h,' old_h=',(math.cos(g/200*3.14)))
		return h*100
	def ch_sin(g):
		if g>70:
			return 0.02
		#h=(1-math.sin(g*100/79/200*3.14159))/2.3+0.05		
		#h=(1-math.sin(g*100/79/200*3.14159))/2.3+0.05	
		#h=(1-math.sin(g*100/90/200*3.14159))*1.9+0.03#50
		#h=(1-math.sin(g*100/80/200*3.14159))*1.35+0.02#35
		h=(1-math.sin(g*100/80/200*3.14159))*1.7+0.03#
		
		return h*100							
	def ch_f(g):
		
		#h=(0.04-g)*6*100
		#h=(0.06-g)*14*100#50
		#h=(0.06-g)*10*100#35
		h=(0.06-g)*13*100#
		
		if h<0:
			h=0
		return h+2

			
	def ch(g): #42
		if g>80:
			return 2 # 4

		#h=(71-g)*0.95+4 # 	
		#h=(71-g)*1.5 # 
		#h=(81-g)*1.5+3 # averge 50%/48% dropout rate

		#h=(81-g)*1.2+2 # averge 35% dropout rate  (81-g)*1.2+2 
		h=(81-g)*1.4+2 # averge 35% dropout rate  (81-g)*1.2+2 
		
		
		#h=(71-g)*0.6+2	#
		
		#h=(81-g)*0.4+5 # 
		return h		
	def hook_fn_count(module, input, output): #train時，對隨機到的冷神經元處理(非統計)
		global total, neuron, inactivate, inactivate2, neuron2, first, num, ac, atotal, level, cp1
		global weight_fix, bias_fix
		t=0
		t2=0
		#print('hook_fn_count total=',total,' ',module.weight)	
		if 1==0 or (module.weight.shape[0]==16384 and module.weight.shape[1]==2048 ):#適用於llama up
			total=total%18+1
			
			#print('hook_fn_count total=',total,' weifht',module.weight.shape, ' output:',output.shape)
								
			neuron2+=module.weight.shape[0]
			for z in range(module.weight.shape[0]):#遍歷layer中的每個神經元
				g=ac[total][z]/atotal[total][z] #用已經統計的數據，得到neuron活躍度 (活躍的概率)
				
				if g<=0:
					h=level[0]
				elif g>=0.1:
					h=level[101]
				else:
					h=level[(int)(g*1000)+1]
					
				if dropout_method=='sin':
					gh=ch_sin(h)#
				elif dropout_method=='cos':
					gh=ch_cos(h)#
				elif dropout_method=='fr':
					gh=ch_f(g)#		
				elif dropout_method=='linear':#gh=ch_56(h)
					gh=ch(h)#42	
				else:
					gh=0
					
					
								
				#print('hook_fn_count total=',total,' gh=',gh,' cp1[total][z]=',cp1[total][z])
			
				cp1[total][z]=1
				if random.randint(1, 1001)<gh*10:
					#module.weight.data[z,:]=module.weight.data[z,:]*0.95 #weight_fix
					#module.bias.data[z]=module.bias.data[z]-bias_fix
					t+=module.weight.shape[1]*2
					t2+=1
					cp1[total][z]=0

			neuron+=module.weight.shape[0]*module.weight.shape[1]*2				
			inactivate+=t
			inactivate2+=t2
						
	def hook_fn_train(module, input): # train過程中，隨機dropout冷神經元，須配合  hook_fn_count
		global total, num, ac, atotal, cp1, tmp_weight, all_weight, dropout_method
		new_input = input[0].clone()  # gemma-2b 的 fc2(down)	
		
		
		total=total%18+1
		#print('hook_fn_train total=',total,' wight:',module.weight.shape,' input:',input[0].shape )

		#print('tmp_sum=',tmp_weight,', this=',gg,', ',g)
		#print('normal_dropout=',normal_dropout)
		#normal_dropout=0.1 #
	
				
		if dropout_method=='normal':
			cp1[total]=np.ones(16384)
		cp1_tensor = torch.from_numpy(cp1[total]).to('cuda')
		cp1_tensor = cp1_tensor.to(new_input.dtype)
		
		if dropout_method=='normal':
			random_mask = (torch.rand_like(cp1_tensor) > 0.45).to(new_input.dtype)
			cp1_tensor = cp1_tensor * random_mask
		#print('cp1: ',cp1_tensor)
		#print('ch: ', ((normal_dropout-1)/101))

		keep_ratio = cp1_tensor.mean()
		new_input = (new_input * cp1_tensor)#/ (keep_ratio + 1e-8)

				
		new_input = (new_input,)  # 
		return  new_input
		
	def train_all_layers(name,module,deep, prefix=""): #訓練時對冷神經元做出處理，掛上hook_fn_train
		n=0
		global hook
		for name, module in module.named_children():
			#n+=1
			train_all_layers(name,module,deep+1)
		if 1==1 and hasattr(module, 'weight') and isinstance(module, torch.nn.Linear) and ('down_proj' in name):
			#print('count!!  deep=',deep,' module=',module ,' name=',name )
			#hh=module.register_forward_hook(hook_fn_train)
			hh=module.register_forward_pre_hook(hook_fn_train)
			hook.append(hh)	
					
						
	def remove_all_layers(): 
		for k in hook:
			k.remove()
							
	def count_all_layers(name,module,deep, prefix=""): #訓練時對冷神經元做出處理，加入hook_fn_count
		n=0
		global hook
		for name, module in module.named_children():
			n+=1
			count_all_layers(name,module,deep+1)

		if n==0 and hasattr(module, 'weight') and isinstance(module, nn.Linear) and ('up_proj' in name):
			#print('count!!  deep=',deep,' module=',module ,' name=',name )
			hh=module.register_forward_hook(hook_fn_count)
			hook.append(hh)
				
	def print_time(st):
		now_time = time.time()-start_time  # 取得當前時間（秒數，從1970-01-01開始）			
		min0=(int)(now_time/60)
		hr=(int)(min0/60)
		min0-=hr*60
		ss=(int)(now_time%60)
		print("time: ",hr,' hour, ',min0,' min, ',ss,"s,  ",time.time())
		
	n=0
	for param in model.parameters():
		break
		print('n=',n,' parm=',param,' ',param.shape)
		n+=1	
					
	
# 讀檔(之前統計後存下的)
		
	ac = np.load("train_gemma_ac.npy")
	atotal = np.load("train_gemma_atotal.npy")
	remove_all_layers()	#去除統計經元活度的hook

	input_ids0 = torch.tensor([[1694]])	
	attention_mask0 = torch.tensor([[1]])	
	model.eval() # 設置模型為評估模式 (會影響 Dropout 和 BatchNorm 等層的行為)

	if 1==1: # 訓練時對冷神經元處理
		total=0
		for name, module in model.named_children():
			count_all_layers(name,module,0)			
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

	#print('\n all_wiight=',all_weight)		
	if 1==0: #測試用 不重要
		inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
		output = model(input_ids=inputs.input_ids.to('cuda'), attention_mask=inputs.attention_mask.to('cuda')).logits
		#print('\n\n---output---\n\n')
		#print(output)#print(output.shape)
		generated_tokens = torch.argmax(output, dim=-1)
		generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
		#generated_text=inference(model,input_text)
		#print('output= ',generated_text)
	
	p=100*inactivate/neuron
	p2=100*inactivate2/(neuron2)
	if 1==0 and print0: #輸出不同活躍度區間的神經元比例
		print_time(start_time)
		print("\nlevel_count:")
		n=0
		c=0
		for z in  range(102):
			if z%10==0 or z==101:
				print('level ',z," ",(z/1000)," percent:",round(level[z],4),"%")
				
		#print('total neuron=',neuron,' inactivate neuron=',inactivate)
		#print('inactivate percent=', "{:.3f}".format(p))
	
	if print0 and dropout_method!='normal':
		print('inactivate2 percent==', "{:.3f}".format(p2))
		
	return model



def check_start(model,input_text,input_text_train,tokenizer,print0=True,count=False): #主要函式，掛hook並實際處理用
	
	global hook, total, neuron, inactivate,neuron2,inactivate2, first, num, ac, atotal, tmp_ac, tmp_atotal, an, cp1,copy_neuron, level

	global weight_fix, bias_fix	
	
	total=0
	neuron=1
	inactivate=0
	inactivate2=0 
	neuron2=1
	level=np.zeros((102))
	
	def hook_fn(module, input, output):  #統計神經元活度度的 hook
	
		global total, neuron, inactivate, first, num, ac, atotal, level, an,copy_neuron, ts, sd
		if 1==0 or (module.weight.shape[0]==2048  and module.weight.shape[1]==16384) :# down
			total+=1
			if total>18: # model的最後一個此種layer
				num+=1
				total=1
			#print('\nhook_fn num:',num,' total:',total,"input_shape: ",input[0].shape)
		
			for t in range(input[0].shape[0]):#遍歷layer中的每個batch
				for z in range(input[0].shape[2]):#遍歷layer中的每個神經元
					er=0
					for i in range(0, input[0].shape[1]): #遍歷inference時的每個輸入token
						if input[0][t][i][z]>0:  #-module.bias[z]:
							er+=1
							#break		
					an+=1 #第幾次統計
					atotal[total][z]=atotal[total][z]+input[0].shape[1]  #total是第幾層 z是該曾第幾神經元
					ac[total][z]=ac[total][z]+er
	
	def hook_fn_count(module, input, output): #train時，對隨機到的冷神經元處理(非統計)
		global total, neuron, inactivate, inactivate2, neuron2, first, num, ac, atotal, level, cp1
		global weight_fix, bias_fix
		t=0
		t2=0
		#print('hook_fn_count total=',total,' ',module.weight)	
		if 1==0 or (module.weight.shape[0]==16384 and module.weight.shape[1]==2048 ):#適用於llama up
			total=total%18+1			
			#print('hook_fn_count total=',total,' weifht',module.weight.shape, ' input:',input[0].shape)
			#print('hook_fn_cancel fc1 total=',total,' ',input[0].shape,' ',ac[total][0],' ',atotal[total][0],' g=',ac[total][0]/atotal[total][0])	
			for z in range(module.weight.shape[0]):#遍歷layer中的每個神經元
				g=ac[total][z]/atotal[total][z] #用已經統計的數據，得到neuron活躍度 (活躍的概率)
				if g<=0:
					level[0]+=1              #紀錄有多少神經元在某個活躍度的區間
				elif g>=0.1:
					level[101]+=1
				else:
					level[(int)(g*1000)+1]+=1
					
									
	def remove_all_layers(): 
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
			
	def count_all_layers(name,module,deep, prefix=""): #訓練時對冷神經元做出處理，加入hook_fn_count
		n=0
		global hook
		for name, module in module.named_children():
			n+=1
			count_all_layers(name,module,deep+1)

		if n==0 and hasattr(module, 'weight') and isinstance(module, nn.Linear) and ('up_proj' in name):
			#print('count!!  deep=',deep,' module=',module ,' name=',name )
			hh=module.register_forward_hook(hook_fn_count)
			hook.append(hh)
			
	def print_time(st):
		now_time = time.time()-start_time  # 取得當前時間（秒數，從1970-01-01開始）			
		min0=(int)(now_time/60)
		hr=(int)(min0/60)
		min0-=hr*60
		ss=(int)(now_time%60)
		print("time: ",hr,' hour, ',min0,' min, ',ss,"s,  ",time.time())
		
	n=0
	for param in model.parameters():
		break
		print('n=',n,' parm=',param,' ',param.shape)
		n+=1	

	if count: # 如果需要統計 神經元的活躍度
		#np.save("test/test.npy", ac)
		ac=np.zeros((33, 17384))
		atotal=np.zeros((33, 17384))
		an=0
		#total=0
		for name, module in model.named_children(): #掛上統計經元活度的hook
			print_all_layers(name,module,0)
		i=0
		for t in input_text_train:
			print_time(start_time)
			print('t=',i,'/',len(input_text_train))#,' ,' t=',t text=',t
			i+=1
			if i>6: #只統計文檔中的前五段文字(因為統計過程很耗時間)
				break
			inputs_train = tokenizer(t, return_tensors="pt").to('cuda')
			
			output = model(input_ids=inputs_train.input_ids.to('cuda'), attention_mask=inputs_train.attention_mask.to('cuda')).logits	 #透過inference，實際開始統計經元活度	
		print_time(start_time)		
		np.save("train_gemma_ac.npy", ac)
		np.save("train_gemma_atotal.npy", atotal)
		print('save ac, ac=')
		#print('save atotal, atotal=',atotal)
			
	else: # 讀檔(之前統計後存下的)
		print('---load and not  ---')
		ac = np.load("train_gemma_ac.npy")
		atotal = np.load("train_gemma_atotal.npy")
	
	remove_all_layers()	#去除統計經元活度的hook
	
		
	input_ids0 = torch.tensor([[1694]])	
	attention_mask0 = torch.tensor([[1]])	
	model.eval() # 設置模型為評估模式 (會影響 Dropout 和 BatchNorm 等層的行為)

	if 1==1: # 訓練時對冷神經元處理
		total=0
		for name, module in model.named_children():
			count_all_layers(name,module,0)			
		output = model(input_ids=input_ids0.to('cuda'), attention_mask=attention_mask0.to('cuda')).logits
			
	remove_all_layers()	
	p=100*inactivate/neuron
	p2=100*inactivate2/(neuron2)
	if 1==1: #輸出不同活躍度區間的神經元比例
		print_time(start_time)
		print("\nlevel_count:")
		n=0
		c=0
		for z in  range(102):
			n+=level[z]
		for z in  range(102):
			c+=level[z]
			level[z]=c*100/n
			if z%10==0 or z==101:
				print('level ',z," ",(z/1000)," percent:",round(c*100/n,4),"%")

		
	return model
	       
def inference(model,input_text):
	inputs = tokenizer(input_text, return_tensors="pt")
	model.eval()
	#print("\n\n-------------------------\n")
	#print("input:\n")
	#print(input_text)
	with torch.no_grad():	
	    outputs = model.generate(input_ids=inputs.input_ids.to('cuda'),
	    	attention_mask=inputs.attention_mask.to('cuda'),
		max_length=inputs.input_ids.shape[1]+12,
		#temperature=0,
		num_return_sequences=1,
		pad_token_id=tokenizer.eos_token_id)  # Ensure to use the appropriate pad token id)

	generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
	#print("\n\noutput:\n")
	#print(generated_text)
	#print("\n\n-------------------------\n")
	return generated_text
	
import torch.nn as nn	
from transformers import TrainerCallback


		
class ReloadModelCallback(TrainerCallback): #訓練時對冷神經元處理
    def __init__(self, trainer):
        self.trainer = trainer
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps-1:
        	print("\n\n***End load***\n\n")
        	check(self.trainer.model, input_text, data_train, tokenizer, True) 
        	return	
        	#self.trainer.model.train()  # 确保模型处于训练模式
        global l1, l1_lr, l1_real_lr
        
        if l1=='yes':
        	h=0.5*(math.sin((-0.5+state.global_step/state.max_steps)*math.pi)+1)
        	l1_real_lr=l1_lr/10+h*(l1_lr*0.9)
	       
        	if state.global_step % 50 == 0 or state.global_step<4:
        		print('l1_real_lr=',l1_real_lr,'   l1_lr=',l1_lr)
        		
        self.trainer.model.train()  # 确保模型处于训练模式	
        if state.global_step % 2 != 0 or ch=='no':
        	if state.global_step % 200==0:
        		print('no check')
        	return
        	
        	
        if state.global_step % 500==0 or state.global_step<10:
        	check(self.trainer.model, input_text, data_train, tokenizer, True) 
        else:  
        	check(self.trainer.model, input_text, data_train, tokenizer, False) #掛上hook處理冷神經元
        self.trainer.model.train()  # 确保模型处于训练模式
       
nnn=0  
import torch.nn.functional as F  
class SparsityTrainer(Trainer):

   
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    	tmp=0
    	hook=[]	
    		
    	def train_all_layers(name,module,deep, prefix=""): #訓練時對冷神經元做出處理，加入hook_fn_train
    		nonlocal hook
    		for name, m in module.named_children():
    			train_all_layers(name,m,deep+1)
    		if isinstance(module, torch.nn.Linear) and ('down_proj' in name):
    			hh=module.register_forward_hook(hook_fn_train)
    			hook.append(hh)	
    	def hook_fn_train(module, input,output): # train過程中，隨機dropout冷神經元，須配合  hook_fn_count		
    		#print('compute_loss hook',' input:',input[0].shape,' ',input[0] )
    		nonlocal tmp
    		g=input[0].sum()
    		gg=torch.abs(g)
    		tmp+=gg	
    		#print('x: ',input[0])
    		#x_dropped = F.dropout(input[0], p=0.1, training=module.training)
    		#print('x_dropped: ',x_dropped)
    		#return (x_dropped,)
    		
    		#print('gg=',(int)(gg),' tmp=',(int)(tmp))
    	for name, module in model.named_children():
    		train_all_layers(name,module,0)
		
    	loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)   	
    	fix_r=l1_real_lr
    	fix_tmp=tmp*fix_r
    	global nnn
    	nnn=nnn+1
    	if nnn%300==1:
    		print('\nTHIS loss=',loss,' tmp=',tmp,' fix_tmp=',fix_tmp,'\n')
    	loss=loss+fix_tmp
    	
    	for handle in hook:
    		handle.remove()
    	return (loss, outputs) if return_outputs else loss
        
        
        
   
def train_model(model,output_name, my_evel_step, my_train_data,my_test_data,real_train,my_lr):

    torch.autograd.set_detect_anomaly(True)
    model.train()
    data_collator =   DataCollatorForSeq2Seq(tokenizer, model=model, 
    	padding=True, label_pad_token_id=-100)
    
    #data_collator = DefaultDataCollator() # output/mini_1000_baseline_model mini_1000_expand2_model
    output_path = "output/" +output_name   # opt_model   

    print("model: ",output_path)
    args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=4,  # Reduced batch size 4
        per_device_eval_batch_size=4,  # Reduced batch size
        eval_strategy="steps",
        eval_steps=300,
        gradient_accumulation_steps=4,  # Increased accumulation steps
        num_train_epochs=1,
        weight_decay=1e-3,#1e3
        #max_steps=100,
        warmup_steps=20,  # Increased warmup steps
        lr_scheduler_type="cosine",
        learning_rate=my_lr,#my_lr,  #3e-5 baseline 3e-5 Adjusted learning rate
        #save_total_limit=1,
        #max_grad_norm=1.0, 
        save_strategy="steps",
        save_steps=200000,
        deepspeed="deepspeed2.json",  # Path to DeepSpeed config file
        report_to="none",
        #do_train=False,  
        bf16=True,
        #fp16=True
    )
    
    if l1=='yes':
    	print("train with l1");
    	trainer = SparsityTrainer(  #Trainer(
	    #trainer = Trainer(
		model=model,
		tokenizer=tokenizer,
		args=args,
		data_collator=data_collator,
		train_dataset=my_train_data, #tokenized_data["train"],
		eval_dataset=my_test_data, #tokenized_data["test"],
	    )     
    else:
    	print("no l1");
    	trainer = Trainer(
		model=model,
		tokenizer=tokenizer,
		args=args,
		data_collator=data_collator,
		train_dataset=my_train_data, #tokenized_data["train"],
		eval_dataset=my_test_data, #tokenized_data["test"],
	    )

   
    trainer.add_callback(ReloadModelCallback(trainer))
    
    if real_train=='yes':
    	trainer.train()
    	print('train end')
    else:
    	print('pass train ')
    	
    		#trainer2.train()
    	
   
    print('\nstart save at ',output_path,'\n\n')
    trainer.save_model(f"{output_path}/final_model")
    #trainer.evaluate()
    
    return trainer.model
    
import argparse    
start_time = time.time()  # 取得當前時間（秒數，從1970-01-01開始）

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
tokenizer.pad_token = tokenizer.eos_token

input_text="How many people in the "
input_text="Following Viking raids and settlement in the ninth"
input_text="the Anglo-Saxon kingdom of Wessex emerged as the dominant English"
input_text="Kyiv and Moscow are not known to have held direct talks at"

if 1==1:
	with open("data.txt", "r") as f:
		data_train = f.readlines()  # Reads all lines into a list

weight_fix=0
bias_fix=0
ch=True	
from transformers import AutoModel
from transformers.activations import GELUActivation

def replace(model):
	print('replace to Relu')
	replace_count=0
	for name, module in model.named_modules():
	    #print('name:',name,' m=',module)	
	    if isinstance(module,  GELUActivation):
	    	#print('eulu:', module)
	    	parts = name.rsplit('.', 1)
	    	if len(parts) == 2:
	    		parent_name, child_name = parts[0], parts[1]
	    		parent_module = model
	    		for part in parent_name.split('.'):
	    			parent_module = getattr(parent_module, part)
	
	    		setattr(parent_module, child_name, nn.ReLU().to(model.device)) 
	    		replace_count += 1
	    		#print(f"替換: {name} 從 GELUActivation 到 nn.ReLU")
	    	else:
	    		print(f"警告: 發現頂層激活函數 '{name}'，未替換。")
	model.config.hidden_act = 'relu'
	model.config.hidden_dropout = 0.1
	
	print(f"\n總共替換了 {replace_count} 個激活函數。")
	return model



def main(args):
	global weight_fix, bias_fix, ch, dropout_method, l1, l1_lr, l1_real_lr, normal_dropout
	if args.load=='no':
		print('google/gemma-2b')
		#model = AutoModelForCausalLM.from_pretrained("SparseLLM/ReluLLaMA-7B", device_map="cuda", torch_dtype=torch.bfloat16)
		
		#model = AutoModelForCausalLM.from_pretrained("SparseLLM/prosparse-llama-2-7b", device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True)		
		
		model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="cuda",torch_dtype=torch.bfloat16)	
		if args.relu=='yes':
			replace(model)	
		
	else:
		print('my model')
		model = AutoModelForCausalLM.from_pretrained("output/"+args.input+"/final_model",torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)  #final_model   #checkpoint-500 torch.bfloat16

	
	weight_fix = args.weight_fix
	bias_fix = args.bias_fix
	ch = args.check
	dropout_method = args.dropout_method
	l1= args.l1
	l1_lr= args.l1_lr
	l1_real_lr=0
	
	print(f"load: {args.load}")
	print(f"check: {ch}, train: {args.train}")
	print(f"l1: {args.l1}, l1_lr={l1_lr}")	
	print(f"Learning Rate: {args.lr}")
	print(f"dorpout_method: {dropout_method}")
	print(f"input_model: {args.input}, output_model: {args.output}")
		
	print(model)	
	show_num(model)	
	
	real_output=inference(model,input_text)
	#model_copy = copy.deepcopy(model)  # 深複製模型
	#model=check(model,input_text,data_train,tokenizer)
	#return 
		
	if ch=='no':
		print('no check')
	elif args.recount=='yes':
		print('check count')
		model=check_start(model, input_text, data_train, tokenizer, True,True)
	elif args.train=='yes':
		print('check no count')
		model=check_start(model,input_text,data_train,tokenizer, True,False)
	else:
		print('check only')
		model=check_start(model,input_text,data_train,tokenizer, True,False)
	
	print('input:', input_text)
	print('real_output:', real_output)
	inference(model,input_text)


	if 1==1 and args.train=='yes':
		#train_data=load_dataset('Salesforce/wikitext', "wikitext-2-raw-v1",split="train[:40%]")
		if args.data=='piqa':
			train_data=load_dataset("baber/piqa",split="train[:40%]")
			test_data=load_dataset("baber/piqa",split="test[:200]")
			tokenized_train_data = train_data.map(tokenize4_piqa)
			tokenized_test_data = test_data.map(tokenize4_piqa)	
		elif args.data=='alpaca':
			train_data=load_dataset("yahma/alpaca-cleaned",split="train[:40%]")
			test_data=load_dataset("yahma/alpaca-cleaned",split="train[:200]")
			tokenized_train_data = train_data.map(tokenize4)
			tokenized_test_data = test_data.map(tokenize4)						
		elif args.data=='alpaca_short':
			train_data=load_dataset("yahma/alpaca-cleaned",split="train[:5%]")
			test_data=load_dataset("yahma/alpaca-cleaned",split="train[:200]")
			tokenized_train_data = train_data.map(tokenize4)
			tokenized_test_data = test_data.map(tokenize4)		
		elif args.data=='wikitext':
			train_data=load_dataset('Salesforce/wikitext', "wikitext-2-raw-v1",split="train[:10%]")
			tokenized_train_data = train_data.map(tokenize4_wiki)
			test_data=load_dataset('Salesforce/wikitext', "wikitext-2-raw-v1",split="train[:200]")
			tokenized_test_data = test_data.map(tokenize4_wiki)
		elif args.data=='swag':
			#train_data=load_dataset("allenai/c4", "en", split="train[:100%]", streaming=True)
			train_data=load_dataset("allenai/swag",split="train[:10%]")
			tokenized_train_data = train_data.map(tokenize4_swag)
			#print('data=',tokenized_train_data['labels'][:100%])
			test_data=load_dataset("yahma/alpaca-cleaned",split="train[:200]")
			tokenized_test_data = test_data.map(tokenize4)
			print('test_data=',tokenized_test_data['labels'][:20])
		elif args.data=='siqa':
			train_data=load_dataset('lighteval/siqa', split="train[:50%]")
			tokenized_train_data = train_data.map(tokenize4_siqa)
			test_data=load_dataset('lighteval/siqa', split="validation[:200]")
			tokenized_test_data = test_data.map(tokenize4_siqa)
		elif args.data=='siqa_short':
			train_data=load_dataset('lighteval/siqa', split="train[:5%]")
			tokenized_train_data = train_data.map(tokenize4_siqa)
			test_data=load_dataset('lighteval/siqa', split="validation[:200]")
			tokenized_test_data = test_data.map(tokenize4_siqa)
																		
		else:
			print('\n\n\n*****  error **** \n\n ***** dataset error *****\n\n')

	
		print('train data= ',args.data)
		model=train_model(model,args.output,10, tokenized_train_data,tokenized_test_data, args.train, args.lr)	
		print('end train')
		if ch=='yes':
			model=check(model, input_text, data_train, tokenizer, True)
	
	
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with given parameters.")

    parser.add_argument("-load", type=str, default='yes', help="load local Model or not")
    parser.add_argument("-train", type=str,default='yes', help="train or not")
    parser.add_argument("-recount", type=str,default='yes', help="recount or not")
    parser.add_argument("-lr", type=float, default=3e-5, help="Learning rate ")
    parser.add_argument("-output", type=str,default='gemma_model', help="output model name")
    parser.add_argument("-input", type=str,default='input_model', help="input model name")
    parser.add_argument("-weight_fix", type=float,default=1, help="old trash")
    parser.add_argument("-bias_fix", type=float,default=0, help="old trash")
    parser.add_argument("-check", type=str,default='yes', help="check or not")
    parser.add_argument("-relu", type=str,default='yes', help="relu or not")
    parser.add_argument("-data", type=str,default='alpaca', help="train dataset")
    parser.add_argument("-dropout_method", type=str,default='random', help="random=linear, my method")
    parser.add_argument("-l1", type=str,default='no', help="if start l1 train")
    parser.add_argument("-l1_lr", type=float,default=1e-6, help="l1 Learning rate ")
    
    
    args = parser.parse_args()
    main(args)

	
