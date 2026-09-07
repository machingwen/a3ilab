
import re
import sys
import os

def extract_xy(log_text):
    # x = inactivate2 percent（全部）
    x_list = re.findall(r"inactivate2 percent=\s*([0-9.]+)", log_text)
    x_list = [float(x) for x in x_list]

    # y = acc（全部）
    y_list = re.findall(r"acc\s*\|\s*↑\s*\|\s*([0-9.]+)\s*\|±", log_text)
    y_list = [float(y) for y in y_list]

    return x_list, y_list
    
    
def extract_cdf(log_text):
    # x = inactivate2 percent（全部）
    x_list = re.findall(r"neuron percent:\s*([0-9.]+)%", log_text)
    x_list = [float(x) for x in x_list]

    # y = acc（全部）
    y_list = re.findall(r"cdf= \s*([0-9.]+)", log_text)
    y_list = [float(y) for y in y_list]

    return x_list, y_list
def show(d,text):
    t="log/"+d+'.log'
    if not os.path.exists(t):  
    	print("this log not exist")
    else:
	    with open(t, "r") as f:
	    	log = f.read()
	    x, y = extract_xy(log)
	    
	    print(text," ",d)
	    print(f"x={x}")
	    print(f"y={y}")

def show_cdf_bad(d,text):
    t="log/"+d+'.log'
    if not os.path.exists(t):  
    	print("this log not exist")
    else:
	    with open(t, "r") as f:
	    	log = f.read()
	    x, y = extract_cdf(log)
	    print()
	    print(text," ",d)
	    print(f"neuron percent={x}")
	    print(f"cdf={y}")
	    
def show_cdf(d,text):

    t="log/"+d+'.log'
    if not os.path.exists(t):  
    	print("this log not exist")
    	return
    
    with open(t, "r") as f:
	    	log_text = f.read()
    blocks = re.findall(
        r"---load and not\s*---\s*.*?level_count:\s*(.*?)(?=total neuron=)",
        log_text,
        re.DOTALL
    )
    y=" "

    if not blocks:
        y=""
    else:
    	y=  blocks[-1] 
    	
    	print()
    	print(text," ",d)
    	print(y)
	    
    
    	      	
if __name__ == "__main__":

    #show("eval.log")	
    

    
    print("\n-----\n")
    show_cdf("eval_b0","baseline(no dropout, no l1)")
    show_cdf("eval_np","My l1 method")
    show_cdf("eval_l1","old l1  method")
    
    print("x=masked neuron rate, y=acc(siqa)")
    

    show("eval_np","My l1 method")
    show("eval_l1","old l1  method")	
 




