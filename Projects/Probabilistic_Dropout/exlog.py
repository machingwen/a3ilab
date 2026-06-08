
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
  	
if __name__ == "__main__":

    #show("eval.log")	
    
    print("x=masked neuron rate, y=acc(siqa)")
    show("eval_b0","Old baseline(no dropout, no l1)")
    show("eval_b1","Baseline (normal dropout)")
    show("eval_p1","Probabilistic dropout, cdf, linear")
    
    show("eval_fr","Probabilistic dropout, Activity-Freq")
    show("eval_sin","Probabilistic dropout, sin")
    show("eval_cos","Probabilistic dropout, cos")

    show("eval_1e6","l1-le6")
    show("eval_1e7","l1-le7")
    show("eval_1e8","l1-le8")	
   




