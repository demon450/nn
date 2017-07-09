'''
Created on Jun 29, 2017

@author: demon
'''
import numpy as np

def load_qa(path):
    import ast
    with open(path) as json_data:
        json_txt = json_data.readlines()
        json_obj = [ast.literal_eval(x) for x in json_txt]
        #print(json_obj[3]['question'])
        x = np.array(json_obj[::]['question'])
        y = np.array(json_obj[::]['answer'])
        print(x)
            
        return x,y

        
load_qa("dataset/qa_Software.json")  