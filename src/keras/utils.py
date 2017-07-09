'''
Created on Jun 29, 2017

@author: demon
'''
import numpy as np

class Utils():

    def load_json(self,path):
        with open(path) as json_data:
            json_txt = json_data.readlines()
            json_arr = [txt.strip() for txt in json_txt]
            json_np = np.array(json_arr)
            return json_np
        
    
        
'''            
u = Utils()
u.load_json("dataset/qa_Software.json")
'''