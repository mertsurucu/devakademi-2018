import json
import codecs
import numpy as np

file = codecs.open("../all_data 3.json", "r", "utf-8")
category=[]
json_str = file.read()
json_data = json.loads(json_str)
for i in range(len(json_data)):
    if json_data[i]["event_category"] not in category:
        category.append(json_data[i]["event_category"])

category = np.array(category)
#TODO write to txt file