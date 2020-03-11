import json


with open("generated_phrases.json",'r') as infile:
    with open("fixed_generated_phrases.json",'w') as outfile:
        data = json.load(infile)

        fixed_dict = {}

        for k,v in data.items():
            if v[-1] == ".":
                fixed_dict[k] = v
            else:
                v = ''.join(v.rpartition('.')[:2])
                fixed_dict[k] = v

        json.dump(fixed_dict, outfile)

