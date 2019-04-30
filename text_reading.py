import pandas as pd

file = '../../projects/learning/landF/LF_Classification.py'
LF_Class = []
with open(file) as temp_file:
    for line in temp_file:
        LF_Class.append(line)

LF_Final = []
for line in LF_Class:
        if ':' in line:
            LF_Final.append(line)

code = []
system_name = []
ct = 0

for line in LF_Final:
    tmp = line.split(':')
    code.append(tmp[0])
    system_name.append(tmp[1].strip())
d = {'system code':code, 'system name':system_name}
df = pd.DataFrame(data = d)
df.to_csv('LandFireSystems.csv',index=False)
