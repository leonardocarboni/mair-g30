import os
import matplotlib as plt

d = {}

for filename in os.listdir(os.getcwd()+"/web/chats"):
    if filename.endswith(".txt"):
        with open(os.path.join(os.getcwd()+"/web/chats", filename), 'r') as f:
            identifier = ""
            counts = []
            counttemp = 0
            for line in f:
                if line.startswith('['):
                    a = line.split(',')
                    identifier = a[0][1:]
                    email = a[1]
                    ml = "True" in a[2]

                elif 'start over' in line:
                    counts.append(counttemp)
                    counttemp = 0
                elif 'USER:' in line:
                    counttemp += 1
            counts.append(counttemp)
            d[identifier] = [email, ml, counts]
            
print(d)

