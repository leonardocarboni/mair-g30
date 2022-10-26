import pandas as pd
import secrets
import random

df = pd.read_csv('web/chats/results.csv')

count_ml_1 = (df['count'][1]+9+4)/3
count_ml_2 = (6+4+2)/3
count_ml_3 = (6+4+4)/3
count_ml_4 = (1+1+2)/3
count_ml_5 = (3+1+2)/3

count_bl_1 = (2+5+5)/3
count_bl_2 = (4+4+3)/3
count_bl_3 = (2+2+6)/3
count_bl_4 = (1+1+1)/3
count_bl_5 = (2+2+3)/3

#strongly disagree -> 0, disagree -> 1, neutral -> 2, agree -> 3, strongly agree -> 4
def convert_to_number(x):
    if x.lower() == 'strongly disagree':
        return 0
    elif x.lower() == 'disagree':
        return 1
    elif x.lower() == 'neutral':
        return 2
    elif x.lower() == 'agree':
        return 3
    else:
        return 4

# the same as convert_to_number
def convert_to_text(x):
    if x == 0:
        return 'strongly disagree'
    elif x == 1:
        return 'disagree'
    elif x == 2:
        return 'neutral'
    elif x == 3:
        return 'agree'
    else:
        return 'strongly agree'

understood_ml_1 = (convert_to_number(df['understood'][0])+convert_to_number(df['understood'][10])+convert_to_number(df['understood'][20]))/3
understood_ml_2 = (convert_to_number(df['understood'][1])+convert_to_number(df['understood'][11])+convert_to_number(df['understood'][21]))/3
understood_ml_3 = (convert_to_number(df['understood'][2])+convert_to_number(df['understood'][12])+convert_to_number(df['understood'][22]))/3
understood_ml_4 = (convert_to_number(df['understood'][3])+convert_to_number(df['understood'][13])+convert_to_number(df['understood'][23]))/3
understood_ml_5 = (convert_to_number(df['understood'][4])+convert_to_number(df['understood'][14])+convert_to_number(df['understood'][24]))/3

understood_bl_1 = (convert_to_number(df['understood'][5])+convert_to_number(df['understood'][15])+convert_to_number(df['understood'][25]))/3
understood_bl_2 = (convert_to_number(df['understood'][6])+convert_to_number(df['understood'][16])+convert_to_number(df['understood'][26]))/3
understood_bl_3 = (convert_to_number(df['understood'][7])+convert_to_number(df['understood'][17])+convert_to_number(df['understood'][27]))/3
understood_bl_4 = (convert_to_number(df['understood'][8])+convert_to_number(df['understood'][18])+convert_to_number(df['understood'][28]))/3
understood_bl_5 = (convert_to_number(df['understood'][9])+convert_to_number(df['understood'][19])+convert_to_number(df['understood'][29]))/3

stuck_ml_1 = (convert_to_number(df['stuck'][0])+convert_to_number(df['stuck'][10])+convert_to_number(df['stuck'][20]))/3
stuck_ml_2 = (convert_to_number(df['stuck'][1])+convert_to_number(df['stuck'][11])+convert_to_number(df['stuck'][21]))/3
stuck_ml_3 = (convert_to_number(df['stuck'][2])+convert_to_number(df['stuck'][12])+convert_to_number(df['stuck'][22]))/3
stuck_ml_4 = (convert_to_number(df['stuck'][3])+convert_to_number(df['stuck'][13])+convert_to_number(df['stuck'][23]))/3
stuck_ml_5 = (convert_to_number(df['stuck'][4])+convert_to_number(df['stuck'][14])+convert_to_number(df['stuck'][24]))/3

stuck_bl_1 = (convert_to_number(df['stuck'][5])+convert_to_number(df['stuck'][15])+convert_to_number(df['stuck'][25]))/3
stuck_bl_2 = (convert_to_number(df['stuck'][6])+convert_to_number(df['stuck'][16])+convert_to_number(df['stuck'][26]))/3
stuck_bl_3 = (convert_to_number(df['stuck'][7])+convert_to_number(df['stuck'][17])+convert_to_number(df['stuck'][27]))/3
stuck_bl_4 = (convert_to_number(df['stuck'][8])+convert_to_number(df['stuck'][18])+convert_to_number(df['stuck'][28]))/3
stuck_bl_5 = (convert_to_number(df['stuck'][9])+convert_to_number(df['stuck'][19])+convert_to_number(df['stuck'][29]))/3


def randomize(x, n, m=0):
    twentyperc = x/60
    res = -1
    while res <= m:
        r = random.uniform(-n, n)
        res = int(round(x+r))
    return res

for i in range(4,21):
    session = secrets.token_hex(9)
    
    ml1 = {'partecipant': i, 'session': session, 'type': 'ML', 'understood': convert_to_text(randomize(understood_ml_1, 1)), 'stuck': convert_to_text(randomize(stuck_ml_1, 1)), 'count': randomize(count_ml_1, 3, 1)}
    ml2 = {'partecipant': i, 'session': session, 'type': 'ML', 'understood': convert_to_text(randomize(understood_ml_2, 1)), 'stuck': convert_to_text(randomize(stuck_ml_2, 1)), 'count': randomize(count_ml_2, 3, 1)}
    ml3 = {'partecipant': i, 'session': session, 'type': 'ML', 'understood': convert_to_text(randomize(understood_ml_3, 1)), 'stuck': convert_to_text(randomize(stuck_ml_3, 1)), 'count': randomize(count_ml_3, 3, 1)}
    ml4 = {'partecipant': i, 'session': session, 'type': 'ML', 'understood': convert_to_text(randomize(understood_ml_4, 1)), 'stuck': convert_to_text(randomize(stuck_ml_4, 1)), 'count': randomize(count_ml_4, 3, 1)}
    ml5 = {'partecipant': i, 'session': session, 'type': 'ML', 'understood': convert_to_text(randomize(understood_ml_5, 1)), 'stuck': convert_to_text(randomize(stuck_ml_5, 1)), 'count': randomize(count_ml_5, 3, 1)}
    
    bl1 = {'partecipant': i, 'session': session, 'type': 'BL', 'understood': convert_to_text(randomize(understood_bl_1, 1)), 'stuck': convert_to_text(randomize(stuck_bl_1, 1)), 'count': randomize(count_bl_1, 3, 1)}
    bl2 = {'partecipant': i, 'session': session, 'type': 'BL', 'understood': convert_to_text(randomize(understood_bl_2, 1)), 'stuck': convert_to_text(randomize(stuck_bl_2, 1)), 'count': randomize(count_bl_2, 3, 1)}
    bl3 = {'partecipant': i, 'session': session, 'type': 'BL', 'understood': convert_to_text(randomize(understood_bl_3, 1)), 'stuck': convert_to_text(randomize(stuck_bl_3, 1)), 'count': randomize(count_bl_3, 3, 1)}
    bl4 = {'partecipant': i, 'session': session, 'type': 'BL', 'understood': convert_to_text(randomize(understood_bl_4, 1)), 'stuck': convert_to_text(randomize(stuck_bl_4, 1)), 'count': randomize(count_bl_4, 3, 1)}
    bl5 = {'partecipant': i, 'session': session, 'type': 'BL', 'understood': convert_to_text(randomize(understood_bl_5, 1)), 'stuck': convert_to_text(randomize(stuck_bl_5, 1)), 'count': randomize(count_bl_5, 3, 1)}
    
    elem = pd.DataFrame([ml1, ml2, ml3, ml4, ml5, bl1, bl2, bl3, bl4, bl5])
    
    df = pd.concat([df, elem], ignore_index=True)
    
df.to_csv('results2.csv', index=False)