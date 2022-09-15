
"""
Part 1a: text classification
Group G30 (Giacomo Bais, Leonardo Carboni, Merel de Goede, Merel van den Bos)
"""

import pandas as pd
dialog_act = []
utterance_content= []
with open('dialog_acts.dat', 'r') as file:
    for line in file:
        row = line.lower().strip('\n')
        dialog_act.append(row.split(maxsplit = 1)[0])
        utterance_content.append(row.split(maxsplit = 1)[1])
        
d = {'dialog_act': dialog_act, 'utterance_content': utterance_content}
df = pd.DataFrame(data=d)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(df['utterance_content'], \
                                                    df['dialog_act'], \
                                                    test_size=0.15, \
                                                    random_state = 42)

majority = Y_train.mode()[0]

classes = {
    'ack': ['ok', 'okay'],
    'affirm': ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'right', 'indeed'],
    'bye': ['bye', 'goodbye', 'see you', 'see you later', 'talk to you later'],
    'confirm': ['is it', 'is there', 'is that', 'does it', 'do they'],
    'deny': ['dont', 'don\'t', 'do not'],
    'hello': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
    'inform': ['look', 'looking for', 'search', 'find'],
    'negate': ['no'],
    'null': ['cough', 'clear throat', 'laugh', 'sigh', 'sniff'],
    'repeat': ['again', 'repeat', 'what was that', 'what did you say', 'what did you mean'],
    'reqalts': ['how about', 'alternatives', 'other options', 'other choices', 'other suggestions', \
                'another', 'different', 'else', 'other', 'anything else'],
    'reqmore': ['more'],
    'request': ["?", "what"],
    'restart': ['start over', 'restart', 'start again', 'start from the beginning'],
    'thankyou': ['thank you', 'thanks', 'thank you very much', 'thanks very much'],
}


while True:
    print("\nChoose what you want to do:")
    print("1. Baseline 1")
    print("2. Baseline 2")
    print("3. Test")
    print("\n0. Exit")
    choice = input("Enter your choice: ")
    
    if choice == "1":
        """
        Baseline 1:
        A baseline system that, regardless of the content of the utterance, 
        always assigns the majority class of in the data.
        """
        print("Baseline 1 - Write a sentence or a word")

        prompt = input()
        print(majority)

    elif choice == "2":
        """
        Baseline 2:
        A baseline rule-based system based on keyword matching.
        """
        print("Baseline 2 - Write a sentence or a word")

        prompt = input()
        for key, value in classes.items():
            for v in value:
                if v in prompt:
                    print(key)
        else:
            print(majority)
    else:
        break

