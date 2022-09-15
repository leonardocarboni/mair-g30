
"""
Part 1a: text classification
Group G30 (Giacomo Bais, Leonardo Carboni, Merel de Goede, Merel van den Bos)
"""

from sklearn.model_selection import train_test_split
import pandas as pd
dialog_act = []
utterance_content = []

d = pd.read_csv('dialog_acts.dat', header=None)
df = pd.DataFrame(data=d)
df.columns = ['dialog_act']

df[['dialog_act', 'utterance_content']
   ] = df.dialog_act.str.split(' ', 1, expand=True)


X_train, X_test, Y_train, Y_test = train_test_split(df['utterance_content'],
                                                    df['dialog_act'],
                                                    test_size=0.15,
                                                    random_state=42)

majority = Y_train.mode()[0]

classes = {
    'ack': ['ok', 'okay', "fine", "all right"],
    'affirm': ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'right', 'indeed'],
    'bye': ['bye', 'goodbye', 'see you', 'see you later', 'talk to you later'],
    'confirm': ['is it', 'is there', 'is that', 'does it', 'do they'],
    'deny': ['dont', 'don\'t', 'do not', 'does not', 'is not', 'was not', 'were not', 'are not', 'cannot', 'can not', 'cant', 'can\'t', 'no', 'nope', 'not', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor', 'never', 'no one', 'none', 'not a single', 'not any', 'not any more', 'not at all', 'not either', 'not at all', 'not ever', 'not in the least', 'not less than', 'not more than', 'not at all', 'not quite', 'not so', 'not yet', 'hardly', 'scarcely', 'barely', 'rarely', 'seldom', 'little', 'less', 'least', 'few', 'fewer', 'fewest', 'short', 'shorter', 'shortest', 'low', 'lower', 'lowest', 'small', 'smaller', 'smallest', 'tiny', 'tiniest', 'young', 'younger', 'youngest', 'little', 'less', 'least', 'few', 'fewer', 'fewest', 'short', 'shorter', 'shortest', 'low', 'lower', 'lowest', 'small', 'smaller', 'smallest', 'tiny', 'tiniest', 'young', 'younger', 'youngest', 'little', 'less', 'least', 'few', 'fewer', 'fewest', 'short', 'shorter', 'shortest', 'low', 'lower', 'lowest', 'small', 'smaller', 'smallest', 'tiny', 'tiniest', 'young', 'younger', 'youngest', 'little', 'less', 'least', 'few', 'fewer', 'fewest', 'short', 'shorter', 'shortest', 'low', 'lower', 'lowest', 'small', 'smaller', 'smallest', 'tiny', 'tiniest', 'young', 'younger', 'youngest', 'little', 'less', 'least', 'few', 'fewer', 'fewest', 'short', 'shorter', 'shortest', 'low', 'lower', 'lowest', 'small', 'smaller', 'smallest', 'tiny', ],
    'hello': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
    'inform': ['look', 'looking for', 'search', 'find', 'want', 'need', 'require', 'requirement', 'requirements', 'want to', 'need to', 'require to', 'requirement to', 'requirements to', 'want to find', 'need to find', 'require to find', 'requirement to find', 'requirements to find', 'want to look', 'need to look', 'require to look', 'requirement to look', 'requirements to look', 'want to search', 'need to search', 'require to search', 'requirement to search', 'requirements to search', 'want to find', 'need to find', 'require to find', 'requirement to find', 'requirements to find', 'want to look', 'need to look', 'require to look', 'requirement to look', 'requirements to look', 'want to search', 'need to search', 'require to search', 'requirement to search', 'requirements to search', 'want to find', 'need to find', 'require to find', 'requirement to find', 'requirements to find', 'want to look', 'need to look', 'require to look', 'requirement to look', 'requirements to look', 'want to search', 'need to search', 'require to search', 'requirement to search', 'requirements to search', 'want to find', 'need to find', 'require to find', 'requirement to find', 'requirements to find', 'want to look', 'need to look', 'require to look', 'requirement to look', 'requirements to look', 'want to search', 'need to search', 'require to search', 'requirement to search', 'requirements to search', 'want to find', 'need to find', 'require to find', 'requirement to find', 'requirements to find', 'want to look', 'need to look', 'require to look', 'requirement to look', 'requirements to look', 'want to search', 'need to search', 'require to search', 'requirement to search', 'requirements to search', 'want to find', 'need to find', 'require to find', 'requirement to find', 'requirements to find', 'want to look', 'need to look', 'require to look', 'requirement to look', 'requirements to look', 'want to search', 'need to search', 'require to search', 'requirement to search', ],
    'negate': ['no', 'nope', 'not', 'never', 'none', 'nothing'],
    'null': ['cough', 'clear throat', 'laugh', 'sigh', 'sniff'],
    'repeat': ['again', 'repeat', 'what was that', 'what did you say', 'what did you mean'],
    'reqalts': ['how about', 'alternatives', 'other options', 'other choices', 'other suggestions',
                'another', 'different', 'else', 'other', 'anything else'],
    'reqmore': ['more', 'else', 'another', 'different', 'anything else'],
    'request': ["?", "what", "pub", "restaurant", "hotel", "attraction", "train", "taxi", "plane"],
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
        print("\nBaseline 1 - Write a sentence or a word")

        prompt = input()
        print(majority)

    elif choice == "2":
        """
        Baseline 2:
        A baseline rule-based system based on keyword matching.
        """
        print("\nBaseline 2 - Write a sentence or a word")

        prompt = input()
        for key, value in classes.items():
            for v in value:
                if v in prompt:
                    print(key)
    elif choice == "3":
        """
        Testing baseline 2.
        """
        count = 0
        for i, x in enumerate(X_test):
            for key, value in classes.items():
                for v in value:
                    if v in x:
                        if key == Y_test.iloc[i]:
                            count += 1
                            break
                        else:
                            if majority == Y_test.iloc[i]:
                                count += 1
        print(count/len(X_test))
    else:
        break