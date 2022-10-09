G30 Group project for Methods in AI Research @ Utrecht University.

The system helps the user choose a restaurant based on some features.

# Configurations:

The user will be able to specify:

- Whether to use a Decision tree classifier or the second baseline

- Whether to enable auto correction vie leveinshtein or not (only words that are longer than 3 and have distance <=1 will be corrected)

- Whether the output should be in all caps or not

# How to Use:

After the welcome phase, where the system will greet you, you will be able to start specifying your preferences.
If after the welcome message the user input won't be classified as inform, the system will continue greeting him.

    S: welcome message
    U: Hello
    S: welcome message
    U: i am looking for a chinese restaurant
    ...

You can specify more than one preference directly after the welcome message

    S: welcome message
    U: chinese south cheap
    ...

The system will automatically look for suitable restaurants.
If no restaurants are found it will alert the user about that and await new command.
The user can specify a new preference and the system will automatically check all the suitable restaurants again.

If just one restaurant is found, the system will print the name right away.

    S: welcome message
    U: chinese north
    S: the hotpot is a nice place in the north of town serving chinese food.

If more than one restaurant is found the system will ask for additional requirements and choose the restaurant that suits the request the best. If more than one requirement is given, only the lst one will be considered.

    S: Hello, welcome to the MAIR G30 restaurant system? You can ask for restaurants by area, price range or food type. You can also add additional requirements among: Romantic; Touristic; Children; Assigned seats. How may I help you?
    U: I want to eat chinese food
    S: What area would you like to eat in?
    U: in the south part of town
    S: What price range do you prefer?
    U: i'd prefer if it was cheap
    S: Do you have additional requirements?
    U: i'm with my family so a children suitable restaurant if possible
    S: the missing sock is a nice place in the south of town in the cheap price range serving chinese food. The restaurant is for children because it allows you to stay for a short time.


After finding a restaurant, the user will be able to ask for more details about it (cellphone, postcode and address)

    U: what's the post code
    S: The post code of the hotpot is c.b 4.

If the user's input will be categorized as `bye` anytime during the execution of the program, the system will shut down.
After that a new session can be started.