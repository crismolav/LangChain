import re
import ast
from openai import OpenAI
from dotenv import load_dotenv

_ = load_dotenv()

client = OpenAI()

prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
If you are not sure of an action, you can ask the user for input using action ask_user.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

get_price:
e.g. get_price: pencil
Returns the price of an item when given the name of the item

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

sql_query:
e.g. sql_query: ("brand", ["cocacola", "pepsi"], 0.5)
returns a string "New Scenario created" string   - if you are missing any information call ask_user action

ask_user: 
e.g. ask_user: What percentage do you want to change the prices by?
returns the user input.

Example session:

Question: How much do 2 Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs
Thought: I should calculate the weight of 2 bulldogs
Action: calculate: 51 * 2
PAUSE

You will be called again with this:
Observation: 102

You then output:

Answer: 2 bulldogs weight 102 lbs

Example session:

Question: Increase all prices of brand cocacola and pepsi by 10%
Thought: I should create a new sql query using sql_query
Action: sql_query: ("brand", ["cocacola", "pepsi"], 0.1)
PAUSE

You will be called again with this:
Observation: New Scenario created.

You then output:

Answer: New Scenario created.

Example session:

Question: Increase all prices of brand coca cola and pepsi.
Thought: I'm missing the percentage to create a new sql query.
Action: ask_user: What percentage do you want to change the prices by?
PAUSE

You will be called again with this:
Observation: By -5 percent
Thought: I should create a new sql query.
Action: sql_query: ("brand", ["cocacola", "pepsi"], -0.05)

You then output:

Answer: New Scenario created.
""".strip()


class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # "gpt-4o",
            temperature=0,
            messages=self.messages)
        return completion.choices[0].message.content


def ask_user(user_question):
    return input(f"{user_question}:")


def calculate(what):
    return eval(what)


def average_dog_weight(name):
    if name in "Scottish Terrier":
        return ("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return ("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return ("a toy poodles average weight is 7 lbs")
    else:
        return ("An average dog weights 50 lbs")


def get_price(name):
    if name in "pencil":
        return "A pencil costs 1.0 Euros"
    elif name in "notebook":
        return "A notebook costs 2.5 Euros"
    else:
        return "An average item costs 3.5 Euros"


def sql_query(column: str, values: list[str], change_price: float) -> str:
    # makes sure to format all the values to the ones stored in the database
    return "New Scenario created"


known_actions = {
    "calculate": calculate,
    "get_price": get_price,
    "average_dog_weight": average_dog_weight,
    "sql_query": sql_query,
    "ask_user": ask_user
}


def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    action_re = re.compile('^Action: (\w+): (.*)$')  # python regular expression to selection action
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a)
            for a in result.split('\n')
            if action_re.match(a)
        ]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()

            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            if action == 'sql_query':
                action_input = ast.literal_eval(action_input)
                observation = known_actions[action](*action_input)
            else:
                observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:

            return


question1 = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
question2 = """I have 3 items, 2 pencils and a notebook. \
What is their combined cost?"""
question3 = """Create a scenario where you increase prices for pepsi brand"""
print(f"Question: {question2}")
query(question2)

