from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import uuid
import time

# Define a memory checkpoint for the agent with built-in database under the hood.
memory = SqliteSaver.from_conn_string(":memory:")
_ = load_dotenv()
tool = TavilySearchResults(max_results=2)


def generate_unique_thread_id():
    unique_part = uuid.uuid4().hex  # Generates a unique UUID
    timestamp_part = str(int(time.time()))  # Current timestamp as an integer
    return f"{timestamp_part}-{unique_part}"


class AgentState(TypedDict):
    """
    Defines the state structure for an agent in the language graph.

    Attributes:
        messages (Annotated[list[AnyMessage], operator.add]): A list of messages that can be operated on.
    """
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    """
    Represents an agent that interacts with a state graph to process messages using OpenAI and custom tools.

    Attributes:
        system (str): An initial system message to prepend to all interactions.
        graph (StateGraph): The compiled state graph for message processing.
        tools (dict): A dictionary mapping tool names to their instances.
        model (ChatOpenAI): The OpenAI model bound with tools for message processing.

    Methods:
        __init__(self, model, tools, system=""): Initializes the agent.
        exists_action(self, state: AgentState): Checks if the last message in the state has any tool calls.
        call_openai(self, state: AgentState): Processes messages through the OpenAI model.
        take_action(self, state: AgentState): Invokes the appropriate tool based on the last message's tool calls.
    """

    def __init__(self, model, tools, checkpointer, system=""):
        """
        Initializes the agent with a model, tools, and an optional system message.

        Parameters:
            model (ChatOpenAI): The OpenAI model to use for processing messages.
            tools (list): A list of tool instances available for the agent.
            system (str, optional): An initial system message. Defaults to an empty string.
        """
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    @staticmethod
    def exists_action(state: AgentState):
        """
        Determines if the last message in the state contains any tool calls.

        Parameters:
            state (AgentState): The current state of the agent.

        Returns:
            bool: True if there are tool calls, False otherwise.
        """
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        """
        Processes messages through the OpenAI model.

        Parameters:
            state (AgentState): The current state of the agent.
        """
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        """
        Processes tool calls from the last message in the state and invokes the corresponding tool.

        This method iterates over each tool call found in the last message of the state. It checks if the tool name
        exists within the agent's tools dictionary. If the tool name is not found, it logs a message indicating a bad
        tool name and sets the result to a retry message. Otherwise, it invokes the tool with the provided arguments
        and collects the results. Each result is wrapped in a ToolMessage object. After processing all tool calls,
        it returns a dictionary containing the list of ToolMessage objects.

        Parameters:
            state (AgentState): The current state of the agent, containing messages among other information.

        Returns:
            dict: A dictionary with a key 'messages' containing a list of ToolMessage objects representing the results
            of the invoked tools.
        """
        tool_calls = state['messages'][-1].tool_calls  # Extract tool calls from the last message
        results = []  # Initialize an empty list to store results
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:  # Check if the tool name exists in the agent's tools
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # Set a retry message for the result
            else:
                result = self.tools[t['name']].invoke(t['args'])  # Invoke the tool with arguments
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))  # Append the result
        print("Back to the model!")
        return {'messages': results}  # Return the results wrapped in a dictionary


def main():
    prompt = """You are a smart research assistant. Use the search engine to look up information. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If a user message is missing context or information you should ask clarifying questions. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """
    debug = True
    chat_gpt_model = 'gpt-3.5-turbo'  # gpt-3.5-turbo, gpt-4o
    model = ChatOpenAI(model=chat_gpt_model)
    abot = Agent(model, [tool], system=prompt, checkpointer=memory)
    print("Sample query: Who was the president of the United States in 1990 and who was their spouse back then")
    thread = {"configurable": {"thread_id": generate_unique_thread_id()}}  # Define a thread ID for the conversation
    while True:
        query = input("Query: ")
        messages = [HumanMessage(content=query)]  # Initialize the messages with the user query
        # to allow for multiple conversations
        if debug:
            for event in abot.graph.stream({"messages": messages}, thread):
                for v in event.values():
                    print(v['messages'])
        else:
            result = abot.graph.invoke({"messages": messages}, thread)
            print(result['messages'][-1].content)


if __name__ == "__main__":
    main()
