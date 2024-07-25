#  Modified from langgraph tutorial
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
import os
from langchain_core.pydantic_v1 import BaseModel
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

_ = load_dotenv()


PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""


class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    queries: List[str]


class EssayAgent:
    """
    Represent and agent that uses LangGraph to generate an essay.

    Attributes:
        system (str): An initial system message to prepend to all interactions.
        graph (StateGraph): The compiled state graph for message processing.
        search_tool (funct): A function that searches for information based on a query.
        model (ChatOpenAI): The OpenAI model bound with tools for message processing.

    Methods:
        __init__(self, model, tools, system=""): Initializes the agent.
        exists_action(self, state: AgentState): Checks if the last message in the state has any tool calls.
        call_openai(self, state: AgentState): Processes messages through the OpenAI model.
        take_action(self, state: AgentState): Invokes the appropriate tool based on the last message's tool calls.
    """

    def __init__(self, model, search_tool, checkpointer, system=""):
        """
        Initializes the agent with a model, tools, and an optional system message.

        Parameters:
            model (ChatOpenAI): The OpenAI model to use for processing messages.
            search_tool (Callable): A function that searches for information based on a query.
            system (str, optional): An initial system message. Defaults to an empty string.
        """
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("planner", self.plan_node)
        graph.add_node("generate", self.generation_node)
        graph.add_node("reflect", self.reflection_node)
        graph.add_node("research_plan", self.research_plan_node)
        graph.add_node("research_critique", self.research_critique_node)

        graph.set_entry_point("planner")  # Start with the planner node
        graph.add_conditional_edges(
            "generate",
            self.should_continue,
            {END: END, "reflect": "reflect"}
        )
        graph.add_edge("planner", "research_plan")
        graph.add_edge("research_plan", "generate")

        graph.add_edge("reflect", "research_critique")
        graph.add_edge("research_critique", "generate")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.search_tool = search_tool
        self.model = model

    def plan_node(self, state: AgentState):
        messages = [
            SystemMessage(content=PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ]
        response = self.model.invoke(messages)
        return {"plan": response.content}

    def research_plan_node(self, state: AgentState):
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content=RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ])
        content = state['content'] or []
        for q in queries.queries:
            print(f"Query: {q}")
            response = self.search_tool.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}

    def generation_node(self, state: AgentState):
        content = "\n\n".join(state['content'] or [])
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
        messages = [
            SystemMessage(
                content=WRITER_PROMPT.format(content=content)
            ),
            user_message
            ]
        response = self.model.invoke(messages)
        return {
            "draft": response.content,
            "revision_number": state.get("revision_number", 1) + 1
        }

    def reflection_node(self, state: AgentState):
        messages = [
            SystemMessage(content=REFLECTION_PROMPT),
            HumanMessage(content=state['draft'])
        ]
        response = self.model.invoke(messages)
        return {"critique": response.content}

    def research_critique_node(self, state: AgentState):
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state['critique'])
        ])
        content = state['content'] or []
        for q in queries.queries:
            response = self.search_tool.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}

    @staticmethod
    def should_continue(state):
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "reflect"


def main():
    chat_gpt_model = 'gpt-4o'  # gpt-3.5-turbo, gpt-4o
    model = ChatOpenAI(model=chat_gpt_model)
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    memory = SqliteSaver.from_conn_string(":memory:")
    abot = EssayAgent(model=model, search_tool=tavily, checkpointer=memory)
    thread = {"configurable": {"thread_id": "1"}}
    debug = True
    task = input("Enter a topic: ")
    if not debug:
        result = abot.graph.invoke(
            {"task": task, "max_revisions": 2, "revision_number": 1},
            thread
        )
        print(result)
        print(result['draft'])
    else:
        for s in abot.graph.stream({
            'task': task,
            "max_revisions": 2,
            "revision_number": 1,
        }, thread):
            print(s)


if __name__ == "__main__":
    main()
