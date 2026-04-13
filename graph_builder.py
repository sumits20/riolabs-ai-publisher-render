from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools.wordpress_tools import (
    get_recent_posts_tool,
    create_draft_post_tool,
)
from tools.tavily_research import research_topic
from tools.content_tools import (
    choose_best_topic_tool,
    write_article_gpt_tool,
    write_article_claude_tool,
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


tools = [
    get_recent_posts_tool,
    research_topic,
    choose_best_topic_tool,
    write_article_gpt_tool,
    write_article_claude_tool,
    create_draft_post_tool,
]

tool_node = ToolNode(tools)


def build_graph(llm):
    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: ChatState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def route_tools(state: ChatState):
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        return END

    graph = StateGraph(ChatState)
    graph.add_node("chatbot", chatbot_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("chatbot")
    graph.add_conditional_edges("chatbot", route_tools)
    graph.add_edge("tools", "chatbot")

    return graph.compile()
