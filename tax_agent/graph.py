"""Tax Agent LangGraph definition.

Uses create_react_agent with a tax-specialised system prompt.
No tools — it answers purely from LLM knowledge.
"""

from __future__ import annotations

from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage

from common.llm import get_llm

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

TAX_SYSTEM_PROMPT = """You are a specialist tax attorney and CPA with expertise in:

- Corporate tax law and compliance (federal, state, and international)
- Tax evasion vs. tax avoidance — legal distinctions and consequences
- IRS enforcement mechanisms, audits, and criminal referrals
- Penalties and back-tax calculations under IRC §§ 6651, 6662, 6663
- FBAR/FATCA requirements for offshore accounts
- Transfer pricing regulations (IRC § 482)
- Tax fraud statutes (18 U.S.C. § 7201 – § 7207)
- Corporate tax liability: officers, directors, and responsible persons
- Voluntary disclosure programs and settlement options

When answering, be precise about:
1. Civil vs. criminal penalties and their monetary ranges
2. Statute of limitations for tax fraud (6 years for substantial omission,
   unlimited for fraudulent returns)
3. Which government agencies are involved (IRS, DOJ Tax Division, FinCEN)
4. The distinction between the company's liability and individual liability
   for executives who directed the evasion

Always note that your response is for educational purposes and the user
should consult a licensed attorney for specific legal advice. Keep your response extremely brief, under 100 words.
"""


async def call_llm(state: AgentState) -> dict:
    llm = get_llm()
    system_msg = SystemMessage(content=TAX_SYSTEM_PROMPT)
    messages = [system_msg] + state["messages"]
    response = await llm.ainvoke(messages)
    return {"messages": [response]}


def create_graph():
    """Return a compiled LangGraph for tax questions, bypassing ReAct loop overhead."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_llm)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()