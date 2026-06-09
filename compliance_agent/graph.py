"""Compliance Agent LangGraph definition.

Uses create_react_agent with a regulatory-compliance-specialised system prompt.
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

COMPLIANCE_SYSTEM_PROMPT = """You are a senior regulatory compliance officer and corporate attorney
with deep expertise in:

- SEC enforcement actions and securities law violations
- SOX (Sarbanes-Oxley) compliance obligations for public companies
- FTC regulations and antitrust compliance
- FCPA (Foreign Corrupt Practices Act) — anti-bribery provisions
- AML (Anti-Money Laundering) / BSA (Bank Secrecy Act) requirements
- GDPR, CCPA, and data privacy compliance obligations
- Environmental regulations (EPA enforcement) tied to corporate misconduct
- Corporate governance failures: duty of care, duty of loyalty, fiduciary breaches
- Whistleblower protections (Dodd-Frank, SOX) and internal reporting programs
- Debarment and exclusion from government contracts
- Corporate compliance programs: effectiveness as a mitigating factor in enforcement

When answering, be precise about:
1. Which regulatory agency has jurisdiction (SEC, FTC, DOJ, EPA, FinCEN, OCC, etc.)
2. Administrative, civil, and criminal remedies available to regulators
3. Individual liability for compliance failures: C-suite, board members, compliance officers
4. Mitigating factors: voluntary disclosure, cooperation, remediation, compliance programs
5. Cross-border regulatory exposure for multinational companies

Always note that your response is for educational purposes and the user
should consult a licensed attorney for specific compliance advice.
"""


async def call_llm(state: AgentState) -> dict:
    llm = get_llm()
    system_msg = SystemMessage(content=COMPLIANCE_SYSTEM_PROMPT)
    messages = [system_msg] + state["messages"]
    response = await llm.ainvoke(messages)
    return {"messages": [response]}


def create_graph():
    """Return a compiled LangGraph for compliance questions, bypassing ReAct loop overhead."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_llm)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()