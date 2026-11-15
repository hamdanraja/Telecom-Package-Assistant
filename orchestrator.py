from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Optional, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

try:
    from langgraph.types import Send
except ImportError:                                  # older langgraph
    from langgraph.constants import Send

import tools
from agents import (AnswerAgent, ComparisonAgent, OperatorAgent, OperatorResult, PlannerAgent,
                    OPERATORS, Comparison, is_followup, resolve_references)
from data_loader import load_all
from schemas import Plan



# ===================================================================== graph state
class State(TypedDict, total=False):
    query: str
    last_plan: Optional[Plan]
    plan: Plan
    results: Annotated[list, operator.add]     # parallel operator nodes append here
    round: int                                 # 0 = normal run, 1 = relaxed retry
    relax: bool
    retry: bool
    comparison: Optional[Comparison]


def build_graph(df: pd.DataFrame):
    planner = PlannerAgent()
    comparer = ComparisonAgent()
    op_agents = {op: OperatorAgent(op, df) for op in OPERATORS}

    def plan_node(s: State):
        plan = planner.plan(s["query"], s.get("last_plan"))
        return {"plan": plan, "round": 0, "relax": False, "retry": False, "comparison": None}

    def operator_node(p: dict):
        res = op_agents[p["op"]].run(p["plan"], p["relax"])
        return {"results": [{"round": p["round"], "res": res}]}

    def check_node(s: State):
        plan = s["plan"]
        cur = [x["res"] for x in s["results"] if x["round"] == s["round"]]
        need_retry = (not s["relax"] and bool(cur) and all(r.full.empty for r in cur)
                      and plan.intent not in ("package_details", "compare_packages"))
        if need_retry:
            return {"round": s["round"] + 1, "relax": True, "retry": True}
        return {"retry": False}

    def compare_node(s: State):
        plan = s["plan"]
        order = {o: i for i, o in enumerate(plan.operators or OPERATORS)}
        cur = [x["res"] for x in s["results"] if x["round"] == s["round"]]
        cur.sort(key=lambda r: order.get(r.operator, 99))
        return {"comparison": comparer.run(cur, plan)}

    def similar_node(s: State):
        plan = s["plan"]
        refs, problems = resolve_references(df, plan.similar_to)
        problems = [f"[reference] {p}" for p in problems]
        if refs.empty:
            return {"comparison": Comparison(pd.DataFrame(), pd.DataFrame(), {}, problems or [
                f"[reference] Package(s) {plan.similar_to} were not found in any operator's data."])}

        refs = (refs.drop_duplicates(subset=["operator", "package_name", "price_pkr"])
                    .head(2).reset_index(drop=True))
        ref_ops = list(dict.fromkeys(refs["operator"]))
        comp_ops = ([o for o in (plan.operators or OPERATORS) if o not in ref_ops]
                    or [o for o in OPERATORS if o not in ref_ops])
        k = 3 if len(refs) == 1 else 2

        results: list[OperatorResult] = []
        for op in ref_ops:
            r = refs[refs["operator"] == op].assign(role="REFERENCE")
            results.append(OperatorResult(op, r, r, tools.operator_stats(r), []))
        for op in comp_ops:
            res = op_agents[op].similar(refs, k)
            if not res.top.empty:
                res.top = res.top.assign(role="similar to " + res.top["match_for"].astype(str))
            results.append(res)

        comp = comparer.run(results, plan)
        comp.notes = problems + comp.notes
        return {"comparison": comp}

    def _fan_out(plan: Plan, relax: bool, rnd: int):
        return [Send("operator", {"op": o, "plan": plan, "relax": relax, "round": rnd})
                for o in (plan.operators or OPERATORS)]

    def route_after_plan(s: State):
        plan = s["plan"]
        if plan.intent == "chitchat":
            return END
        if plan.similar_to:
            return "similar"
        return _fan_out(plan, relax=False, rnd=0)

    def route_after_check(s: State):
        if s.get("retry"):
            return _fan_out(s["plan"], relax=True, rnd=s["round"])
        return "compare"

    g = StateGraph(State)
    g.add_node("plan", plan_node)
    g.add_node("operator", operator_node)
    g.add_node("check", check_node)
    g.add_node("compare", compare_node)
    g.add_node("similar", similar_node)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", route_after_plan)
    g.add_edge("operator", "check")
    g.add_conditional_edges("check", route_after_check)
    g.add_edge("compare", END)
    g.add_edge("similar", END)
    return g.compile()


def _status_label(node: str, upd: dict) -> str | None:
    if node == "plan":
        return "Question understood - searching packages..."
    if node == "operator":
        try:
            res = upd["results"][0]["res"]
            return f"Checked {res.operator.title()}: {len(res.full)} matching package(s)"
        except Exception:
            return "Checking operators..."
    if node == "check" and upd.get("retry"):
        return "Nothing matched - looking for the closest alternatives..."
    if node in ("compare", "similar"):
        return "Comparing packages..."
    return None


# ===================================================================== public API
@dataclass
class Trace:
    plan: Plan
    comparison: Comparison | None
    answer_stream: object


class Orchestrator:
    def __init__(self):
        self.df = load_all()
        self.graph = build_graph(self.df)
        self.answerer = AnswerAgent()
        self.history: list[tuple[str, str]] = []
        self.last_plan: Plan | None = None

    def reset(self):
        self.history, self.last_plan = [], None

    # ---- old API (kept so nothing else breaks): blocks until the tables are ready
    def run(self, query: str) -> Trace:
        followup = bool(self.last_plan) and is_followup(query)
        out = self.graph.invoke({"query": query, "last_plan": self.last_plan, "results": []})
        plan: Plan = out["plan"]
        comp: Comparison | None = out.get("comparison")
        if plan.intent != "chitchat":
            self.last_plan = plan
        hist = self.history if followup else []
        stream = self._remember(query, self.answerer.stream(query, plan, comp, hist))
        return Trace(plan, comp, stream)

    # ---- new API: events appear as soon as they exist
    def run_stream(self, query: str):
        """Yields (event, payload):
             'status' -> str   progress line
             'plan'   -> Plan  how the question was understood
             'table'  -> str   markdown tables (ready BEFORE the LLM starts writing)
             'token'  -> str   answer text, streamed
        One graph pass only."""
        followup = bool(self.last_plan) and is_followup(query)
        state = {"query": query, "last_plan": self.last_plan, "results": []}
        final = dict(state)

        yield "status", "Understanding your question..."
        for step in self.graph.stream(state, stream_mode="updates"):
            for node, upd in step.items():
                if not isinstance(upd, dict):
                    continue
                for k, v in upd.items():
                    final[k] = (final.get(k, []) + v) if k == "results" else v
                label = _status_label(node, upd)
                if label:
                    yield "status", label

        plan: Plan = final["plan"]
        comp: Comparison | None = final.get("comparison")
        if plan.intent != "chitchat":
            self.last_plan = plan
        yield "plan", plan

        if plan.intent != "chitchat":
            table = self.answerer.render_tables(plan, comp)
            if table:
                yield "table", table

        yield "status", "Writing the summary..."
        hist = self.history if followup else []
        buf: list[str] = []
        for tok in self.answerer.stream_text(query, plan, comp, hist):
            buf.append(tok)
            yield "token", tok
        self.history.append((query, "".join(buf)))

    def _remember(self, query, gen):
        buf = []
        for tok in gen:
            buf.append(tok)
            yield tok
        self.history.append((query, "".join(buf)))