import re

import streamlit as st

from llm import LLMUnavailable
from orchestrator import Orchestrator

st.set_page_config(page_title="Telecom Package Assistant", page_icon="📱", layout="wide")

CSS = """
<style>
.hero{background:linear-gradient(120deg,#4f46e5 0%,#9333ea 55%,#ec4899 100%);
      padding:26px 32px;border-radius:20px;color:#fff;margin-bottom:18px;
      box-shadow:0 10px 30px rgba(147,51,234,.25)}
.hero h1{margin:0 0 6px 0;font-size:2rem;color:#fff}
.hero p{margin:0 0 12px 0;opacity:.92}
.chip{display:inline-block;padding:2px 12px;border-radius:999px;color:#fff;font-weight:600;
      font-size:.8rem;margin-right:6px}
.chip.jazz{background:#e11d2e}.chip.zong{background:#16a34a}.chip.ufone{background:#f26b21}

div[data-testid="stMarkdownContainer"] table{display:block;overflow-x:auto;width:100%;
      border-collapse:separate;border-spacing:0;border-radius:12px;font-size:.85rem;
      border:1px solid rgba(128,128,128,.25)}
div[data-testid="stMarkdownContainer"] th{background:linear-gradient(90deg,#4f46e5,#9333ea);
      color:#fff;padding:8px 10px;white-space:nowrap;text-align:left}
div[data-testid="stMarkdownContainer"] td{padding:7px 10px;border-top:1px solid rgba(128,128,128,.15)}
div[data-testid="stMarkdownContainer"] tr:nth-child(even) td{background:rgba(128,128,128,.07)}
div[data-testid="stMarkdownContainer"] tr:hover td{background:rgba(147,51,234,.13)}

div[data-testid="stChatMessage"]{border-radius:16px;border:1px solid rgba(128,128,128,.18);
      padding:12px 16px;margin-bottom:10px}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(79,70,229,.10),rgba(236,72,153,.08))}
section[data-testid="stSidebar"] .stButton>button{width:100%;text-align:left;border-radius:12px;
      border:1px solid rgba(147,51,234,.35);transition:all .15s}
section[data-testid="stSidebar"] .stButton>button:hover{border-color:#9333ea;transform:translateX(3px)}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><h1>📱 Telecom Package Assistant</h1>'
    "<p>Compare prepaid packages, like for like.</p>"
    '<span class="chip jazz">Jazz</span><span class="chip zong">Zong</span>'
    '<span class="chip ufone">Ufone</span></div>',
    unsafe_allow_html=True)

CHIP_RX = re.compile(r"(?<=\|)\s*(Jazz|Zong|Ufone)\s*(?=\|)")


def style_tables(md: str) -> str:
    """Colour operator names inside the markdown tables. If chips don't render on your
    Streamlit version, make this function `return md`."""
    return CHIP_RX.sub(lambda m: f' <span class="chip {m.group(1).lower()}">{m.group(1)}</span> ', md)


EXAMPLES = [
    "Compare weekly packages of Zong, Jazz and Ufone by on-net minutes",
    "Compare monthly packages of Zong with Jazz",
    "Compare My5 with its competitors in Ufone and Jazz",
    "Compare 3 days packages of Zong, Jazz and Ufone",
    "Cheapest 1 day package of Ufone and Zong",
]

# ---------------------------------------------------------------- session
if "orch" not in st.session_state:
    with st.spinner("Loading package data..."):
        st.session_state.orch = Orchestrator()
    st.session_state.msgs = []
orch: Orchestrator = st.session_state.orch

with st.sidebar:
    st.markdown("### 💡 Try asking")
    for ex in EXAMPLES:
        if st.button(ex, key=ex):
            st.session_state.pending = ex
    st.divider()
    if st.button("🗑️ New chat"):
        orch.reset()
        st.session_state.msgs = []
        st.rerun()
    st.caption("Prices as listed on the operators' sites. Check the Tax column when comparing.")

# ---------------------------------------------------------------- past messages
for m in st.session_state.msgs:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "📱"):
        if m.get("table"):
            st.markdown(style_tables(m["table"]), unsafe_allow_html=True)
        if m.get("text"):
            st.markdown(m["text"])

# ---------------------------------------------------------------- new question
q = st.chat_input("Ask about Jazz, Zong and Ufone packages...") or st.session_state.pop("pending", None)
if q:
    st.session_state.msgs.append({"role": "user", "text": q})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(q)

    with st.chat_message("assistant", avatar="📱"):
        status = st.status("Understanding your question...", expanded=False)
        table_ph, text_ph = st.empty(), st.empty()
        table_md, text, plan_json = "", "", ""
        try:
            for ev, payload in orch.run_stream(q):
                if ev == "status":
                    status.update(label=payload)
                elif ev == "plan":
                    plan_json = payload.model_dump_json(exclude_defaults=True, indent=2)
                elif ev == "table":
                    table_md = payload
                    table_ph.markdown(style_tables(table_md), unsafe_allow_html=True)
                elif ev == "token":
                    text += payload
                    text_ph.markdown(text + " ▌")
            text_ph.markdown(text)
            status.update(label="Done", state="complete")
        except LLMUnavailable as e:
            status.update(label="The language model is unavailable", state="error")
            st.error(str(e))
        if plan_json:
            with st.expander("How I understood your question"):
                st.code(plan_json, language="json")

    st.session_state.msgs.append({"role": "assistant", "table": table_md, "text": text})