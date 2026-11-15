from __future__ import annotations
import difflib
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import tools
from config import THINK_FINAL, DATA_BASIS, NEAR_TOLERANCE
from formatting import to_markdown
from llm import chat_json, chat_stream
from schemas import Plan

OPERATORS = ["jazz", "zong", "ufone"]
OP_PATTERNS = {
    "jazz": r"\b(jazz|mobilink|warid)\b",
    "zong": r"\b(zong|cmpak|china mobile)\b",
    "ufone": r"\b(ufone|u-fone|ptcl)\b",
}

# ===================================================================== rule-based extraction helpers
CAT_PATTERNS = {
    "hourly":    r"\bhourly\b|\b(?:1|2|3|4|6|8|12)[\s-]?hours?\b",
    "daily":     r"\bdaily\b|\bone[\s-]?day\b|\b1[\s-]?days?\b|\b24[\s-]?hours?\b",
    "weekly":    r"\bweekly\b|\b7[\s-]?days?\b|\bone[\s-]?week\b|\b1[\s-]?week\b",
    "fortnightly": r"\bfortnight(?:ly)?\b|\b15[\s-]?days?\b",
    "monthly":   r"\bmonthly\b|\bone[\s-]?month\b|\b1[\s-]?month\b|\b30[\s-]?days?\b|\bmahana\b",
    "long_term": r"\byearly\b|\bannual\b|\bquarterly\b|\b[3-9][\s-]?months?\b|\b(?:90|180|365)[\s-]?days?\b",
}
UNITS = {"gb", "gbs", "mb", "tb", "min", "mins", "minute", "minutes", "sms", "day", "days",
         "hour", "hours", "week", "weeks", "month", "months", "package", "packages", "bundle",
         "bundles", "plan", "plans", "offer", "offers", "operator", "operators", "result", "results"}

CAP_WORDS = r"(?:under|below|less than|upto|up to|max|maximum|within)\s*$"
FLOOR_WORDS = r"(?:at\s*least|atleast|minimum|min|more than|over|above|greater than)\s*$"
FLOOR_AFTER = r"^\s*(?:\+|or more|and above|or above|and more|plus|or higher|or greater)"
NEAR_WORDS = (r"\b(?:near(?:ly|by|est)?|around|about|approx(?:imately)?|close(?:st)?|similar|roughly|"
              r"more or less|almost|aas paas|qareeb)\b")
SIMILAR_RX = r"competitors?|rivals?|similar|alternatives?|equivalents?|comparable|counterparts?|equal to"
ADD_WORDS = r"\b(?:also|too|as well|bhi|add)\b"

GENERIC_KW = {"package", "packages", "bundle", "bundles", "data", "internet", "gb", "mb", "offer",
              "offers", "plan", "plans", "monthly", "weekly", "daily", "hourly", "jazz", "zong",
              "ufone", "cheap", "cheapest", "best", "good", "provide", "suggest", "compare",
              "near", "around", "about", "approx", "similar", "competitors", "competitor", "alternatives",
              "close", "offering", "providing", "resources", "price", "prices", "more", "less", "highest",
              "lowest", "better", "which", "resource", "minutes", "sms", "calls", "onnet", "on-net"}
NAME_NOISE = re.compile(r"\b(jazz|zong|ufone|mobilink|package|packages|bundle|bundles|offer|offers|plan|plans|"
                        r"competitors?|rivals?|similar|alternatives?|equivalents?|the|of|its|with|and|or|to|vs)\b", re.I)
FOLLOWUP = re.compile(r"^\s*(and|also|what about|how about|what if|same|now|then|ok|okay|but|only|"
                      r"aur|or|more|show more|those|them)\b", re.I)

TIME_RE = re.compile(r"\b(night|midnight|overnight|morning|early|late)\b", re.I)
TIME_WORDS = {"night", "midnight", "overnight", "morning", "early", "late"}


def is_followup(q: str) -> bool:
    if FOLLOWUP.search(q):
        return True
    ql = q.lower()
    words = re.findall(r"\w+", ql)
    return (len(words) <= 3 and
            not re.search(r"package|bundle|plan|offer|monthly|weekly|daily|hourly|\d|jazz|zong|ufone", ql))


def representative_mode(plan: Plan) -> bool:
    """Broad operator comparison with no resource named -> like-for-like groups across operators."""
    return plan.intent == "compare_operators" and plan.sort_by is None and not plan.target_data_gb


def detect_operators(q: str) -> list[str]:
    found = [op for op, pat in OP_PATTERNS.items() if re.search(pat, q)]
    for w in re.findall(r"[a-z]{4,}", q):                      # typos: ufine, jazs, zongg ...
        m = difflib.get_close_matches(w, OPERATORS, 1, 0.8)
        if m and m[0] not in found:
            found.append(m[0])
    return found


def _data_rules(q: str):
    """returns (min_gb, max_gb, target_gb, found)."""
    T = NEAR_TOLERANCE
    items = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(tb|gb|mb)\b", q):
        gb = float(m.group(1)) * {"tb": 1000, "gb": 1, "mb": 1 / 1000}[m.group(2)]
        items.append((gb, q[max(0, m.start() - 25):m.start()], q[m.end():m.end() + 14]))
    if not items:
        return None, None, None, False
    vals = sorted({round(i[0], 4) for i in items})
    near = bool(re.search(NEAR_WORDS, q))
    if len(vals) >= 2:
        lo, hi = vals[0], vals[-1]
        if near: lo, hi = lo * (1 - T / 2), hi * (1 + T / 2)
        return lo, hi, (vals[0] + vals[-1]) / 2, True
    gb, before, after = items[0]
    if near:
        return gb * (1 - T), gb * (1 + T), gb, True
    if re.search(CAP_WORDS, before):
        return None, gb, None, True
    if re.search(FLOOR_WORDS, before) or re.search(FLOOR_AFTER, after):
        return gb, None, None, True
    return gb * (1 - T), gb * (1 + T), gb, True


def _price_rules(q: str):
    mx = mn = None
    for m in re.finditer(r"(?:under|below|less than|within|upto|up to|maximum|max|budget of|budget)"
                         r"\s*(?:rs\.?|pkr)?\s*(\d[\d,]*(?:\.\d+)?)\s*([a-z]*)", q):
        if m.group(2) not in UNITS: mx = float(m.group(1).replace(",", ""))
    for m in re.finditer(r"(?:above|over|more than|at least|minimum|min)"
                         r"\s*(?:rs\.?|pkr)?\s*(\d[\d,]*(?:\.\d+)?)\s*([a-z]*)", q):
        if m.group(2) not in UNITS: mn = float(m.group(1).replace(",", ""))
    return mn, mx


def _type_rule(q: str):
    if re.search(r"\b(?:internet|data)[\s-]?only\b|\bpure (?:internet|data)\b", q): return "internet"
    if re.search(r"\b(?:call|calling|voice|minutes?)\s+(?:only\s+)?(?:packages?|bundles?|offers?)\b", q): return "call"
    if re.search(r"\bsms\s+(?:only\s+)?(?:packages?|bundles?|offers?)\b", q): return "sms"
    return None


# ===================================================================== STRICT VALIDITY
VALIDITY_BANDS = {"hourly": (0.0, 0.999), "daily": (0.999, 1.5), "weekly": (6.5, 7.5),
                  "fortnightly": (14.0, 15.5), "monthly": (28.0, 31.5), "long_term": (32.0, 10_000.0)}


def validity_ranges(plan: Plan) -> list[tuple[float, float]]:
    if plan.validity_days:                                     # an exact number beats a category
        d = float(plan.validity_days)
        t = max(0.5, 0.06 * d)
        return [(d - t, d + t)]
    return [VALIDITY_BANDS[c] for c in plan.categories if c in VALIDITY_BANDS]


def describe_validity(plan: Plan) -> str:
    if plan.validity_days:
        return f"{plan.validity_days:g} days"
    return "/".join(plan.categories) or "any"


def strict_validity(df: pd.DataFrame, plan: Plan) -> pd.DataFrame:
    rng = validity_ranges(plan)
    if not rng or df.empty:
        return df
    v = pd.to_numeric(df["validity_days"], errors="coerce")
    mask = pd.Series(False, index=df.index)
    for lo, hi in rng:
        mask |= v.between(lo, hi)
    return df[mask]


def vbucket(d) -> int:
    if d is None or pd.isna(d): return -1
    d = float(d)
    if d < 1:    return 0
    if d <= 7:   return int(round(d))
    if d <= 20:  return 15
    if d <= 35:  return 30
    if d <= 120: return 90
    if d <= 250: return 180
    return 365


# ===================================================================== DERIVED COLUMNS + LIKE-FOR-LIKE MATH
_WARNED = {"onnet": False}


def _find_col(df: pd.DataFrame, include: str, exclude: str | None = None):
    for c in df.columns:
        lc = str(c).lower()
        if re.search(include, lc) and not (exclude and re.search(exclude, lc)):
            return c
    return None


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Adds onnet_eff, price_per_100_onnet, vbucket, time_restricted, profile. Safe to call twice."""
    if "profile" in df.columns:
        return df
    d = df.copy()

    def num(col):
        return pd.to_numeric(d[col], errors="coerce") if col in d.columns else pd.Series(np.nan, index=d.index)

    for c in ("price_pkr", "validity_days", "data_gb", "internet_gb", "minutes_eff", "total_minutes", "sms"):
        if c in d.columns:
            d[c] = num(c)

    on = _find_col(d, r"on[\s_-]?net", r"off")
    al = _find_col(d, r"all[\s_-]?net|any[\s_-]?net")
    if on:
        d["onnet_eff"] = num(on).fillna(0) + (num(al).fillna(0) if al else 0)   # all-net can be used on-net
    else:
        if not _WARNED["onnet"]:
            print("WARNING: no on-net minutes column found; falling back to minutes_eff. "
                  "Rename the regex in _find_col or your column.")
            _WARNED["onnet"] = True
        d["onnet_eff"] = num("minutes_eff").fillna(0)

    ok = np.isfinite(d["onnet_eff"]) & (d["onnet_eff"] > 0)
    d["price_per_100_onnet"] = (num("price_pkr") / d["onnet_eff"] * 100).where(ok).round(2)
    d["vbucket"] = d["validity_days"].map(vbucket) if "validity_days" in d else -1
    d["time_restricted"] = d["package_name"].astype(str).str.contains(TIME_RE)

    mins = num("minutes_eff").fillna(0)
    data = num("data_gb").fillna(0)
    net = num("internet_gb").fillna(0)
    unl = d["internet_unlimited"].fillna(False).astype(bool) if "internet_unlimited" in d.columns \
        else pd.Series(False, index=d.index)
    has_data = (data > 0) | unl
    voice = mins >= 50
    d["profile"] = np.select(
        [has_data & voice, voice, has_data & (net == 0) & ~unl, has_data, num("sms").fillna(0) > 0],
        ["combo", "voice", "social", "data", "sms"], default="other")
    return d


def _v(x) -> float:
    x = 0.0 if x is None or pd.isna(x) else float(x)
    return min(x, 1e6)


def distance(a, b, mcol: str = "minutes_eff") -> float:
    """0 = identical resources (data, minutes, sms). Compared in log-ratio."""
    tot = w = 0.0
    for col, wt, eps in (("data_gb", 3.0, 0.1), (mcol, 2.0, 10.0), ("sms", 0.5, 50.0)):
        x, y = _v(a[col]), _v(b[col])
        if x == 0 and y == 0:
            continue
        tot += wt * abs(math.log((x + eps) / (y + eps)))
        w += wt
    return tot / w if w else 9.0


def sim_label(d: float) -> str:
    return "near-identical" if d <= 0.25 else "close" if d <= 0.5 else "partial"


def like_for_like(pool: pd.DataFrame, max_groups: int = 6, mcol: str = "minutes_eff",
                  thresh: float = 0.5) -> pd.DataFrame:
    """Groups of packages from DIFFERENT operators with the same validity bucket, same profile and close resources.
    One package per operator per group. Only groups with >= 2 operators are returned."""
    if pool.empty or pool["operator"].nunique() < 2:
        return pool.iloc[0:0]
    pool = pool.reset_index(drop=True)
    ops = list(pool["operator"].unique())
    used: set = set()
    groups = []
    for i in pool.sort_values("price_pkr").index:
        if i in used:
            continue
        seed = pool.loc[i]
        members = [(i, 0.0)]
        for op in ops:
            if op == seed["operator"]:
                continue
            cand = pool[(pool["operator"] == op) & (pool["profile"] == seed["profile"])
                        & (pool["vbucket"] == seed["vbucket"]) & (~pool.index.isin(used))]
            if cand.empty:
                continue
            dd = cand.apply(lambda r: distance(seed, r, mcol), axis=1)
            best = pd.DataFrame({"d": dd, "p": cand["price_pkr"]}).sort_values(["d", "p"]).iloc[0]
            if best["d"] <= thresh:
                members.append((best.name, float(best["d"])))
        if len(members) >= 2:
            used.update(m[0] for m in members)
            groups.append(members)
    if not groups:
        return pool.iloc[0:0]
    if len(groups) > max_groups:                               # spread across price tiers
        sel = np.unique(np.linspace(0, len(groups) - 1, max_groups).round().astype(int))
        groups = [groups[k] for k in sel]
    rows = []
    for n, members in enumerate(groups, 1):
        lab = f"G{n} {sim_label(max(d for _, d in members))}"
        for idx, _ in members:
            r = pool.loc[idx].copy()
            r["group"] = lab
            rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)


RESOURCE_SORT = {"onnet_minutes": ("onnet_eff", False),
                 "price_per_100_onnet": ("price_per_100_onnet", True),
                 "sms": ("sms", False),
                 "total_minutes": ("total_minutes", False)}


def sort_any(df: pd.DataFrame, plan: Plan) -> pd.DataFrame:
    spec = RESOURCE_SORT.get(plan.sort_by)
    if not spec or df.empty or spec[0] not in df.columns:
        return tools.sort_df(df, plan)
    col, asc = spec
    d = df.assign(_k=df[col].replace(float("inf"), 1e9))
    d = d[d["_k"].fillna(0) > 0]                               # resource must actually exist
    return d.sort_values(["_k", "price_pkr"], ascending=[asc, True]).drop(columns="_k")


def _tok(s) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split()


def resolve_references(df: pd.DataFrame, names: list[str]):
    """Find the named reference package(s) across ALL operators.
    Returns (refs_df, problems). Never picks a different package silently."""
    d = prepare(df)
    d = d.drop_duplicates(subset=["operator", "package_name", "price_pkr"])
    keys = d["package_name"].map(lambda p: "".join(_tok(p)))
    refs, problems = [], []
    for name in names:
        toks = _tok(name)
        exact = d[keys == "".join(toks)]
        hits = exact if len(exact) else d[d["package_name"].map(lambda p: all(t in set(_tok(p)) for t in toks))]
        if hits.empty:
            sugg = difflib.get_close_matches("".join(toks), keys.tolist(), 5, 0.6)
            names_s = d[keys.isin(sugg)]["package_name"].tolist()
            problems.append(f"Package '{name}' was NOT found. Closest names: {names_s or 'none'}. "
                            f"Do not substitute another package.")
            continue
        extra = hits["package_name"].map(lambda p: len(_tok(p)) - len(toks))
        best = hits[extra == extra.min()]
        if len(best) > 1:
            opts = [f"{r.operator.title()} {r.package_name}" for r in best.head(6).itertuples()]
            problems.append(f"'{name}' is ambiguous: {opts}. Ask the user which one.")
            continue
        refs.append(best.assign(role="REFERENCE", match_for=best["package_name"], match_score=100.0,
                                similarity="reference"))
    return (pd.concat(refs) if refs else d.iloc[0:0]), problems


# ===================================================================== 1. PLANNER (LLM + rules)
PLANNER_SYSTEM = """You convert questions about Pakistani mobile packages (Jazz, Zong, Ufone) into a JSON plan.
Fill ONLY what the user asked for; leave everything else null/empty.

intent:
- package_details: about ONE named package (price, validity, what's included...)
- compare_packages: compares specific NAMED packages, or a named package with its competitors/similar packages
- compare_operators: compares operators in general or for a category/need
- recommend: asks for best / cheapest / most suitable package
- search: list packages matching conditions
- statistics: average / count / min / max across packages
- chitchat: greeting or unrelated

Fields: operators (jazz|zong|ufone; empty = all), package_names (ONLY real package names such as "Super Duper",
"My5", "Weekly Pro Plus" - never operator names or words like monthly/weekly/packages), categories (list of:
hourly|daily|weekly|fortnightly|monthly|long_term), min_price, max_price (PKR), min_data_gb, max_data_gb,
min_minutes, min_sms, validity_days, keywords (feature words only: whatsapp, youtube, night, 4g...),
sort_by (price|data_gb|price_per_gb|price_per_day|validity_days|total_minutes|sms),
ascending (true for cheapest/lowest; false for most/highest/biggest), limit.

Rules:
- "monthly/weekly/daily" go to categories, NEVER keywords. "daily or weekly" -> categories ["daily","weekly"].
- Do not set package_type unless the user says "internet only" / "call package" / "sms package".
- Set min_minutes / min_sms only if the user gives a number of minutes / SMS.
- Use PREVIOUS PLAN only when the question is clearly a short follow-up ("what about zong?").

Examples:
Q: Compare Jazz Super Duper and Zong Super Plus
-> {"intent":"compare_packages","operators":["jazz","zong"],"package_names":["Super Duper","Super Plus"]}
Q: Compare My5 with its competitors in ufone and jazz
-> {"intent":"compare_packages","operators":["ufone","jazz"],"package_names":["My5"]}
Q: Compare monthly packages of all operators near 200gb
-> {"intent":"compare_operators","categories":["monthly"],"min_data_gb":150,"max_data_gb":250}
Q: Show daily or weekly packages of jazz
-> {"intent":"search","operators":["jazz"],"categories":["daily","weekly"]}
Q: Average price of monthly packages for each operator
-> {"intent":"statistics","categories":["monthly"]}
"""


class PlannerAgent:
    def plan(self, query: str, prev: Plan | None) -> Plan:
        followup = bool(prev) and is_followup(query)
        prev_txt = prev.model_dump_json(exclude_defaults=True) if followup else "none"
        raw = chat_json([{"role": "system", "content": PLANNER_SYSTEM},
                         {"role": "user", "content": f"PREVIOUS PLAN: {prev_txt}\nQUESTION: {query}"}],
                        Plan.model_json_schema())
        try:
            plan = Plan.model_validate(raw)
        except Exception:
            plan = Plan(intent="search")
        return self._fix(plan, query, followup, prev)

    @staticmethod
    def _fix(plan: Plan, query: str, followup: bool, prev: Plan | None = None) -> Plan:
        """Deterministic rules override the LLM for everything explicit in the text.
        Follow-ups copy missing fields from the previous plan IN CODE (not via the LLM)."""
        q = query.lower()
        carry = followup and prev is not None

        # ---- operators (typo tolerant; "also zong" ADDS to previous operators)
        mentioned = detect_operators(q)
        if mentioned:
            if carry and re.search(ADD_WORDS, q):
                plan.operators = list(dict.fromkeys(list(prev.operators) + mentioned))
            else:
                plan.operators = mentioned
        elif carry:
            plan.operators = list(prev.operators)
        else:
            plan.operators = []

        # ---- validity: category and/or exact days
        cats = [c for c, pat in CAT_PATTERNS.items() if re.search(pat, q)]
        vd = None
        m = re.search(r"(\d+(?:\.\d+)?)\s*[- ]?(?:days?|din)\b", q)
        mm = re.search(r"\b(\d+)\s*[- ]?months?\b", q)
        if m:
            vd = float(m.group(1))
        elif mm and int(mm.group(1)) > 1:
            n = int(mm.group(1))
            vd = 365.0 if n == 12 else 30.0 * n
        elif re.search(r"\b(?:yearly|annual|1 year|one year)\b", q):
            vd = 365.0
        elif re.search(r"\bquarterly\b", q):
            vd = 90.0
        if cats or vd:
            plan.categories, plan.validity_days = cats, vd
        elif carry:
            plan.categories, plan.validity_days = list(prev.categories), prev.validity_days
        else:
            plan.categories, plan.validity_days = [], None

        # ---- data: band / min / max
        mn, mx, tgt, found = _data_rules(q)
        if found:
            plan.min_data_gb, plan.max_data_gb, plan.target_data_gb = mn, mx, tgt
        elif carry:
            plan.min_data_gb, plan.max_data_gb = prev.min_data_gb, prev.max_data_gb
            plan.target_data_gb = prev.target_data_gb
        else:
            plan.min_data_gb = plan.max_data_gb = plan.target_data_gb = None

        # ---- prices
        pmin, pmax = _price_rules(q)
        if pmin is not None or pmax is not None:
            plan.min_price, plan.max_price = pmin, pmax
        elif carry:
            plan.min_price, plan.max_price = prev.min_price, prev.max_price
        else:
            plan.min_price = plan.max_price = None

        plan.package_type = _type_rule(q)

        # ---- package names
        names = []
        for n in plan.package_names:
            cleaned = NAME_NOISE.sub(" ", n).strip()
            if cleaned and cleaned.lower() not in {"monthly", "weekly", "daily", "hourly"} \
                    and not re.fullmatch(r"\s*\d+(?:\.\d+)?\s*(?:gb|mb)\s*", n.lower()):
                names.append(n)
        if not names and carry and prev.package_names and not (cats or vd):
            names = list(prev.package_names)
            if prev.intent in ("package_details", "compare_packages"):
                plan.intent = prev.intent
        plan.package_names = names

        # ---- sorting / resource to rank by
        rx = None
        if re.search(r"\bon[\s-]?net\b|\bonnet\b", q):
            rx = ("price_per_100_onnet", True) if re.search(r"value|per minute|per 100|cost per", q) \
                else ("onnet_minutes", False)
        elif re.search(r"cheapest|lowest (?:price|cost)|least expensive|\bcheap\b|\bsast[ai]\b", q):
            rx = ("price", True)
        elif re.search(r"highest (?:price|cost)|most expensive|costliest|expensive", q):
            rx = ("price", False)
        elif re.search(r"most data|highest data|maximum data|biggest data|largest data", q):
            rx = ("data_gb", False)
        elif re.search(r"best value|value for money|price per gb|per gb", q):
            rx = ("price_per_gb", True)
        elif re.search(r"most minutes|most calls?|highest minutes|depending on (?:minutes|calls)|by minutes", q):
            rx = ("total_minutes", False)
        elif re.search(r"most sms|highest sms|depending on sms|by sms|more sms", q):
            rx = ("sms", False)
        elif re.search(r"longest validity|longest", q):
            rx = ("validity_days", False)
        if rx:
            plan.sort_by, plan.ascending = rx
        elif carry and not plan.target_data_gb:
            plan.sort_by, plan.ascending = prev.sort_by, prev.ascending
        else:
            plan.sort_by = plan.ascending = None

        # ---- LLM-invented filters: keep only what the text mentions
        if not followup:
            if not re.search(r"\b(?:minutes?|mins?|calls?)\b", q): plan.min_minutes = None
            if not re.search(r"\b(?:sms|messages?)\b", q):          plan.min_sms = None
        plan.keywords = [k for k in plan.keywords
                         if k.lower() in q and k.lower() not in GENERIC_KW
                         and not re.fullmatch(r"\d+(gb|mb)?", k.lower())]

        # ---- similar / competitors mode
        plan.similar_to = []
        if plan.package_names and (re.search(SIMILAR_RX, q) or (carry and prev.similar_to)):
            plan.similar_to = plan.package_names[:2]
            plan.intent = "compare_packages"

        # ---- intent sanity
        if plan.intent in ("package_details", "compare_packages") and not plan.package_names:
            plan.intent = "search"
        if (re.search(r"\bcompare\b|\bvs\b|versus|difference|which operator|\bbetter\b", q)
                and len(plan.operators) != 1 and not plan.package_names):
            plan.intent = "compare_operators"
        if plan.intent in ("compare_operators", "statistics") and not plan.operators:
            plan.operators = list(OPERATORS)

        # ---- rows per operator
        m = re.search(r"\btop\s*(\d{1,2})\b|\b(\d{1,2})\s+(?:packages|bundles|plans|offers)\b", q)
        if m:
            plan.limit = int(m.group(1) or m.group(2))
        elif re.search(r"\ball (?:the )?(?:packages|bundles|plans|offers)\b|\bevery\b|\blist all\b", q):
            plan.limit = 15
        else:
            plan.limit = 8 if (plan.intent in ("compare_operators", "compare_packages")
                               or plan.target_data_gb) else 5
        plan.limit = max(1, min(plan.limit, 15))
        return plan


# ===================================================================== 2. OPERATOR AGENTS
@dataclass
class OperatorResult:
    operator: str
    top: pd.DataFrame
    full: pd.DataFrame
    stats: dict
    notes: list[str] = field(default_factory=list)


GENERAL_INTENTS = ("compare_operators", "recommend", "search")


class OperatorAgent:
    """Owns ONE operator's data and answers the plan against it."""

    def __init__(self, operator: str, df: pd.DataFrame):
        self.operator = operator
        self.df = prepare(df[df["operator"] == operator].reset_index(drop=True))

    def run(self, plan: Plan, relax: bool = False) -> OperatorResult:
        notes: list[str] = []
        if self.df.empty:
            return OperatorResult(self.operator, self.df, self.df, {"matches": 0}, ["no data loaded"])

        # a) named packages
        if plan.package_names and plan.intent in ("package_details", "compare_packages"):
            parts = [tools.find_by_name(self.df, n) for n in plan.package_names]
            full = pd.concat(parts) if parts else self.df.iloc[0:0]
            if not full.empty:
                full = full.drop_duplicates(subset=["package_name", "price_pkr"])
            else:
                notes.append(f"No package matching {plan.package_names} found for this operator.")
            return OperatorResult(self.operator, full, full, tools.operator_stats(full), notes)

        # b) HARD validity filter first; it can never be relaxed
        base = strict_validity(self.df, plan)
        if base.empty:
            avail = sorted({int(round(x)) for x in self.df["validity_days"].dropna()})
            notes.append(f"NO PACKAGE with validity {describe_validity(plan)} for this operator. "
                         f"Validities that exist (days): {avail}. Do NOT present other durations as matches.")
            return OperatorResult(self.operator, base, base, {"matches": 0}, notes)

        plan_nv = plan.model_copy(update={"categories": [], "validity_days": None})
        full = tools.apply_filters(base, plan_nv)
        used = plan_nv
        if full.empty:
            notes.append("NO EXACT MATCH. Filter funnel: " + tools.funnel(base, plan_nv))
            if relax:
                full, used, dropped = tools.relax(base, plan_nv)
                if not full.empty:
                    notes.append("Rows shown are the CLOSEST ALTERNATIVES, NOT exact matches. "
                                 "Adjustments: " + ", ".join(dropped))
        full = sort_any(full, used)

        if plan.intent in GENERAL_INTENTS and not full.empty:
            # hide packs with no general internet and no minutes (WhatsApp-only, SMS-only...)
            if not (plan.keywords or plan.package_type or plan.min_sms):
                no_core = (full["internet_gb"].isna() & ~full["internet_unlimited"]
                           & (full["minutes_eff"].fillna(0) == 0))
                if (~no_core).any():
                    full = full[~no_core]
            # hide night / morning-only packs unless asked for
            if not ({k.lower() for k in plan.keywords} & TIME_WORDS):
                n_hidden = int(full["time_restricted"].sum())
                if n_hidden and (~full["time_restricted"]).any():
                    full = full[~full["time_restricted"]]
                    notes.append(f"{n_hidden} night/morning-only package(s) hidden (not all-day).")

        return OperatorResult(self.operator, full.head(plan.limit), full, tools.operator_stats(full), notes)

    def similar(self, refs: pd.DataFrame, k: int = 3, max_dist: float = 0.9) -> OperatorResult:
        """Like-for-like packages of THIS operator for each reference (same validity bucket, same profile)."""
        refs = prepare(refs)
        parts, notes = [], []
        for i in refs.index:
            ref_df, ref = refs.loc[[i]], refs.loc[i]
            if ref["operator"] == self.operator:               # reference's own operator: show the reference
                parts.append(ref_df.assign(role="REFERENCE", match_for=ref["package_name"],
                                           match_score=100.0, similarity="reference"))
                continue
            c = self.df[(self.df["profile"] == ref["profile"]) & (self.df["vbucket"] == ref["vbucket"])
                        & (self.df["time_restricted"] == ref["time_restricted"])]
            if not c.empty:
                c = c.assign(_d=c.apply(lambda r: distance(ref, r), axis=1))
                c = c[c["_d"] <= max_dist].sort_values(["_d", "price_pkr"]).head(k)
            if c.empty:
                notes.append(f"No like-for-like package (same validity, same type, close resources) "
                             f"for {ref['package_name']}.")
                continue
            parts.append(c.assign(role="SIMILAR", match_for=ref["package_name"],
                                  match_score=(100 * (1 - c["_d"].clip(upper=1))).round(1),
                                  similarity=c["_d"].map(sim_label)).drop(columns="_d"))
        top = pd.concat(parts).drop_duplicates(subset=["package_name", "price_pkr", "match_for"]) \
            if parts else self.df.iloc[0:0]
        return OperatorResult(self.operator, top, top, tools.operator_stats(top), notes)


# ===================================================================== 3. COMPARISON AGENT
@dataclass
class Comparison:
    table: pd.DataFrame
    summary: pd.DataFrame
    winners: dict[str, str]
    notes: list[str]
    facts: list[str] = field(default_factory=list)


def _delta(a, b, unit="", nd=0):
    try:
        if pd.isna(a) or pd.isna(b) or a in (float("inf"),) or b in (float("inf"),): return "n/a"
    except Exception:
        return "n/a"
    d = a - b
    return f"{d:+,.{nd}f}{unit}"


def _n(x) -> str:
    if x is None or pd.isna(x): return "-"
    if x == float("inf"): return "Unlimited"
    return f"{x:,.1f}".rstrip("0").rstrip(".")


class ComparisonAgent:
    def run(self, results: list[OperatorResult], plan: Plan) -> Comparison:
        tops = [r.top for r in results if not r.top.empty]
        fulls = [r.full for r in results if not r.full.empty]
        table = pd.concat(tops, ignore_index=True) if tops else pd.DataFrame()
        pool = pd.concat(fulls, ignore_index=True) if fulls else pd.DataFrame()
        notes = [f"[{r.operator}] {n}" for r in results for n in r.notes]
        facts: list[str] = []

        if (plan.intent in ("compare_packages", "package_details") and not plan.similar_to
                and not table.empty and "match_for" in table):
            keep = []
            for _, g in table.groupby("match_for"):
                keep.append(g[g["match_score"] >= g["match_score"].max() - 8])
            table = pd.concat(keep).sort_values("price_pkr").reset_index(drop=True)

        # generic operator comparison -> like-for-like groups only
        lfl = False
        if representative_mode(plan) and not pool.empty:
            lfl = True
            mcol = "onnet_eff" if plan.sort_by in ("onnet_minutes", "price_per_100_onnet") else "minutes_eff"
            table = like_for_like(pool, max_groups=min(plan.limit, 6), mcol=mcol)
            if table.empty:
                notes.append("NO LIKE-FOR-LIKE packages across these operators (same validity, same type, "
                             "close resources). Say so plainly; do not compare unrelated packages.")
            else:
                facts = self._group_facts(table)

        summary = pd.DataFrame([{"operator": r.operator.title(), **r.stats} for r in results])

        winners: dict[str, str] = {}
        if not pool.empty:
            def pick(col, fn, label, fmt):
                if col not in pool.columns: return
                s = pool.replace([float("inf")], float("nan")).dropna(subset=[col])
                s = s[s[col] > 0] if col in ("onnet_eff", "total_minutes", "data_gb", "sms") else s
                if s.empty: return
                r = s.loc[getattr(s[col], fn)()]
                winners[label] = f"{r['operator'].title()} - {r['package_name']} ({fmt(r)})"
            pick("price_pkr", "idxmin", "Cheapest price", lambda r: f"Rs {r['price_pkr']:,.0f}")
            pick("price_per_gb", "idxmin", "Lowest price per GB", lambda r: f"Rs {r['price_per_gb']:,.1f}/GB")
            pick("data_gb", "idxmax", "Most data", lambda r: f"{r['data_gb']:,.1f} GB, Rs {r['price_pkr']:,.0f}")
            pick("price_per_day", "idxmin", "Lowest price per day", lambda r: f"Rs {r['price_per_day']:,.1f}/day")
            pick("onnet_eff", "idxmax", "Most on-net minutes",
                 lambda r: f"{r['onnet_eff']:,.0f} on-net min, Rs {r['price_pkr']:,.0f}")
            pick("price_per_100_onnet", "idxmin", "Lowest Rs per 100 on-net minutes",
                 lambda r: f"Rs {r['price_per_100_onnet']:,.1f}/100 min")
            pick("total_minutes", "idxmax", "Most call minutes",
                 lambda r: f"{r['total_minutes']:,.0f} min, Rs {r['price_pkr']:,.0f}")
            pick("sms", "idxmax", "Most SMS", lambda r: f"{r['sms']:,.0f} SMS, Rs {r['price_pkr']:,.0f}")

        if plan.similar_to and not table.empty and "role" in table:
            facts = self._similar_facts(table)
        return Comparison(table, summary, winners, notes, facts)

    @staticmethod
    def _group_facts(table: pd.DataFrame) -> list[str]:
        facts, tally = [], {}
        n_groups = table["group"].nunique()
        for g, rows in table.groupby("group", sort=False):
            desc = "; ".join(
                f"{r.operator.title()} {r.package_name}: Rs {r.price_pkr:,.0f}, {_n(r.data_gb)} GB, "
                f"{_n(r.minutes_eff)} min (on-net {_n(r.onnet_eff)}), {_n(r.sms)} SMS"
                for r in rows.itertuples())
            line = f"{g}: {desc}"
            if len(rows) >= 2:
                c = rows.loc[rows["price_pkr"].idxmin()]
                tally[c["operator"].title()] = tally.get(c["operator"].title(), 0) + 1
                line += f" || cheapest: {c['operator'].title()} (Rs {c['price_pkr']:,.0f})"
                for col, lab in (("data_gb", "most data"), ("onnet_eff", "most on-net minutes"), ("sms", "most SMS")):
                    s = rows[col].replace(float("inf"), 1e9).fillna(0)
                    if s.nunique() > 1 and s.max() > 0:
                        line += f"; {lab}: {rows.loc[s.idxmax()]['operator'].title()}"
            facts.append(line)
        facts.append(f"Groups where each operator is cheapest (of {n_groups}): {tally}")
        return facts

    @staticmethod
    def _similar_facts(table: pd.DataFrame) -> list[str]:
        facts = []
        refs = table[table["role"] == "REFERENCE"]
        for _, r in refs.iterrows():
            facts.append(f"REFERENCE = {r['operator'].title()} {r['package_name']}: Rs {r['price_pkr']:,.0f}, "
                         f"{r['validity_days']:g} days, data {_n(r['data_gb'])} GB, "
                         f"minutes {_n(r['minutes_eff'])}, sms {_n(r['sms'])}")
        for _, c in table[table["role"] != "REFERENCE"].iterrows():
            ref = refs[refs["package_name"] == c["match_for"]]
            if ref.empty: continue
            r = ref.iloc[0]
            facts.append(
                f"{c['operator'].title()} {c['package_name']} ({c.get('similarity', 'similar')}) vs {r['package_name']}: "
                f"price {_delta(c['price_pkr'], r['price_pkr'], ' Rs')}, "
                f"validity {_delta(c['validity_days'], r['validity_days'], ' days', 1)}, "
                f"data {_delta(c['data_gb'], r['data_gb'], ' GB', 1)}, "
                f"minutes {_delta(c['minutes_eff'], r['minutes_eff'])}, "
                f"sms {_delta(c['sms'], r['sms'])}")
        return facts


# ===================================================================== 4. ANSWER AGENT (LLM)
ANSWER_SYSTEM = f"""You are a helpful assistant that answers questions about Pakistani mobile packages
(Jazz, Zong, Ufone) using ONLY the DATA block provided.

IMPORTANT: the tables of matching packages are ALREADY shown to the user automatically, right above your text.
So: do NOT draw any table and do NOT list every package again. Write only:
  - first line: "Scope: <operators>, <duration>, <criterion>" taken from INTERPRETED FILTERS,
  - 2-5 short bullet insights,
  - then a one-line recommendation.

Column meanings: Internet = general internet data; Social/App = app-only data (WhatsApp, TikTok...);
Total Data = Internet + Social/App. Minutes lists on-net / off-net / all-net. Rs/GB is calculated on the
"{DATA_BASIS}" data column. Tax: Incl. / Excl. / '-' (not listed).

Rules:
1. Mention only packages that appear in MATCHING PACKAGES, OPERATOR SUMMARY, WINNERS or GROUP FACTS. Never invent
   packages, prices or features. Copy numbers exactly; never recompute. A package not in the DATA does not exist for you.
2. The table shows only some rows per operator; SUMMARY and WINNERS cover ALL matching packages.
3. For "near X GB" questions the rows are sorted by closeness to X: say which are exactly X and which are slightly
   below/above, and compare their prices.
4. If a REFERENCE package is given, explain how each similar package differs from it using the DIFFERENCES lines.
   Say they are the closest matches, not identical.
5. NOTES are authoritative. "NO EXACT MATCH" -> say so plainly and quote the funnel facts. "CLOSEST ALTERNATIVES"
   -> say clearly they are NOT exact matches. "NO PACKAGE with validity ..." -> say that operator has no package of
   that duration and mention the validities it does have; never present other durations as matches.
   "NO LIKE-FOR-LIKE" -> say there is no comparable package; never compare unrelated ones.
6. A '-' means not listed in the data. Mention Tax only if one operator says "Excl." and another "Incl.".
7. Be concise. Answer in the user's language (English or Roman Urdu).
8. Compare like with like. Use "significantly" only if the difference is 20% or more.
   When asked "which is better", state the criterion (price, data, on-net minutes...) for each verdict.
9. GROUP FACTS: packages with the same group label (G1, G2...) are like-for-like; compare only within a group. For
   "best value", use the per-group cheapest/most-resource lines and the last line (groups won per operator).
10. If the user asked to rank by a resource (on-net minutes, SMS, total minutes), the table is sorted by it; use WINNERS
    for that resource. If an operator has no row in the table, say it has no matching package."""


class AnswerAgent:
    @staticmethod
    def _summary_md(summary: pd.DataFrame) -> str:
        cols = {"operator": "Operator", "matches": "Matches", "min_price": "Min Rs", "median_price": "Median Rs",
                "max_price": "Max Rs", "cheapest": "Cheapest", "best_price_per_gb": "Best Rs/GB",
                "best_value_package": "Best-value pkg", "max_data_gb": "Max data (GB)",
                "max_data_package": "Max-data pkg"}
        s = summary[[c for c in cols if c in summary.columns]].rename(columns=cols)
        s = s.astype(object).where(s.notna(), "-")
        return s.to_markdown(index=False, floatfmt=".1f")

    def render_tables(self, plan: Plan, comp: Comparison | None) -> str:
        """Tables are built by CODE, so the LLM can never mistype a number."""
        if comp is None:
            return ""
        out = []
        if (plan.intent in ("compare_operators", "statistics") and not plan.similar_to
                and not comp.summary.empty and "matches" in comp.summary):
            out.append("**Operator summary (all matching packages)**\n\n" + self._summary_md(comp.summary))
        if comp.table.empty:
            return "\n\n".join(out)
        total = int(comp.summary["matches"].sum()) if "matches" in comp.summary else len(comp.table)
        if plan.similar_to:
            head = "**Reference package and closest like-for-like matches**"
        elif representative_mode(plan):
            head = (f"**Like-for-like packages** (same validity, same type, closest resources; "
                    f"{comp.table['group'].nunique()} groups, {len(comp.table)} packages of {total} matches)")
        else:
            head = f"**Matching packages** (showing {len(comp.table)} of {total})"
        out.append(head + "\n\n" + to_markdown(comp.table))
        return "\n\n".join(out)

    def build_context(self, query: str, plan: Plan, comp: Comparison) -> str:
        parts = [f"USER QUESTION: {query}",
                 f"INTERPRETED FILTERS: {plan.model_dump_json(exclude_defaults=True)}",
                 f"DURATION FILTER: {describe_validity(plan)} (strict)"]
        if not comp.summary.empty:
            parts.append("OPERATOR SUMMARY (over ALL matching packages):\n" + self._summary_md(comp.summary))
        if representative_mode(plan):
            parts.append("TABLE SELECTION: rows are like-for-like GROUPS across operators (same validity, same type, "
                         "close resources). SUMMARY and WINNERS cover all matching packages.")
        if not comp.table.empty:
            parts.append("MATCHING PACKAGES (already shown to the user):\n" + to_markdown(comp.table.head(18)))
        else:
            parts.append("MATCHING PACKAGES: none")
        if comp.winners:
            parts.append("WINNERS (over all matching packages):\n" + "\n".join(f"- {k}: {v}" for k, v in comp.winners.items()))
        if comp.facts:
            title = ("DIFFERENCES vs REFERENCE (precomputed, copy exactly)" if plan.similar_to
                     else "GROUP FACTS (precomputed, copy exactly)")
            parts.append(title + ":\n" + "\n".join("- " + f for f in comp.facts))
        if plan.intent in ("package_details", "compare_packages") and not comp.table.empty:
            det = [f"* {r['operator'].title()} / {r['package_name']}: {r['full_text'][:600]} | URL: {r['url'] or '-'}"
                   for _, r in comp.table.head(6).iterrows()]
            parts.append("FULL DETAILS (raw scraped text):\n" + "\n".join(det))
        if comp.notes:
            parts.append("NOTES:\n" + "\n".join(comp.notes))
        return "\n\n".join(parts)

    def stream_text(self, query, plan, comp, history):
        """LLM summary only (the tables are rendered separately by render_tables)."""
        msgs = [{"role": "system", "content": ANSWER_SYSTEM}]
        for q, a in history[-2:]:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a[-600:]}]
        if plan.intent == "chitchat":
            msgs.append({"role": "user", "content": query + "\n(Not a package question: reply briefly and "
                         "offer to help compare Jazz/Zong/Ufone packages.)"})
        else:
            msgs.append({"role": "user", "content": "DATA:\n" + self.build_context(query, plan, comp)})
        yield from chat_stream(msgs, think=THINK_FINAL)

    def stream(self, query, plan, comp, history):
        """Old API: table first, then text."""
        if plan.intent != "chitchat":
            pre = self.render_tables(plan, comp)
            if pre:
                yield pre + "\n\n---\n\n"
        yield from self.stream_text(query, plan, comp, history)