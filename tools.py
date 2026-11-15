from __future__ import annotations
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from schemas import Plan

SORT_COL = {
    "price": "price_pkr", "data_gb": "data_eff", "price_per_gb": "price_per_gb",
    "price_per_day": "price_per_day", "validity_days": "validity_days",
    "total_minutes": "minutes_eff", "sms": "sms",
}
DESC_BY_DEFAULT = {"data_gb", "total_minutes", "sms", "validity_days"}


def apply_filters(df: pd.DataFrame, p: Plan) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if p.package_type: m &= df["package_type"] == p.package_type
    if p.categories:   m &= df["category"].isin(p.categories)
    if p.min_price is not None: m &= df["price_pkr"] >= p.min_price
    if p.max_price is not None: m &= df["price_pkr"] <= p.max_price
    if p.min_data_gb is not None: m &= df["data_eff"] >= p.min_data_gb
    if p.max_data_gb is not None: m &= df["data_eff"] <= p.max_data_gb
    if p.min_minutes is not None: m &= df["minutes_eff"] >= p.min_minutes
    if p.min_sms is not None:     m &= (df["sms"] >= p.min_sms) | df["sms_unlimited"]
    if p.validity_days is not None: m &= (df["validity_days"] - p.validity_days).abs() <= 0.5
    if p.keywords:
        kws = [k.lower() for k in p.keywords]
        text = (df["package_name"] + " " + df["full_text"]).str.lower()
        m &= text.apply(lambda t: any(k in t for k in kws))
    return df[m]


def sort_df(df: pd.DataFrame, p: Plan) -> pd.DataFrame:
    # "near X GB" queries: closest to X first, then cheapest
    if p.target_data_gb and p.sort_by is None:
        dist = (df["data_eff"].replace(np.inf, 1e9) - p.target_data_gb).abs()
        out = df.assign(_dist=dist).sort_values(["_dist", "price_pkr"], na_position="last")
        return out.drop(columns="_dist")
    key = p.sort_by or "price"
    asc = p.ascending if p.ascending is not None else (key not in DESC_BY_DEFAULT)
    return df.sort_values(SORT_COL[key], ascending=asc, na_position="last")


# ------------------------------------------------------------------ comparable rows
TIERS = [(0.5, 5), (5, 15), (15, 40), (40, 100), (100, 1e9)]


def representative(df: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    """Cheapest package in each data tier (+ cheapest call-only pack) -> comparable rows across operators."""
    d = df[df["data_gb"].notna() & ~df["data_unlimited"]]
    rows = [t.sort_values("price_pkr").head(1)
            for lo, hi in TIERS
            for t in [d[(d["data_gb"] >= lo) & (d["data_gb"] < hi)]] if not t.empty]
    calls = df[df["data_gb"].isna() & ~df["data_unlimited"] & (df["minutes_eff"].fillna(0) > 0)]
    if not calls.empty:
        rows.append(calls.sort_values("price_pkr").head(1))
    if not rows:
        return df.head(limit)
    return pd.concat(rows).sort_values("data_gb", na_position="last").head(limit)


# ------------------------------------------------------------------ diagnostics + relaxation
FUNNEL_FIELDS = ["package_type", "categories", "validity_days", "min_data_gb", "max_data_gb",
                 "min_price", "max_price", "min_minutes", "min_sms", "keywords"]

RELAX_ORDER = [("keywords", []), ("validity_days", None), ("max_price", None), ("min_price", None),
               ("min_minutes", None), ("min_sms", None), ("package_type", None),
               ("categories", []), ("max_data_gb", None), ("min_data_gb", None)]


def funnel(df: pd.DataFrame, p: Plan) -> str:
    """Applies the filters one by one and says where the results disappeared."""
    cleared = {f: ([] if isinstance(getattr(p, f), list) else None) for f in FUNNEL_FIELDS}
    cur = p.model_copy(update=cleared)
    steps = [f"{len(df)} packages total"]
    prev = df
    for f in FUNNEL_FIELDS:
        v = getattr(p, f)
        if v is None or v == []:
            continue
        cur = cur.model_copy(update={f: v})
        out = apply_filters(df, cur)
        steps.append(f"{f}={v if not isinstance(v, float) else round(v, 1)} -> {len(out)} left")
        if out.empty:
            if f in ("min_data_gb", "max_data_gb"):
                d = prev.loc[~prev["data_unlimited"], "data_gb"].dropna()
                if not d.empty:
                    steps.append(f"data range available before this filter = {d.min():g} to {d.max():g} GB")
            if f in ("max_price", "min_price") and prev["price_pkr"].notna().any():
                steps.append(f"price range available before this filter = Rs {prev['price_pkr'].min():g} "
                             f"to Rs {prev['price_pkr'].max():g}")
            break
        prev = out
    return "; ".join(steps)


def relax(df: pd.DataFrame, p: Plan):
    """Widens a 'near X GB' band, then drops constraints one at a time until something matches."""
    cur, dropped = p, []
    t = p.target_data_gb
    if t and (p.min_data_gb is not None or p.max_data_gb is not None):
        for tol in (0.5, 1.0):
            cand = cur.model_copy(update={"min_data_gb": (t * (1 - tol)) if tol < 1 else None,
                                          "max_data_gb": t * (1 + tol)})
            out = apply_filters(df, cand)
            if not out.empty:
                return out, cand, [f"data range widened to +/-{int(tol * 100)}% around {t:g} GB"]
    for f, empty in RELAX_ORDER:
        v = getattr(cur, f)
        if v is None or v == []:
            continue
        cur = cur.model_copy(update={f: empty})
        dropped.append(f)
        out = apply_filters(df, cur)
        if not out.empty:
            if "min_data_gb" in dropped and p.sort_by is None and not p.target_data_gb:
                cur = cur.model_copy(update={"sort_by": "data_gb", "ascending": False})
            return out, cur, dropped
    return df.iloc[0:0], cur, dropped


# ------------------------------------------------------------------ name lookup
_STRIP = re.compile(r"\b(jazz|zong|ufone|mobilink|package|packages|bundle|offer|plan|the|of|in)\b", re.I)


def _compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_by_name(df: pd.DataFrame, name: str, top: int = 8, cutoff: int = 65) -> pd.DataFrame:
    """1) packages whose name CONTAINS the query ('super5' matches 'Super 5', 'Super 5 Call'); 2) fuzzy."""
    q = _STRIP.sub(" ", name).strip() or name
    qc = _compact(q)
    if qc:
        hit = df[df["package_name"].map(_compact).str.contains(re.escape(qc), regex=True)]
        if not hit.empty:
            out = hit.head(top).copy()
            out["match_for"] = name
            out["match_score"] = 100.0
            return out
    hits = process.extract(q, df["package_name"].tolist(), scorer=fuzz.WRatio, limit=3, score_cutoff=cutoff)
    if not hits:
        return df.iloc[0:0].assign(match_for=None, match_score=None)
    out = df.iloc[[h[2] for h in hits]].copy()
    out["match_for"] = name
    out["match_score"] = [h[1] for h in hits]
    return out


def find_reference(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Exact (spacing/case-insensitive) name match first, else contains/fuzzy."""
    q = _STRIP.sub(" ", name).strip() or name
    exact = df[df["package_name"].map(_compact) == _compact(q)]
    if not exact.empty:
        out = exact.copy()
        out["match_for"] = name
        out["match_score"] = 100.0
        return out
    return find_by_name(df, name, top=3)


# ------------------------------------------------------------------ similar packages
_W = {"validity": 3.0, "data": 2.0, "price": 1.5, "minutes": 1.0, "sms": 0.3}


def _feat(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["validity"] = df["validity_days"].astype(float).fillna(0)
    f["data"] = df["data_eff"].astype(float).replace(np.inf, 5000.0).fillna(0)
    f["price"] = df["price_pkr"].astype(float).fillna(0)
    f["minutes"] = df["minutes_eff"].astype(float).replace(np.inf, 1e6).fillna(0)
    f["sms"] = np.where(df["sms_unlimited"].astype(bool), 1e6, df["sms"].astype(float).fillna(0))
    return f


def find_similar(cands: pd.DataFrame, ref_row: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """ref_row = one-row DataFrame. Closest packages by log-distance on validity/data/price/minutes/sms."""
    if cands.empty:
        return cands
    fc, fr = _feat(cands), _feat(ref_row).iloc[0]
    dist = sum(w * (np.log1p(fc[c]) - np.log1p(fr[c])).abs() for c, w in _W.items())
    out = cands.assign(similarity_dist=dist).sort_values("similarity_dist").head(k).copy()
    out["match_for"] = ref_row["package_name"].iloc[0]
    out["match_score"] = 100.0
    return out


def operator_stats(df: pd.DataFrame) -> dict:
    if df.empty: return {"matches": 0}
    priced = df.dropna(subset=["price_pkr"])
    s = {"matches": int(len(df))}
    if not priced.empty:
        c = priced.loc[priced["price_pkr"].idxmin()]
        s.update(min_price=float(priced["price_pkr"].min()),
                 median_price=float(priced["price_pkr"].median()),
                 max_price=float(priced["price_pkr"].max()),
                 cheapest=c["package_name"])
    pg = df.dropna(subset=["price_per_gb"])
    if not pg.empty:
        b = pg.loc[pg["price_per_gb"].idxmin()]
        s.update(best_price_per_gb=float(b["price_per_gb"]), best_value_package=b["package_name"])
    d = df.replace([float("inf")], float("nan")).dropna(subset=["data_gb"])
    if not d.empty:
        b = d.loc[d["data_gb"].idxmax()]
        s.update(max_data_gb=float(b["data_gb"]), max_data_package=b["package_name"])
    return s