import math
import pandas as pd


def _isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _num(x, nd=0):
    if _isnan(x): return "-"
    if isinstance(x, float) and math.isinf(x): return "Unlimited"
    s = f"{x:,.{nd}f}"
    return s.rstrip("0").rstrip(".") if nd else s


def _gb(val, unl):
    if unl: return "Unlimited"
    if _isnan(val): return "-"
    return f"{val * 1000:,.0f} MB" if val < 1 else _num(val, 2) + " GB"


def _minutes(r):
    if r["minutes_unlimited"]: return "Unlimited"
    parts = []
    for col, label in (("onnet_minutes", "on-net"), ("offnet_minutes", "off-net"), ("allnet_minutes", "all-net")):
        if not _isnan(r[col]) and r[col] > 0:
            parts.append(f"{_num(r[col])} {label}")
    return ", ".join(parts) if parts else "-"


def _sms(r):
    if r["sms_unlimited"]: return "Unlimited"
    return _num(r["sms"]) if not _isnan(r["sms"]) and r["sms"] > 0 else "-"


def _validity(d):
    if _isnan(d): return "-"
    return f"{d * 24:.0f} h" if d < 1 else _num(d, 1)


def _tax(t):
    t = (t or "").lower()
    return "Incl." if "incl" in t else "Excl." if "excl" in t else "-"


def display_frame(df: pd.DataFrame) -> pd.DataFrame:
    has_role = "role" in df.columns
    rows = []
    for _, r in df.iterrows():
        row = {"Operator": r["operator"].title(), "Package": r["package_name"]}
        if has_role:
            row["Role"] = r["role"] if isinstance(r["role"], str) else ""
        row.update({
            "Type": str(r["package_type"]).title(),
            "Price (Rs)": _num(r["price_pkr"]),
            "Validity (days)": _validity(r["validity_days"]),
            "Internet": _gb(r["internet_gb"], r["internet_unlimited"]),
            "Social/App": _gb(r["social_gb"], False),
            "Total Data": _gb(r["total_gb"], r["total_unlimited"]),
            "Minutes": _minutes(r),
            "SMS": _sms(r),
            "Rs/GB": _num(r["price_per_gb"], 1),
            "Rs/day": _num(r["price_per_day"], 1),
            "Tax": _tax(r["tax_note"]),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty: return "(no rows)"
    return display_frame(df).to_markdown(index=False)