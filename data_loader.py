"""Loads the scraped Jazz / Zong / Ufone CSVs and normalises them into ONE unified schema."""
from __future__ import annotations
import math
import re
import numpy as np
import pandas as pd
from pathlib import Path
from config import DATA_DIR, OPERATOR_FILES, DATA_BASIS

NAN = float("nan")
# the scraped sheets store 30 MB as 0.03 GB, i.e. 1 GB = 1000 MB
_TO_GB = {"tb": 1000, "gb": 1, "mb": 1 / 1000}
_NULLS = {"", "nan", "none", "null", "n/a", "na", "-", "--"}


def _isnull(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip().lower() in _NULLS


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


# ------------------------------------------------------------------ column mapping
RULES = [
    ("package_name",     lambda k: k == "packagename"),
    ("site_category",    lambda k: k == "category"),
    ("validity_text",    lambda k: k == "validity"),
    ("validity_days",    lambda k: k.startswith("validitydays")),
    ("internet_gb",      lambda k: k.startswith("internetdata")),
    ("total_gb",         lambda k: k.startswith("totaldata")),
    ("social_gb",        lambda k: k.startswith("socialappdata")),
    ("social_breakdown", lambda k: "breakdown" in k),
    ("onnet",            lambda k: "onnet" in k and "offnet" not in k),
    ("offnet",           lambda k: k.startswith("offnet")),
    ("allnet",           lambda k: k.startswith("allnet")),
    ("sms",              lambda k: k == "sms"),
    ("price",            lambda k: k.startswith("price")),
    ("tax_note",         lambda k: k.startswith("tax")),
    ("resources",        lambda k: k.startswith("resources")),
    ("site_sections",    lambda k: k.startswith("sitesection")),
    ("found_on",         lambda k: k.startswith("foundon")),
    ("url",              lambda k: k.startswith("sourceurl")),
]


def map_columns(columns: list[str]) -> dict[str, str]:
    mapping, used = {}, set()
    for field, pred in RULES:
        for c in columns:
            if c not in used and pred(_key(c)):
                mapping[field] = c
                used.add(c)
                break
    return mapping


# ------------------------------------------------------------------ parsers
def parse_num(v) -> tuple[float, bool]:
    """Numeric cell -> (value, unlimited). Handles '10', '10 GB', '500 MB', 'Unlimited'."""
    if _isnull(v): return NAN, False
    s = str(v).lower().replace(",", "").strip()
    if "unlimited" in s or "unltd" in s: return NAN, True
    m = re.search(r"\d+(?:\.\d+)?", s)
    if not m: return NAN, False
    x = float(m.group())
    if re.search(r"\d\s*mb\b", s): x /= 1000
    return x, False


def parse_gb(v) -> tuple[float, bool]:
    return parse_num(v)


def parse_price(v) -> float:
    if _isnull(v): return NAN
    m = re.search(r"\d+(?:\.\d+)?", str(v).replace(",", ""))
    return float(m.group()) if m else NAN


def parse_validity_days(text) -> float:
    if _isnull(text): return NAN
    s = str(text).lower().replace(",", "")
    if re.fullmatch(r"\s*\d+(\.\d+)?\s*", s): return float(s)
    for pat, mult in [(r"(\d+(?:\.\d+)?)\s*(?:days?|din)\b", 1),
                      (r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|hr)\b", 1 / 24),
                      (r"(\d+(?:\.\d+)?)\s*weeks?\b", 7),
                      (r"(\d+(?:\.\d+)?)\s*months?\b", 30),
                      (r"(\d+(?:\.\d+)?)\s*years?\b", 365)]:
        m = re.search(pat, s)
        if m: return float(m.group(1)) * mult
    for kw, d in [("hourly", 1 / 24), ("daily", 1), ("weekly", 7), ("fortnight", 15),
                  ("monthly", 30), ("quarterly", 90), ("yearly", 365), ("annual", 365)]:
        if kw in s: return d
    return NAN


def normalise_category(site_cat: str, name: str, days: float) -> str:
    """Validity days decide the bucket (reliable). Site wording is only a fallback."""
    if not math.isnan(days):
        if days < 1: return "hourly"
        if days <= 1: return "daily"
        if days <= 7: return "weekly"          # 2-7 days
        if days <= 27: return "fortnightly"    # 8-27 days
        if days <= 31: return "monthly"        # 28-31 days
        return "long_term"
    text = f"{site_cat} {name}".lower()
    for k in ("monthly", "weekly", "daily", "hourly"):
        if k in text: return k
    return "other"


# ------------------------------------------------------------------ normalisation
def normalize_frame(raw: pd.DataFrame, operator: str) -> pd.DataFrame:
    raw = raw.rename(columns=lambda c: str(c).strip())
    m = map_columns(list(raw.columns))
    n0 = lambda x: 0.0 if (x is None or math.isnan(x)) else x
    rows = []
    for _, r in raw.iterrows():
        g = lambda f: (r[m[f]] if f in m else None)

        def s(f):
            v = g(f)
            return "" if _isnull(v) else str(v).strip()

        name = s("package_name")
        if not name:
            continue
        res_low = s("resources").lower().replace(",", "")

        # ---- price
        price = parse_price(g("price"))
        if math.isnan(price):
            mm = re.search(r"(?:rs\.?|pkr)\s*(\d+(?:\.\d+)?)", res_low)
            price = float(mm.group(1)) if mm else NAN

        # ---- validity
        days = parse_validity_days(g("validity_days"))
        if math.isnan(days): days = parse_validity_days(g("validity_text"))
        if math.isnan(days): days = parse_validity_days(name)

        # ---- data: reconcile the three columns (operators fill them differently)
        internet, int_unl = parse_gb(g("internet_gb"))
        total, tot_unl = parse_gb(g("total_gb"))
        social, _ = parse_gb(g("social_gb"))
        numeric_given = any(not math.isnan(x) for x in (internet, total, social))
        total_fix = max(n0(total), n0(internet) + n0(social))
        internet_fix = n0(internet) if n0(internet) > 0 else max(n0(total) - n0(social), 0.0)
        if total_fix == 0:                      # nothing numeric -> try the resources text
            vals = [float(n) * _TO_GB[u] for n, u in
                    re.findall(r"(\d+(?:\.\d+)?)\s*(tb|gb|mb)\b", res_low)]
            if vals:
                total_fix = internet_fix = max(vals)
        total_fix = total_fix if total_fix > 0 else NAN
        internet_fix = internet_fix if internet_fix > 0 else NAN
        social = social if n0(social) > 0 else NAN
        data_unl = bool(int_unl or tot_unl)
        # the text fallback for "unlimited" only applies when the numeric columns are blank;
        # cleaned rows have explicit 0s and must stay 0 (e.g. app-only "Unlimited Google Maps")
        if (not data_unl and not numeric_given and math.isnan(total_fix)
                and re.search(r"unlimited\s+(?:internet|data)", res_low)):
            data_unl = True
        data_gb = internet_fix if DATA_BASIS == "internet" else total_fix

        # ---- minutes
        on, on_u = parse_num(g("onnet"))
        off, off_u = parse_num(g("offnet"))
        alln, all_u = parse_num(g("allnet"))
        if all(math.isnan(x) for x in (on, off, alln)):
            mm = re.search(r"(\d[\d,]*)\s*(?:mins?|minutes)\b", res_low)
            if mm: alln = float(mm.group(1).replace(",", ""))
        parts = [x for x in (on, off, alln) if not math.isnan(x)]
        total_min = sum(parts) if parts else NAN
        min_unl = on_u or off_u or all_u
        if not parts and not min_unl and re.search(r"unlimited\s+(?:calls?|minutes|mins)", res_low):
            min_unl = True

        # ---- sms
        sms, sms_unl = parse_num(g("sms"))
        if math.isnan(sms) and not sms_unl:
            mm = re.search(r"(\d[\d,]*)\s*sms", res_low)
            if mm: sms = float(mm.group(1).replace(",", ""))
            elif re.search(r"unlimited\s+sms", res_low): sms_unl = True

        has_data = bool(data_unl or not math.isnan(total_fix))
        has_min = bool(min_unl or (not math.isnan(total_min) and total_min > 0))
        has_sms = bool(sms_unl or (not math.isnan(sms) and sms > 0))
        ptype = ("bundle" if has_data and (has_min or has_sms) else
                 "internet" if has_data else "call" if has_min else "sms" if has_sms else "other")

        full_text = " | ".join(f"{lbl}: {val}" for lbl, val in [
            ("Category", s("site_category")), ("Sections", s("site_sections")),
            ("Resources", s("resources")), ("Social/App", s("social_breakdown")),
            ("Tax", s("tax_note"))] if val)

        rows.append(dict(
            operator=operator, package_name=name, package_type=ptype,
            site_category=s("site_category"),
            category=normalise_category(s("site_category"), name, days),
            price_pkr=price, validity_days=days,
            data_gb=data_gb, data_unlimited=data_unl,
            internet_gb=internet_fix, internet_unlimited=bool(int_unl),
            total_gb=total_fix, total_unlimited=bool(tot_unl),
            social_gb=social, social_breakdown=s("social_breakdown"),
            onnet_minutes=on, offnet_minutes=off, allnet_minutes=alln,
            total_minutes=total_min, minutes_unlimited=bool(min_unl),
            sms=sms, sms_unlimited=bool(sms_unl),
            tax_note=s("tax_note"), url=s("url") or None, code=None,
            full_text=full_text,
        ))
    return pd.DataFrame(rows)


def _add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df["data_eff"] = np.where(df["data_unlimited"], np.inf, df["data_gb"])
    df["minutes_eff"] = np.where(df["minutes_unlimited"], np.inf, df["total_minutes"])
    # Rs/GB is meaningless for tiny allowances (e.g. 30 MB WhatsApp), so require >= 0.5 GB
    ok = (df["data_gb"] >= 0.5) & ~df["data_unlimited"]
    df["price_per_gb"] = np.where(ok, df["price_pkr"] / df["data_gb"], np.nan)
    df["price_per_day"] = np.where(df["validity_days"] > 0, df["price_pkr"] / df["validity_days"], np.nan)
    df["monthly_equiv_price"] = df["price_per_day"] * 30
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    first = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]
    sep = "\t" if first.count("\t") > first.count(",") else ","
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding=enc)
            return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")


def find_csv(data_dir: Path, op: str, fname: str) -> Path | None:
    """Exact filename first, then any CSV (also in subfolders) whose name contains the operator."""
    d = Path(data_dir)
    exact = d / fname
    if exact.exists():
        return exact
    if not d.exists():
        return None
    cands = [p for p in d.rglob("*")
             if p.is_file() and p.suffix.lower() == ".csv" and op in p.stem.lower()]
    return sorted(cands)[0] if cands else None


def load_all(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = []
    for op, fname in OPERATOR_FILES.items():
        p = find_csv(data_dir, op, fname)
        if p is None:
            print(f"[warn] no CSV for '{op}' in {data_dir} - skipping")
            continue
        print(f"[info] {op}: using {p.name}")
        frames.append(normalize_frame(_read_csv(p), op))
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["operator", "package_name", "price_pkr", "validity_days"])
    return _add_metrics(df).reset_index(drop=True)


def schema_report(data_dir: Path = DATA_DIR) -> str:
    """Shows column mapping, parse coverage and category values."""
    out = []
    for op, fname in OPERATOR_FILES.items():
        p = find_csv(data_dir, op, fname)
        if p is None:
            out.append(f"\n{op}: FILE MISSING in {data_dir}"); continue
        raw = _read_csv(p).rename(columns=lambda c: str(c).strip())
        mp = map_columns(list(raw.columns))
        out.append(f"\n== {op} ({len(raw)} rows) - {p.name} ==")
        out.append("Unmapped CSV columns: " + str([c for c in raw.columns if c not in mp.values()]))
        out.append("Missing fields: " + str([f for f, _ in RULES if f not in mp]))
        n = normalize_frame(raw, op)
        cov = {c: f"{n[c].notna().mean():.0%}" for c in
               ["price_pkr", "validity_days", "data_gb", "total_minutes", "sms"]}
        out.append(f"Parse coverage: {cov}")
        out.append(f"Site categories: {n['site_category'].value_counts().head(12).to_dict()}")
        out.append(f"Normalised category: {n['category'].value_counts().to_dict()}")
        out.append(f"Package types: {n['package_type'].value_counts().to_dict()}")
    return "\n".join(out)


def inspect_report(data_dir: Path = DATA_DIR) -> str:
    """What is REALLY in the data: counts per operator x validity bucket, max data and ranges."""
    df = load_all(data_dir)
    out = ["Packages per operator x category:\n" + pd.crosstab(df["operator"], df["category"]).to_string()]
    out.append("\nMax data (GB) per operator x category:\n" +
               df[~df["data_unlimited"]].pivot_table(index="operator", columns="category",
                                                     values="data_gb", aggfunc="max").round(1).to_string())
    out.append("\nPrice / validity / data ranges:\n" +
               df.groupby("operator")[["price_pkr", "validity_days", "data_gb"]]
                 .agg(["count", "min", "max"]).round(1).to_string())
    return "\n".join(out)