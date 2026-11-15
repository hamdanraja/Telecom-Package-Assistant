"""
clean_data.py - normalise the scraped Jazz / Zong / Ufone CSVs.

Raw scrapes live in data_raw/ (backed up automatically the first time).
Cleaned files are written to data/. Every change goes to data_raw/cleaning_report.csv.
New scrape? Put it in data_raw/ as jazz.csv / zong.csv / ufone.csv and re-run.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA, RAW = ROOT / "data", ROOT / "data_raw"
OPERATORS = ("jazz", "zong", "ufone")

COLS = {
    "name": ("package", "name"), "days": ("validity", "days"),
    "inet": ("internet data",), "total": ("total data",), "social": ("social", "data"),
    "on": ("on-net",), "off": ("off-net",), "all": ("all-net",),
    "price": ("price",), "text": ("resources",),
}
APP_RE = re.compile(r"insta|snapchat|tik ?tok|you ?tube|whatsapp|facebook|\bimo\b|pubg|"
                    r"free ?fire|snack ?video|google maps", re.I)
JUNK_RE = re.compile(r"\|?\s*\d+\s*minutes?\s+ago.*$", re.I | re.S)
TIME_RE = re.compile(r"night|\b\d{1,2}\s*[ap]m\b", re.I)
UF_ON = r"(\d[\d,]*)\s*(?:minutes\s+)?(?:Ufone|On[- ]?Net)\b"
UF_OFF = r"(\d[\d,]*)\s*(?:Off[- ]?Net|Other\s+Network)"
JZ_ON = r"(\d[\d,]*)\s*Jazz\s*Mins"
JZ_OFF = r"(\d[\d,]*)\s*Other\s+Network"
SMS_RE = r"(\d[\d,]*)\s*(?:Jazz\s*Mins\s*&\s*)?SMS"
MIN_PRICE = 1.0   # rows priced below this are dropped (placeholder prices)


# ---------- helpers ----------
def num(x):
    m = re.search(r"-?\d+(?:\.\d+)?", str(x).replace(",", ""))
    return float(m.group()) if m else None

def numz(x):
    v = num(x)
    return 0.0 if v is None else v

def isunl(x):
    return str(x).strip().lower().startswith("unlimited")

def fmt(v):
    if isinstance(v, str):
        return v
    v = float(v)
    return str(int(v)) if v.is_integer() else str(round(v, 4))

def same(a, b):
    if isunl(a) or isunl(b):
        return isunl(a) and isunl(b)
    return abs(numz(a) - numz(b)) < 1e-6

def maxnum(pattern, text):
    vals = [num(x) for x in re.findall(pattern, text, re.I)]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None

def read(path: Path):
    first = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]
    sep = "\t" if first.count("\t") > first.count(",") else ","
    for enc in ("utf-8-sig", "cp1252"):
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding=enc)
            df = df.loc[:, ~df.columns.str.startswith("Unnamed")]   # drop blank trailing columns
            return df, sep
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"cannot decode {path}")


# ---------- cleaning ----------
def clean_df(df: pd.DataFrame, op: str):
    c = {k: next((x for x in df.columns if all(n in x.lower() for n in need)), None)
         for k, need in COLS.items()}
    c["sms"] = next((x for x in df.columns if x.strip().lower() == "sms"), None)
    for k in ("name", "inet", "on", "off", "price"):
        if not c.get(k):
            print(f"  ! [{op}] could not find the '{k}' column - check headers")
    log, drop, junk = [], [], 0

    def get(i, k):
        return df.at[i, c[k]] if c.get(k) else ""

    def put(i, k, new, why, quiet=False):
        if not c.get(k) or same(get(i, k), new):
            return
        old = get(i, k)
        df.at[i, c[k]] = fmt(new)
        if not quiet:
            log.append((op, df.at[i, c["name"]], c[k], old, fmt(new), why))

    for i in df.index:
        name = str(get(i, "name")).strip()
        df.at[i, c["name"]] = name

        # 1) clean the resources text
        text = ""
        if c.get("text"):
            raw = str(get(i, "text"))
            text = re.sub(r"\s+", " ", JUNK_RE.sub("", raw)).strip(" |")
            junk += int(text != re.sub(r"\s+", " ", raw).strip(" |"))
            df.at[i, c["text"]] = text

        # 2) minutes / SMS re-read from text (Ufone + Jazz)
        if op in ("ufone", "jazz") and text:
            if op == "ufone":
                on, off = maxnum(UF_ON, text), maxnum(UF_OFF, text)
                unl_on = re.search(r"Unlimited\s+On[- ]?Net", text, re.I)
            else:
                on, off = maxnum(JZ_ON, text), maxnum(JZ_OFF, text)
                unl_on = None
            if unl_on:
                put(i, "on", "Unlimited", "text says unlimited on-net")
                if isunl(get(i, "off")):
                    put(i, "off", 0, "'Unlimited' was in the off-net column")
            elif on is not None:
                put(i, "on", on, "on-net minutes re-read from resources text")
            if off is not None and not isunl(get(i, "off")):
                put(i, "off", off, "off-net minutes re-read from resources text")
            s = maxnum(SMS_RE, text)
            if s is not None and s > numz(get(i, "sms")):
                put(i, "sms", s, "SMS re-read from resources text")

        # 3) Ufone data fixes
        if op == "ufone" and text and not isunl(get(i, "inet")):
            days = numz(get(i, "days"))
            m_month = re.search(r"(\d[\d.]*)\s*GB\s*/\s*Month", text, re.I)
            m_par = re.match(r"\s*(\d[\d.]*)\s*(GB|MB)\s*\(", text, re.I)
            m_net = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*GB\s*Internet", text, re.I)
            if m_month and days >= 60:
                put(i, "inet", float(m_month.group(1)) * round(days / 30),
                    "'GB / Month' multiplied over the validity")
            elif m_par:
                v = float(m_par.group(1)) / (1000 if m_par.group(2).upper() == "MB" else 1)
                put(i, "inet", v, "bracketed sub-allowances were added to the headline figure")
            elif m_net:
                put(i, "inet", num(m_net.group(1)), "internet figure re-read from text")
            m_soc = re.search(r"(\d[\d.]*)\s*Social\s*GBs?", text, re.I)
            if m_soc and numz(get(i, "social")) == 0:
                sv, iv = float(m_soc.group(1)), numz(get(i, "inet"))
                if iv >= sv:
                    put(i, "social", sv, "social GB split out of the headline figure")
                    put(i, "inet", iv - sv, "social GB split out of the headline figure")

        # 4) general vs app-only data
        I, T, S = (numz(get(i, k)) for k in ("inet", "total", "social"))
        voice = (any(isunl(get(i, k)) or numz(get(i, k)) > 0 for k in ("on", "off", "all"))
                 or numz(get(i, "sms")) > 0)
        app_only = bool(APP_RE.search(name)) and not voice
        if isunl(get(i, "inet")):
            if app_only:
                put(i, "inet", 0, "unlimited but app-only (e.g. Google Maps) - not general data")
        else:
            if I == 0 and T > S:
                I = T - S
                put(i, "inet", I, "general data taken from Total minus social")
            if app_only and I > 0:
                S += I
                put(i, "social", S, "app-specific package: moved to Social / App data")
                I = 0
                put(i, "inet", 0, "app-specific package: moved to Social / App data")
            put(i, "social", S, "social/app data", quiet=True)
            put(i, "total", I + S, "total = internet + social", quiet=True)

        # 5) placeholder prices
        p = num(get(i, "price"))
        if p is not None and p < MIN_PRICE:
            drop.append(i)
            log.append((op, name, c["price"], get(i, "price"), "(row dropped)",
                        "placeholder price - would win every 'cheapest' query"))

    df = df.drop(index=drop)

    # 6) numeric cells -> plain numbers (blank stays blank)
    for k in ("days", "inet", "total", "social", "on", "off", "all", "sms", "price"):
        if c.get(k):
            df[c[k]] = ["Unlimited" if isunl(v) else (fmt(num(v)) if num(v) is not None else "")
                        for v in df[c[k]]]

    # 7) warnings (reported, not changed)
    warn = []
    names = df[c["name"]].str.lower().str.strip()
    for n in names[names.duplicated(keep=False)].unique():
        warn.append(f"{op}: duplicate name '{n}' - make sure they are really different offers")
    for i in df.index:
        data, price = numz(get(i, "inet")) + numz(get(i, "social")), num(get(i, "price"))
        nm = df.at[i, c["name"]]
        if data >= 20 and price and price / data < 2.5:
            warn.append(f"{op}: {nm}: {data:g} GB for Rs {price:g} looks too cheap - check the source")
        if TIME_RE.search(nm + " " + str(get(i, "text"))):
            warn.append(f"{op}: {nm}: time-window / night data - counted as normal data")
    if junk:
        print(f"  removed scraper junk text from {junk} rows")
    return df, log, warn


def main():
    DATA.mkdir(exist_ok=True)
    RAW.mkdir(exist_ok=True)
    all_log, all_warn = [], []
    for op in OPERATORS:
        found = sorted(p for p in DATA.glob("*.csv") if op in p.stem.lower())
        src = RAW / f"{op}.csv"
        if not src.exists():
            if not found:
                print(f"[{op}] no CSV found in {DATA}")
                continue
            shutil.copy(found[0], src)
            print(f"[{op}] backed up {found[0].name} -> data_raw/{src.name}")
        target = DATA / f"{op}.csv" if (DATA / f"{op}.csv").exists() or not found else found[0]
        df, sep = read(src)
        print(f"[{op}] {len(df)} rows")
        df, log, warn = clean_df(df, op)
        df.to_csv(target, index=False, encoding="utf-8-sig", sep=sep)
        print(f"  -> {target.name}: {len(df)} rows written, {len(log)} changes")
        all_log += log
        all_warn += warn

    rep = pd.DataFrame(all_log, columns=["operator", "package", "column", "old", "new", "reason"])
    rep.to_csv(RAW / "cleaning_report.csv", index=False)
    print("\nCHANGES BY REASON")
    print(rep.groupby(["operator", "reason"]).size().to_string() if len(rep) else "  none")
    print("\nSAMPLE (full list in data_raw/cleaning_report.csv)")
    print(rep.head(40).to_string(index=False, max_colwidth=38) if len(rep) else "")
    print("\nWARNINGS (not changed - please verify at the source)")
    for w in all_warn[:40]:
        print("  -", w)


if __name__ == "__main__":
    main()