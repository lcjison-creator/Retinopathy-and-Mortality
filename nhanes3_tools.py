"""
nhanes3_tools.py  (v9.1)
========================
v9.1 (26 Sep 2026): absolute_risk_suite() now evaluates the plotted standardized curves at exact time points
(v9 returned the previous month's value at one in three monthly points); all other estimands are unchanged.

Helper functions for the NHANES III (1988-1994) retina x ECG x linked-mortality pipeline, and (v9) for the
NHANES 2005-2008 replication and the absolute-risk add-on:
  * continuous-NHANES SAS transport (XPT) download/reading and LMF validation (read_lmf(expected_seqn=...));
  * delete-one-PSU jackknife for any estimator (survey_jackknife);
  * covariate-standardized (g-formula) cumulative mortality / cause-specific cumulative incidence,
    RMST / RMTL, survey-weighted Harrell's C, fixed-effect pooling (absolute_risk_suite and helpers).
v9 only adds functions; every v8 function is unchanged (regression-tested on the synthetic fixture).

Design rules
------------
* Every number produced downstream is computed from the real NHANES III public-use files.
  Nothing in this module generates placeholder or simulated values.
* Fixed-width layouts are parsed from the official SAS read-in programs distributed by NCHS,
  so column positions never have to be typed by hand.
* Variable labels are parsed from the same SAS programs and printed back to the analyst,
  so every variable mapping can be checked against the official label before use.

Public-use sources (all free, no application required)
-------------------------------------------------------
NHANES III Series 11 No. 1A (July 1997): adult.dat / exam.dat / lab.dat (+ .sas, -acc.pdf)
NHANES III Series 11 No. 2A (April 1998): nh3ecg.dat / nh3ecg.sas / NH3ECG-acc.pdf
NCHS 2019 Public-use Linked Mortality File: NHANES_III_MORT_2019_PUBLIC.dat
    (layout from the NCHS 2019 read-in programs: SEQN in columns 1-6)
"""
from __future__ import annotations

import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# 1. File registry
# --------------------------------------------------------------------------------------
NH3_BASE = "https://wwwn.cdc.gov/nchs/data/nhanes3"
LMF_BASE = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality"

NHANES3_FILES: Dict[str, Dict[str, str]] = {
    # Household adult questionnaire (age >= 17 at interview)
    "adult": {
        "dat": f"{NH3_BASE}/1a/adult.dat",
        "sas": f"{NH3_BASE}/1a/adult.sas",
        "doc": f"{NH3_BASE}/1a/ADULT-acc.pdf",
    },
    # MEC examination file (body measures, blood pressure, fundus photography grading = FPP*)
    "exam": {
        "dat": f"{NH3_BASE}/1a/exam.dat",
        "sas": f"{NH3_BASE}/1a/exam.sas",
        "doc": f"{NH3_BASE}/1a/exam-acc.pdf",
    },
    # Laboratory file (HbA1c GHP, lipids TCP/HDP/TGP, creatinine CEP, CRP, glucose G1P ...)
    "lab": {
        "dat": f"{NH3_BASE}/1a/lab.dat",
        "sas": f"{NH3_BASE}/1a/lab.sas",
        "doc": f"{NH3_BASE}/1a/lab-acc.pdf",
    },
    # Resting 12-lead ECG, age >= 40 (Marquette MAC 12, Novacode, Minnesota code)
    "ecg": {
        "dat": f"{NH3_BASE}/2a/nh3ecg.dat",
        "sas": f"{NH3_BASE}/2a/nh3ecg.sas",
        "doc": f"{NH3_BASE}/2a/NH3ECG-acc.pdf",
    },
    # NCHS 2019 public-use linked mortality file (follow-up through 31 Dec 2019)
    "mort": {
        "dat": f"{LMF_BASE}/NHANES_III_MORT_2019_PUBLIC.dat",
        "doc": "https://www.cdc.gov/nchs/data/datalinkage/public-use-linked-mortality-files-data-dictionary.pdf",
    },
}


# --------------------------------------------------------------------------------------
# 2. Download helper
# --------------------------------------------------------------------------------------
def download(url: str, dest: Path, overwrite: bool = False, timeout: int = 600) -> Path:
    """Download `url` to `dest` (streamed). Skips the download if the file already exists."""
    import requests  # imported lazily so the parser functions work without network

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        print(f"[skip] {dest.name} already present ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    print(f"[get ] {url}")
    t0 = time.time()
    with requests.get(url, stream=True, timeout=timeout,
                      headers={"User-Agent": "Mozilla/5.0 (research pipeline)"}) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        n = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
                    n += len(chunk)
        tmp.replace(dest)
    print(f"      -> {dest.name}: {n/1e6:.1f} MB in {time.time()-t0:.0f} s")
    return dest


def fetch_all(data_dir: Path, keys: Iterable[str] = ("adult", "exam", "lab", "ecg", "mort"),
              overwrite: bool = False) -> Dict[str, Dict[str, Path]]:
    """Download every registered file into `data_dir` and return local paths."""
    out: Dict[str, Dict[str, Path]] = {}
    for k in keys:
        out[k] = {}
        for kind, url in NHANES3_FILES[k].items():
            if kind == "doc":
                continue  # documentation PDFs are for the analyst; links are printed instead
            out[k][kind] = download(url, Path(data_dir) / Path(url).name, overwrite=overwrite)
    return out


# --------------------------------------------------------------------------------------
# 3. SAS read-in program parsing (INPUT + LABEL statements)
# --------------------------------------------------------------------------------------
@dataclass
class Field:
    name: str
    start: int          # 1-based inclusive
    end: int            # 1-based inclusive
    char: bool = False
    dec: int = 0        # implied decimals (SAS "d" in w.d or trailing ".d")

    @property
    def width(self) -> int:
        return self.end - self.start + 1


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")
_COL_RE = re.compile(r"^\d+$")
_DEC_RE = re.compile(r"^\.(\d+)$")
_FMT_RE = re.compile(r"^(\$)?(\d+)\.(\d*)$")          # 5.   $2.   9.2
_SAS_KEYWORDS = {"INPUT", "INFILE", "DATA", "SET", "RUN", "LENGTH", "FORMAT", "LABEL",
                 "PROC", "LRECL", "MISSOVER", "PAD", "TRUNCOVER", "FIRSTOBS", "OBS"}


def _strip_sas_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    # "* comment ;" statements (only when '*' starts a statement)
    text = re.sub(r"(?m)^\s*\*[^;]*;", " ", text)
    return text


def _extract_statement(text: str, keyword: str) -> Optional[str]:
    """Return the body of the first `keyword ... ;` statement (quotes respected)."""
    pat = re.compile(rf"\b{keyword}\b", re.I)
    m = pat.search(text)
    while m:
        # make sure the keyword starts a statement (preceded by ';' or line start)
        prev = text[:m.start()].rstrip()
        if prev == "" or prev.endswith(";") or prev.endswith("\n"):
            break
        m = pat.search(text, m.end())
    if not m:
        return None
    i = m.end()
    body = []
    quote = None
    while i < len(text):
        c = text[i]
        if quote:
            if c == quote:
                quote = None
        elif c in ("'", '"'):
            quote = c
        elif c == ";":
            break
        body.append(c)
        i += 1
    return "".join(body)


def parse_sas_input(sas_text: str) -> List[Field]:
    """Parse the fixed-width layout from a SAS INPUT statement.

    Supports the NCHS column style (``SEQN 1-5``, ``HSSEX 15``, ``NAME $ 10-12``,
    ``WT 59-67 .2``) and the formatted style (``@1 SEQN 5.``, ``NAME $2.``).
    """
    text = _strip_sas_comments(sas_text)
    body = _extract_statement(text, "INPUT")
    if body is None:
        raise ValueError("No INPUT statement found in SAS program")
    toks = body.replace(",", " ").split()
    fields: List[Field] = []
    pointer = 1
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("@"):
            if t[1:].isdigit():
                pointer = int(t[1:])
            i += 1
            continue
        if t == "#" or t.startswith("#"):
            i += 1
            continue
        if not _NAME_RE.match(t) or t.upper() in _SAS_KEYWORDS:
            i += 1
            continue
        name = t.upper()
        char = False
        j = i + 1
        if j < len(toks) and toks[j] == "$":
            char = True
            j += 1
        if j >= len(toks):
            break
        spec = toks[j]
        f: Optional[Field] = None
        m = _RANGE_RE.match(spec)
        if m:
            f = Field(name, int(m.group(1)), int(m.group(2)), char)
            j += 1
        elif _COL_RE.match(spec):
            f = Field(name, int(spec), int(spec), char)
            j += 1
        else:
            m = _FMT_RE.match(spec)
            if m:
                char = char or (m.group(1) == "$")
                w = int(m.group(2))
                dec = int(m.group(3)) if m.group(3) else 0
                f = Field(name, pointer, pointer + w - 1, char, dec)
                pointer += w
                j += 1
        if f is None:
            i += 1  # bare name without a column spec (list input) - unsupported, skip
            continue
        # optional trailing implied-decimal spec ".2"
        if j < len(toks):
            m = _DEC_RE.match(toks[j])
            if m:
                f.dec = int(m.group(1))
                j += 1
        fields.append(f)
        pointer = max(pointer, f.end + 1)
        i = j
    if not fields:
        raise ValueError("INPUT statement parsed but no fields recognised")
    return fields


def parse_sas_labels(sas_text: str) -> Dict[str, str]:
    """Parse ``LABEL name = "text"`` pairs (all LABEL statements in the program)."""
    text = _strip_sas_comments(sas_text)
    labels: Dict[str, str] = {}
    pair = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\"((?:[^\"]|\"\")*)\"|'((?:[^']|'')*)')")
    for m in re.finditer(r"\bLABEL\b", text, flags=re.I):
        pos = m.end()
        while True:
            pm = pair.match(text, pos)
            if not pm:
                break
            val = pm.group(3) if pm.group(3) is not None else pm.group(4)
            labels[pm.group(1).upper()] = val.replace('""', '"').replace("''", "'").strip()
            pos = pm.end()
    return labels


def find_vars(labels: Dict[str, str], *keywords: str, prefix: Optional[str] = None) -> pd.DataFrame:
    """Search variable labels (case-insensitive AND of keywords). Handy for mapping covariates."""
    rows = []
    for k, v in labels.items():
        if prefix and not k.startswith(prefix.upper()):
            continue
        if all(kw.lower() in v.lower() for kw in keywords):
            rows.append((k, v))
    return pd.DataFrame(rows, columns=["variable", "label"])


# --------------------------------------------------------------------------------------
# 4. Fixed-width readers
# --------------------------------------------------------------------------------------
def _has_newlines(path: Path, probe: int = 65536) -> bool:
    with open(path, "rb") as fh:
        head = fh.read(probe)
    return b"\n" in head


def read_fixed(path: Path, fields: Sequence[Field], keep: Optional[Iterable[str]] = None,
               lrecl: Optional[int] = None) -> pd.DataFrame:
    """Read selected variables from a fixed-width NCHS file using a parsed SAS layout.

    Works whether records are newline-terminated (normal) or a pure fixed-record stream.
    Blank fields become NaN; implied decimals are applied when the raw field has no '.'.
    """
    path = Path(path)
    byname = {f.name: f for f in fields}
    if keep is None:
        sel = list(fields)
    else:
        missing = [k.upper() for k in keep if k.upper() not in byname]
        if missing:
            raise KeyError(f"Variables not in layout of {path.name}: {missing}")
        sel = [byname[k.upper()] for k in keep]
    colspecs = [(f.start - 1, f.end) for f in sel]
    names = [f.name for f in sel]

    if _has_newlines(path):
        df = pd.read_fwf(path, colspecs=colspecs, names=names, header=None, dtype=str,
                         encoding="latin-1", keep_default_na=False)
    else:
        rec = lrecl or max(f.end for f in fields)
        raw = path.read_bytes()
        n = len(raw) // rec
        rows = [[raw[r * rec + a: r * rec + b].decode("latin-1") for a, b in colspecs] for r in range(n)]
        df = pd.DataFrame(rows, columns=names)

    for f in sel:
        s = df[f.name].astype(str).str.strip()
        s = s.where(s != "", other=np.nan)
        if f.char:
            df[f.name] = s
            continue
        num = pd.to_numeric(s, errors="coerce")
        if f.dec:
            no_point = ~s.fillna("").str.contains(r"\.", regex=True)
            num = pd.Series(np.where(no_point, num / (10 ** f.dec), num), index=df.index)
        df[f.name] = num
    return df


def load_nhanes3(dat_path: Path, sas_path: Path, keep: Iterable[str]) -> pd.DataFrame:
    """Parse the SAS program, then read only `keep` variables from the .dat file."""
    sas_text = Path(sas_path).read_text(encoding="latin-1", errors="replace")
    fields = parse_sas_input(sas_text)
    return read_fixed(dat_path, fields, keep=list(keep))


# ---- NCHS 2019 public-use linked mortality file --------------------------------------
# NHANES layout used by the NCHS 2019 read-in programs (R: readr::read_fwf with
# fwf_cols(seqn = c(1,6), eligstat = c(15,15), mortstat = c(16,16), ucod_leading = c(17,19),
# diabetes = c(20,20), hyperten = c(21,21), permth_int = c(43,45), permth_exm = c(46,48)),
# na = c("", ".")). Columns 7-14 (rest of the NHIS PUBLICID field) and 22-42 (NHIS-only
# fields) are not used for NHANES; columns 7-14 are read only as a diagnostic.
LMF_LAYOUT: List[Field] = [
    Field("SEQN_RAW", 1, 6, char=True),
    Field("ID_TAIL", 7, 14, char=True),
    Field("ELIGSTAT", 15, 15),
    Field("MORTSTAT", 16, 16),
    Field("UCOD_LEADING", 17, 19),
    Field("DIABETES", 20, 20),
    Field("HYPERTEN", 21, 21),
    Field("PERMTH_INT", 43, 45),
    Field("PERMTH_EXM", 46, 48),
]
# NHANES III documentation (Series 11 No. 7A core file): SEQN "Sample person identification
# number", range 00003-53623, 33,994 records. SEQN is an identifier, not a row index.
NHANES3_SEQN_RANGE = (3, 53623)
NHANES3_N_PERSONS = 33994
UCOD_LABELS = {1: "Diseases of heart", 2: "Malignant neoplasms", 3: "Chronic lower respiratory diseases",
               4: "Accidents (unintentional injuries)", 5: "Cerebrovascular diseases",
               6: "Alzheimer's disease", 7: "Diabetes mellitus", 8: "Influenza and pneumonia",
               9: "Nephritis, nephrotic syndrome and nephrosis", 10: "All other causes"}


def _print_raw_records(path: Path, n: int = 3) -> None:
    with open(path, "rb") as fh:
        for i in range(n):
            line = fh.readline()
            if not line:
                break
            print(f"    raw record {i + 1}: {line!r}")


def read_lmf(path: Path, adult_dat: Optional[Path] = None, adult_sas: Optional[Path] = None,
             min_link_rate: float = 0.95, verbose: bool = True,
             expected_seqn: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """Read an NCHS 2019 public-use linked mortality file (NHANES layout) and validate it.

    Hard checks (stop on failure):
      * every record yields a numeric SEQN from columns 1-6, and SEQN is unique;
      * ELIGSTAT in {1, 2, 3}; MORTSTAT in {0, 1} when ELIGSTAT = 1;
      * UCOD_LEADING in 1-10 for deaths; PERMTH_INT / PERMTH_EXM non-negative;
      * linkage: >= `min_link_rate` of the expected participants are found in the file.
        NHANES III (default): adult.dat participants aged >= 18 (adult.dat/adult.sas are looked up
        in the same folder unless given explicitly). Continuous NHANES: pass `expected_seqn`
        (e.g. every SEQN of the matching DEMO file); the NHANES III range check is then skipped.
    Diagnostics (printed): record count and SEQN range (versus the NHANES III documentation when
    reading the NHANES III file). If no linkage check can run, the documented NHANES III SEQN range
    becomes a hard check instead.
    """
    if expected_seqn is not None:
        return _read_lmf_generic(Path(path), np.asarray(list(expected_seqn), dtype=float),
                                 min_link_rate=min_link_rate, verbose=verbose)
    path = Path(path)
    df = read_fixed(path, LMF_LAYOUT)
    df["SEQN"] = pd.to_numeric(df["SEQN_RAW"].str.strip(), errors="coerce")

    # drop records with no content at all (e.g. a trailing EOF marker such as \x1a)
    empty = (df["SEQN"].isna() & df["ELIGSTAT"].isna() & df["MORTSTAT"].isna()
             & df["PERMTH_INT"].isna() & df["PERMTH_EXM"].isna())
    if empty.any():
        print(f"[LMF] dropped {int(empty.sum())} empty record(s)")
        df = df[~empty].reset_index(drop=True)

    elig1 = df["ELIGSTAT"] == 1
    dead = df["MORTSTAT"] == 1
    lo, hi = NHANES3_SEQN_RANGE

    checks = {
        "every record has a numeric SEQN in columns 1-6": bool(df["SEQN"].notna().all()),
        "SEQN unique": bool(df["SEQN"].dropna().is_unique),
        "ELIGSTAT in {1,2,3}": bool(df["ELIGSTAT"].isin([1, 2, 3]).all()),
        "MORTSTAT in {0,1} when ELIGSTAT = 1": bool(df.loc[elig1, "MORTSTAT"].isin([0, 1]).all()),
        "UCOD_LEADING in 1-10 for deaths": bool(df.loc[dead, "UCOD_LEADING"].dropna().isin(range(1, 11)).all()),
        "PERMTH_INT/PERMTH_EXM non-negative": bool(df["PERMTH_INT"].dropna().ge(0).all()
                                                   and df["PERMTH_EXM"].dropna().ge(0).all()),
    }
    in_range = bool(df["SEQN"].dropna().between(lo, hi).all())

    # --- linkage cross-check against the NHANES III adult file ---------------------------------
    adult_dat = Path(adult_dat) if adult_dat else path.parent / "adult.dat"
    adult_sas = Path(adult_sas) if adult_sas else path.parent / "adult.sas"
    link_rate, n_adult18 = None, 0
    if adult_dat.exists() and adult_sas.exists():
        fields = parse_sas_input(adult_sas.read_text(encoding="latin-1", errors="replace"))
        a = read_fixed(adult_dat, fields, keep=["SEQN", "HSAGEIR"])
        a18 = a[a["HSAGEIR"] >= 18]
        n_adult18 = len(a18)
        link_rate = float(a18["SEQN"].isin(df["SEQN"]).mean()) if n_adult18 else np.nan
        checks[f">= {min_link_rate:.0%} of adult.dat participants aged >= 18 found in LMF"] = \
            bool(n_adult18 > 0 and link_rate >= min_link_rate)
    else:
        checks[f"SEQN within documented NHANES III range {lo}-{hi} (linkage check unavailable)"] = in_range

    if verbose:
        s = df["SEQN"].dropna()
        print(f"[LMF] {path.name}: {len(df):,} records (NHANES III documentation: {NHANES3_N_PERSONS:,} persons)")
        if len(s):
            print(f"[LMF] SEQN range {int(s.min())}-{int(s.max())} "
                  f"(documented {lo}-{hi}; within range: {'yes' if in_range else 'NO'})")
        print(f"[LMF] columns 7-14 non-blank in {int(df['ID_TAIL'].notna().sum())} record(s) (expected 0 for NHANES)")
        print(f"[LMF] ELIGSTAT counts: {df['ELIGSTAT'].value_counts(dropna=False).sort_index().to_dict()}")
        print(f"[LMF] MORTSTAT among ELIGSTAT = 1: {df.loc[elig1, 'MORTSTAT'].value_counts(dropna=False).sort_index().to_dict()}")
        print(f"[LMF] UCOD_LEADING among deaths: {df.loc[dead, 'UCOD_LEADING'].value_counts(dropna=False).sort_index().to_dict()}")
        pe = df.loc[elig1, "PERMTH_EXM"].dropna()
        if len(pe):
            print(f"[LMF] PERMTH_EXM (months) among ELIGSTAT = 1: min {pe.min():.0f}, median {pe.median():.0f}, max {pe.max():.0f}")
        if link_rate is not None:
            print(f"[LMF] linkage: {100 * link_rate:.2f}% of {n_adult18:,} adult.dat participants aged >= 18 found in LMF")

    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        print("[LMF] first raw records, for diagnosis:")
        _print_raw_records(path)
        raise ValueError("Linked mortality file failed validation: " + "; ".join(bad)
                         + ". Compare the raw records above with the NCHS 2019 read-in program layout "
                           "(SEQN 1-6, ELIGSTAT 15, MORTSTAT 16, UCOD_LEADING 17-19, PERMTH_INT 43-45, PERMTH_EXM 46-48).")

    df["SEQN"] = df["SEQN"].astype("int64")
    return df[["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "DIABETES", "HYPERTEN",
               "PERMTH_INT", "PERMTH_EXM"]]


def _read_lmf_generic(path: Path, expected_seqn: np.ndarray, min_link_rate: float = 0.95,
                      verbose: bool = True) -> pd.DataFrame:
    """read_lmf() for continuous-NHANES files: same layout and hard checks; linkage is checked
    against `expected_seqn` (e.g. all SEQN of the matching DEMO file)."""
    df = read_fixed(path, LMF_LAYOUT)
    df["SEQN"] = pd.to_numeric(df["SEQN_RAW"].str.strip(), errors="coerce")
    empty = (df["SEQN"].isna() & df["ELIGSTAT"].isna() & df["MORTSTAT"].isna()
             & df["PERMTH_INT"].isna() & df["PERMTH_EXM"].isna())
    if empty.any():
        print(f"[LMF] dropped {int(empty.sum())} empty record(s)")
        df = df[~empty].reset_index(drop=True)
    elig1 = df["ELIGSTAT"] == 1
    dead = df["MORTSTAT"] == 1
    exp = pd.Series(expected_seqn).dropna()
    link_rate = float(exp.isin(df["SEQN"]).mean()) if len(exp) else np.nan
    checks = {
        "every record has a numeric SEQN in columns 1-6": bool(df["SEQN"].notna().all()),
        "SEQN unique": bool(df["SEQN"].dropna().is_unique),
        "ELIGSTAT in {1,2,3}": bool(df["ELIGSTAT"].isin([1, 2, 3]).all()),
        "MORTSTAT in {0,1} when ELIGSTAT = 1": bool(df.loc[elig1, "MORTSTAT"].isin([0, 1]).all()),
        "UCOD_LEADING in 1-10 for deaths": bool(df.loc[dead, "UCOD_LEADING"].dropna().isin(range(1, 11)).all()),
        "PERMTH_INT/PERMTH_EXM non-negative": bool(df["PERMTH_INT"].dropna().ge(0).all()
                                                   and df["PERMTH_EXM"].dropna().ge(0).all()),
        f">= {min_link_rate:.0%} of expected participants found in LMF": bool(len(exp) > 0 and link_rate >= min_link_rate),
    }
    if verbose:
        s = df["SEQN"].dropna()
        print(f"[LMF] {path.name}: {len(df):,} records; SEQN range "
              f"{int(s.min()) if len(s) else 'NA'}-{int(s.max()) if len(s) else 'NA'}")
        print(f"[LMF] columns 7-14 non-blank in {int(df['ID_TAIL'].notna().sum())} record(s) (expected 0 for NHANES)")
        print(f"[LMF] ELIGSTAT counts: {df['ELIGSTAT'].value_counts(dropna=False).sort_index().to_dict()}")
        print(f"[LMF] MORTSTAT among ELIGSTAT = 1: {df.loc[elig1, 'MORTSTAT'].value_counts(dropna=False).sort_index().to_dict()}")
        print(f"[LMF] UCOD_LEADING among deaths: {df.loc[dead, 'UCOD_LEADING'].value_counts(dropna=False).sort_index().to_dict()}")
        pe = df.loc[elig1, "PERMTH_EXM"].dropna()
        if len(pe):
            print(f"[LMF] PERMTH_EXM (months) among ELIGSTAT = 1: min {pe.min():.0f}, median {pe.median():.0f}, max {pe.max():.0f}")
        print(f"[LMF] linkage: {100 * link_rate:.2f}% of {len(exp):,} expected participants found in LMF")
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        print("[LMF] first raw records, for diagnosis:")
        _print_raw_records(path)
        raise ValueError("Linked mortality file failed validation: " + "; ".join(bad)
                         + ". Compare the raw records above with the NCHS 2019 read-in program layout "
                           "(SEQN 1-6, ELIGSTAT 15, MORTSTAT 16, UCOD_LEADING 17-19, PERMTH_INT 43-45, PERMTH_EXM 46-48).")
    df["SEQN"] = df["SEQN"].astype("int64")
    return df[["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "DIABETES", "HYPERTEN",
               "PERMTH_INT", "PERMTH_EXM"]]


# --------------------------------------------------------------------------------------
# 4b. Continuous NHANES (2005-2008) public-use files: SAS transport (XPT) + LMF
# --------------------------------------------------------------------------------------
NHANES_CONT_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
NHANES_CYCLES = {"D": ("2005-2006", 2005), "E": ("2007-2008", 2007)}


def nhanes_cont_url(component: str, suffix: str, kind: str = "xpt") -> str:
    """URL of a continuous-NHANES data file (kind='xpt') or its codebook page (kind='htm'),
    e.g. nhanes_cont_url('OPXRET', 'D') ->
    https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/OPXRET_D.xpt"""
    begin = NHANES_CYCLES[suffix][1]
    return f"{NHANES_CONT_BASE}/{begin}/DataFiles/{component}_{suffix}.{kind}"


def lmf_cont_url(suffix: str) -> str:
    label = NHANES_CYCLES[suffix][0].replace("-", "_")
    return f"{LMF_BASE}/NHANES_{label}_MORT_2019_PUBLIC.dat"


def fetch_nhanes_cont(data_dir: Path, components: Iterable[str], suffixes: Iterable[str] = ("D", "E"),
                      overwrite: bool = False) -> Dict[str, Dict[str, Path]]:
    """Download continuous-NHANES XPT files and the matching 2019 public-use LMF files.
    Returns {suffix: {component: path, ..., 'MORT': path}}."""
    out: Dict[str, Dict[str, Path]] = {}
    for sfx in suffixes:
        out[sfx] = {}
        for comp in components:
            url = nhanes_cont_url(comp, sfx)
            out[sfx][comp] = download(url, Path(data_dir) / Path(url).name, overwrite=overwrite)
        url = lmf_cont_url(sfx)
        out[sfx]["MORT"] = download(url, Path(data_dir) / Path(url).name, overwrite=overwrite)
    return out


def read_xpt(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read a NHANES SAS transport file. Values that are zero in SAS are sometimes returned as
    ~5.4e-79 by the XPT (IBM float) conversion; they are set back to exactly 0. Missing requested
    columns raise a KeyError listing what the file contains."""
    df = pd.read_sas(path, format="xport", encoding="latin-1")
    df.columns = [c.upper() for c in df.columns]
    if columns is not None:
        cols = [c.upper() for c in columns]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{Path(path).name}: variable(s) not found: {missing}. "
                           f"Available: {list(df.columns)}")
        df = df[cols].copy()
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].mask(df[num].abs() < 1e-50, 0.0)
    if "SEQN" in df.columns:
        df["SEQN"] = df["SEQN"].astype("int64")
    return df


# --------------------------------------------------------------------------------------
# 5. Clinical derivations
# --------------------------------------------------------------------------------------
def qtc_all(qt_ms: pd.Series, hr_bpm: pd.Series) -> pd.DataFrame:
    """QT correction formulas (ms). RR in seconds."""
    rr = 60.0 / hr_bpm
    out = pd.DataFrame(index=qt_ms.index)
    out["QTc_Bazett"] = qt_ms / np.sqrt(rr)
    out["QTc_Fridericia"] = qt_ms / np.cbrt(rr)
    out["QTc_Framingham"] = qt_ms + 154.0 * (1.0 - rr)
    out["QTc_Hodges"] = qt_ms + 1.75 * (hr_bpm - 60.0)
    return out


def egfr_ckdepi2021(scr_mgdl: pd.Series, age: pd.Series, female: pd.Series) -> pd.Series:
    """CKD-EPI 2021 creatinine equation (race-free). scr in mg/dL."""
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = scr_mgdl / kappa
    egfr = 142.0 * np.minimum(ratio, 1.0) ** alpha * np.maximum(ratio, 1.0) ** (-1.200) \
        * 0.9938 ** age * np.where(female, 1.012, 1.0)
    return pd.Series(egfr, index=scr_mgdl.index)


# --------------------------------------------------------------------------------------
# 6. Survey-weighted descriptive helpers
# --------------------------------------------------------------------------------------
def wmean(x: pd.Series, w: pd.Series) -> float:
    m = x.notna() & w.notna()
    return float(np.average(x[m], weights=w[m])) if m.any() else np.nan


def wsd(x: pd.Series, w: pd.Series) -> float:
    m = x.notna() & w.notna()
    if not m.any():
        return np.nan
    mu = np.average(x[m], weights=w[m])
    return float(np.sqrt(np.average((x[m] - mu) ** 2, weights=w[m])))


def wprop(x: pd.Series, w: pd.Series, value=1) -> float:
    m = x.notna() & w.notna()
    return float(np.average((x[m] == value).astype(float), weights=w[m])) if m.any() else np.nan


def table1(df: pd.DataFrame, by: str, w: str, continuous: Sequence[str], binary: Sequence[str],
           categorical: Sequence[str] = ()) -> pd.DataFrame:
    """Survey-weighted Table 1: weighted mean (SD) / weighted % by group, plus unweighted n."""
    groups = [g for g in sorted(df[by].dropna().unique())]
    rows = []
    for v in continuous:
        row = {"variable": v, "type": "mean (SD)"}
        for g in groups:
            sub = df[df[by] == g]
            row[str(g)] = f"{wmean(sub[v], sub[w]):.1f} ({wsd(sub[v], sub[w]):.1f})"
        rows.append(row)
    for v in binary:
        row = {"variable": v, "type": "weighted %"}
        for g in groups:
            sub = df[df[by] == g]
            row[str(g)] = f"{100*wprop(sub[v], sub[w]):.1f}"
        rows.append(row)
    for v in categorical:
        for lev in sorted(df[v].dropna().unique()):
            row = {"variable": f"{v} = {lev}", "type": "weighted %"}
            for g in groups:
                sub = df[df[by] == g]
                row[str(g)] = f"{100*wprop(sub[v], sub[w], lev):.1f}"
            rows.append(row)
    n_row = {"variable": "n (unweighted)", "type": ""}
    for g in groups:
        n_row[str(g)] = str(int((df[by] == g).sum()))
    return pd.DataFrame([n_row] + rows)


# --------------------------------------------------------------------------------------
# 7. Survey-weighted Cox regression with design-based (Taylor-linearisation) variance
# --------------------------------------------------------------------------------------
def design_variance(infl: np.ndarray, strata: np.ndarray, psu: np.ndarray,
                    design_psus: Optional[pd.DataFrame] = None):
    """Taylor-linearisation (with-replacement, ultimate-cluster) variance from influence values.

    infl        n x p matrix of per-observation influence values (e.g. delta-betas).
    strata, psu design stratum and PSU label of each observation.
    design_psus optional DataFrame with columns [stratum, psu] listing every PSU of the full
                design; PSUs with no observations in this (sub)sample then enter as zero totals,
                which is the correct domain (subpopulation) variance.
    Strata with a single PSU are centred on the grand mean of PSU totals (R survey 'adjust').
    Returns (p x p variance matrix, number of single-PSU strata).
    """
    infl = np.asarray(infl, dtype=float)
    if infl.ndim == 1:
        infl = infl[:, None]
    tot = pd.DataFrame(infl).groupby([np.asarray(strata), np.asarray(psu)]).sum()
    if design_psus is not None:
        full = pd.MultiIndex.from_frame(design_psus.iloc[:, :2].drop_duplicates())
        tot = tot.reindex(tot.index.union(full), fill_value=0.0)
    grand = tot.values.mean(axis=0)
    V = np.zeros((infl.shape[1], infl.shape[1]))
    lonely = 0
    for _, g in tot.groupby(level=0):
        vals = g.values
        nh = vals.shape[0]
        if nh == 1:
            dev = vals - grand
            V += dev.T @ dev
            lonely += 1
        else:
            dev = vals - vals.mean(axis=0)
            V += nh / (nh - 1) * (dev.T @ dev)
    return V, lonely


def cox_weighted(df: pd.DataFrame, time: str, event: str, covariates: Sequence[str],
                 weight: str, cluster: Optional[str] = None, normalize_weights: bool = True,
                 strata_col: Optional[str] = None, design_psus: Optional[pd.DataFrame] = None):
    """Survey-weighted Cox model (lifelines point estimates, Efron ties).

    Variance:
      * strata_col and cluster given -> design-based Taylor-linearisation variance (strata + PSU),
        the estimator used by R survey::svycoxph (Binder 1992);
      * only cluster given           -> cluster-robust sandwich (PSU clusters, strata ignored);
      * neither                      -> robust sandwich by observation.
    Covariates are standardised internally (lifelines computes exact score residuals only on
    standardised data) and all results are back-transformed to the original scale.
    Weights are normalised to mean 1 (does not change estimates or design variance).

    Returns (fitted lifelines model on the standardised scale, results table). The table carries
    attrs: n, events, variance ('design' / 'cluster' / 'robust'), lonely_strata, and
    influence (DataFrame of per-observation influence values on the original scale, indexed like
    `df`) for stacked Wald tests across models.
    """
    from lifelines import CoxPHFitter
    from scipy import stats as _st

    cov = list(covariates)
    keep = [time, event, weight] + cov + [c for c in (cluster, strata_col) if c]
    d = df[keep].dropna().copy()
    if normalize_weights:
        d[weight] = d[weight] / d[weight].mean()
    mu, sd = d[cov].mean(), d[cov].std(ddof=1)
    constant = sd.index[(sd == 0) | sd.isna()].tolist()
    if constant:
        raise ValueError(f"covariate(s) constant in this sample: {constant}")
    z = d[[time, event, weight]].copy()
    z[cov] = (d[cov] - mu) / sd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph = CoxPHFitter().fit(z, duration_col=time, event_col=event, weights_col=weight, robust=False)
        hinv = cph.variance_matrix_.values
        u = cph.compute_residuals(z, kind="score").loc[z.index, cov].values
    infl_z = u @ hinv                                   # delta-betas, standardised scale

    if strata_col and cluster:
        vz, lonely = design_variance(infl_z, d[strata_col].values, d[cluster].values, design_psus)
        vtype = "design"
    elif cluster:
        tot = pd.DataFrame(infl_z).groupby(d[cluster].values).sum().values
        vz, lonely, vtype = tot.T @ tot, 0, "cluster"
    else:
        vz, lonely, vtype = infl_z.T @ infl_z, 0, "robust"

    scale = sd.values
    beta = cph.params_.values / scale
    V = vz / np.outer(scale, scale)
    se = np.sqrt(np.diag(V))
    zval = beta / se
    res = pd.DataFrame({"beta": beta, "HR": np.exp(beta),
                        "HR_lo95": np.exp(beta - 1.959964 * se), "HR_hi95": np.exp(beta + 1.959964 * se),
                        "p": 2 * _st.norm.sf(np.abs(zval))}, index=pd.Index(cov, name="covariate"))
    # flag non-identifiable estimates (e.g. (quasi-)separation from sparse exposure cells): never report them.
    # Criteria on the standardised scale: huge coefficient, huge model-based SE, or a design SE that has
    # collapsed relative to the model-based SE (influence values ~0 when a coefficient diverges).
    beta_z = cph.params_.values
    se_model_z = np.sqrt(np.clip(np.diag(hinv), 0, None))
    se_z = np.sqrt(np.clip(np.diag(vz), 0, None))
    unstable = ((np.abs(beta_z) > 10) | (se_model_z > 5) | ~np.isfinite(se) | (se <= 1e-10)
                | (se_z < 0.01 * se_model_z))
    res["unstable"] = unstable
    res.loc[unstable, ["HR", "HR_lo95", "HR_hi95", "p"]] = np.nan
    res.attrs.update({"n": int(len(d)), "events": int(d[event].sum()), "variance": vtype,
                      "lonely_strata": int(lonely), "vcov": pd.DataFrame(V, index=cov, columns=cov),
                      "influence": pd.DataFrame(infl_z / scale, index=d.index, columns=cov)})
    return cph, res


def design_wald(betas: Sequence[float], influences: Sequence[pd.Series], strata: pd.Series, psu: pd.Series,
                contrast: np.ndarray, design_psus: Optional[pd.DataFrame] = None):
    """Wald test of C @ beta = 0 for coefficients estimated in *different* models on overlapping
    samples (e.g. period-specific hazard ratios), using stacked influence values so that the
    between-model covariance is accounted for. `influences` are Series indexed by observation id;
    `strata` / `psu` are Series indexed by the same ids (covering the union of all samples)."""
    from scipy import stats as _st
    idx = strata.index
    M = pd.concat([s.reindex(idx).fillna(0.0) for s in influences], axis=1).values
    V, _ = design_variance(M, strata.values, psu.values, design_psus)
    b = np.asarray(betas, dtype=float)
    C = np.atleast_2d(contrast)
    cb = C @ b
    chi2 = float(cb @ np.linalg.solve(C @ V @ C.T, cb))
    df_ = C.shape[0]
    return {"chi2": chi2, "df": df_, "p": float(_st.chi2.sf(chi2, df_)), "vcov": V}


# --------------------------------------------------------------------------------------
# 8. Figure style (submission conventions)
# --------------------------------------------------------------------------------------
def set_figure_style():
    import matplotlib as mpl
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans") if f in available), "sans-serif")
    if family != "Arial":
        print(f"[figure style] Arial not installed on this machine; falling back to {family}")
    mpl.rcParams.update({
        "font.family": family, "font.size": 12,
        "axes.titlesize": 12, "axes.labelsize": 13, "axes.labelweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "legend.frameon": False,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel_letter(ax, letter: str):
    """Bold panel letter in the upper-left corner (fixed offset). Prefer align_panel_letters()."""
    ax.text(-0.18, 1.02, letter, transform=ax.transAxes, fontsize=16, fontweight="bold",
            va="bottom", ha="left")


def align_panel_letters(fig, axes_letters, y: float = 1.03, fontsize: int = 16):
    """Place bold panel letters so their left edge lines up with each panel's y-axis title
    (or, if a panel has no y-axis title, with the left edge of its y tick labels / spine).
    Call AFTER fig.tight_layout() / subplots_adjust(), because it measures rendered positions."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, letter in axes_letters:
        lab = ax.yaxis.label
        if lab.get_text():
            x0 = lab.get_window_extent(renderer).x0
        else:
            ticks = [t.label1.get_window_extent(renderer).x0 for t in ax.yaxis.get_major_ticks()
                     if t.label1.get_visible() and t.label1.get_text()]
            x0 = min(ticks) if ticks else ax.get_window_extent(renderer).x0
        x_axes = ax.transAxes.inverted().transform((x0, 0))[0]
        ax.text(x_axes, y, letter, transform=ax.transAxes, fontsize=fontsize, fontweight="bold",
                va="bottom", ha="left")


# --------------------------------------------------------------------------------------
# 9. Sensitivity to unmeasured confounding
# --------------------------------------------------------------------------------------
def evalue_hr(hr: float, lo: Optional[float] = None, hi: Optional[float] = None,
              common_outcome: bool = False):
    """E-value for a hazard ratio (VanderWeele & Ding, Ann Intern Med 2017;167:268-274).

    For outcomes that are common by the end of follow-up (> 15%), the HR is first converted
    to an approximate risk ratio, RR = (1 - 0.5**sqrt(HR)) / (1 - 0.5**sqrt(1/HR));
    for rare outcomes RR ~ HR. E-value = RR + sqrt(RR * (RR - 1)) (RR inverted if < 1).
    The CI E-value uses the limit closest to the null and is 1 if the CI includes 1.
    Returns (E-value for the estimate, E-value for the CI limit)."""
    def to_rr(h):
        return (1 - 0.5 ** np.sqrt(h)) / (1 - 0.5 ** np.sqrt(1 / h)) if common_outcome else h

    def ev(rr):
        rr = 1 / rr if rr < 1 else rr
        return float(rr + np.sqrt(rr * (rr - 1)))

    if hr is None or not np.isfinite(hr):
        return np.nan, np.nan
    est = ev(to_rr(hr))
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return est, np.nan
    if lo <= 1 <= hi:
        return est, 1.0
    return est, ev(to_rr(lo if hr > 1 else hi))


GREYS = ["#000000", "#555555", "#999999", "#BBBBBB"]


# --------------------------------------------------------------------------------------
# 10. Design-based delete-one-PSU jackknife (JKn) for arbitrary estimators
# --------------------------------------------------------------------------------------
def jackknife_replicates(strata, psu, design_psus: Optional[pd.DataFrame] = None):
    """Delete-one-PSU replicate weight factors.

    Replicate (h, j) multiplies the weights of PSU j in stratum h by 0 and those of the other PSUs
    of stratum h by n_h / (n_h - 1); all other weights are unchanged. `design_psus` (columns
    [stratum, psu]) lists every PSU of the full design so that PSUs with no observations in the
    analysed (sub)sample still define replicates (domain estimation). Strata with a single PSU
    cannot be jackknifed and are skipped (their number is returned).
    Returns (list of dicts with keys stratum, psu, nh, factor), number of single-PSU strata)."""
    s, u = np.asarray(strata), np.asarray(psu)
    pairs = pd.DataFrame({"s": s, "u": u}).drop_duplicates()
    if design_psus is not None:
        dp = design_psus.iloc[:, :2].drop_duplicates().copy()
        dp.columns = ["s", "u"]
        pairs = pd.concat([pairs, dp], ignore_index=True).drop_duplicates()
    reps, lonely = [], 0
    for h, g in pairs.groupby("s", sort=True):
        nh = len(g)
        if nh < 2:
            lonely += 1
            continue
        in_h = s == h
        for j in g["u"]:
            f = np.ones(len(s))
            f[in_h] = nh / (nh - 1.0)
            f[in_h & (u == j)] = 0.0
            reps.append({"stratum": h, "psu": j, "nh": nh, "factor": f})
    return reps, lonely


def survey_jackknife(estimator, df: pd.DataFrame, weight: str, strata_col: str, psu_col: str,
                     design_psus: Optional[pd.DataFrame] = None, verbose: bool = True,
                     progress_every: int = 10):
    """Design-based standard errors for any estimator by the delete-one-PSU jackknife (JKn).

    estimator(frame) -> pd.Series (or dict) of estimands, computed with the weights in `weight`.
    Variance = sum_h (n_h - 1)/n_h * sum_j (theta_hj - theta_full)^2, centred at the full-sample
    estimate (the slightly conservative 'MSE' form). An estimand whose estimator fails in any
    replicate gets SE = NaN (never silently dropped).
    Returns (DataFrame [estimate, se, lo95, hi95] indexed by estimand, replicate matrix)."""
    import time as _time
    full = pd.Series(estimator(df), dtype=float)
    reps, lonely = jackknife_replicates(df[strata_col].values, df[psu_col].values, design_psus)
    rows, fails = [], 0
    t0 = _time.time()
    for k, r in enumerate(reps, 1):
        dd = df.assign(**{weight: df[weight].values * r["factor"]})
        dd = dd[dd[weight] > 0]
        try:
            est = pd.Series(estimator(dd), dtype=float).reindex(full.index)
        except Exception as e:                                   # recorded, never hidden
            fails += 1
            if verbose:
                print(f"  replicate {k} (stratum {r['stratum']}, PSU {r['psu']}) failed: {str(e)[:120]}")
            est = pd.Series(np.nan, index=full.index)
        rows.append(est.values)
        if verbose and (k % progress_every == 0 or k == len(reps)):
            el = _time.time() - t0
            print(f"  jackknife replicate {k}/{len(reps)}  ({el:.0f} s elapsed, ~{el / k * (len(reps) - k):.0f} s left)")
    R = pd.DataFrame(rows, columns=full.index)
    coef = np.array([(r["nh"] - 1.0) / r["nh"] for r in reps])
    dev = R.values - full.values[None, :]
    se = np.sqrt(np.sum(coef[:, None] * dev ** 2, axis=0))
    out = pd.DataFrame({"estimate": full.values, "se": se,
                        "lo95": full.values - 1.959964 * se, "hi95": full.values + 1.959964 * se},
                       index=full.index)
    out.attrs.update({"n_replicates": len(reps), "failed_replicates": fails, "lonely_strata": lonely})
    if verbose:
        print(f"[jackknife] {len(reps)} replicates, {fails} failed, {lonely} single-PSU strata skipped")
    return out, R


# --------------------------------------------------------------------------------------
# 11. Standardised (g-formula) absolute risks, restricted mean times, discrimination
# --------------------------------------------------------------------------------------
def cox_beta(d: pd.DataFrame, time: str, event: str, covariates: Sequence[str], weight: str,
             strata: Optional[Sequence[str]] = None) -> pd.Series:
    """Weighted Cox coefficients (lifelines, Efron ties) on the original covariate scale.
    Covariates are standardised for the fit only; `strata` columns get separate baseline hazards."""
    from lifelines import CoxPHFitter
    cov = list(covariates)
    mu, sd = d[cov].mean(), d[cov].std(ddof=1)
    constant = sd.index[(sd == 0) | sd.isna()].tolist()
    if constant:
        raise ValueError(f"covariate(s) constant in this sample: {constant}")
    z = d[[time, event]].copy()
    z["_w"] = d[weight] / d[weight].mean()
    z[cov] = (d[cov] - mu) / sd
    for s_ in (strata or []):
        z[s_] = d[s_].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph = CoxPHFitter().fit(z, duration_col=time, event_col=event, weights_col="_w",
                                strata=list(strata) if strata else None, robust=False)
    return pd.Series(cph.params_.reindex(cov).values / sd.values, index=cov)


def breslow_cumhaz(time, event, weight, lp, grid) -> np.ndarray:
    """Weighted Breslow cumulative baseline hazard (at linear predictor 0), evaluated at `grid`
    as a right-continuous step function: H0(t) = sum_{s <= t} dN_w(s) / sum_{T_k >= s} w_k exp(lp_k)."""
    t = np.asarray(time, float)
    e = np.asarray(event, float) > 0
    w = np.asarray(weight, float)
    r = w * np.exp(np.asarray(lp, float))
    grid = np.asarray(grid, float)
    if not e.any():
        return np.zeros(len(grid))
    ev_times = np.unique(t[e])
    order = np.argsort(t, kind="mergesort")
    ts, rs = t[order], r[order]
    at_risk = np.cumsum(rs[::-1])[::-1]                      # sum_{k >= index} r_k (sorted by time)
    risk = at_risk[np.searchsorted(ts, ev_times, side="left")]
    dN = pd.Series(w[e]).groupby(t[e]).sum().reindex(ev_times).values
    H = np.cumsum(dN / risk)
    return np.concatenate([[0.0], H])[np.searchsorted(ev_times, grid, side="right")]


def gformula_curves(d: pd.DataFrame, time: str, death_all: str, death_cause: Optional[str], exposure: str,
                    covariates: Sequence[str], weight: str, grid) -> pd.DataFrame:
    """Covariate-standardised curves under exposure = 0 and = 1 (g-formula, the analysed sample as
    the standard population, survey weights).

    All-cause: Cox model stratified by exposure (so the exposure effect may vary freely over time)
    with common covariate effects -> S_a(t) = sum_i w_i exp(-H0_a(t) exp(lp_i)) / sum_i w_i.
    Cause-specific (if `death_cause` is given): cause-specific Cox models for the cause and for
    death from other causes (competing event), each stratified by exposure ->
    CIF_a(t) = mean_i sum_{s <= t} S_ia(s-) dH_cause,ia(s), with S_ia = exp(-H_cause,ia - H_other,ia).
    `grid` must contain every event time up to the last time of interest (step functions are then
    exact). Returns DataFrame indexed by grid: S0, S1 (and CIF0, CIF1 when a cause is given)."""
    cov = list(covariates)
    X = d[cov].values.astype(float)
    w = d[weight].values.astype(float)
    t = d[time].values.astype(float)
    a_ = d[exposure].values
    Xc = X - np.average(X, axis=0, weights=w)
    grid = np.asarray(grid, float)
    out = {}
    b_all = cox_beta(d, time, death_all, cov, weight, strata=[exposure])
    lp = Xc @ b_all.values
    ev = d[death_all].values
    for a in (0, 1):
        m = a_ == a
        H0 = breslow_cumhaz(t[m], ev[m], w[m], lp[m], grid)
        S = np.exp(-np.outer(np.exp(lp), H0))
        out[f"S{a}"] = (w[:, None] * S).sum(axis=0) / w.sum()
    if death_cause is not None:
        ev_c = d[death_cause].values.astype(int)
        ev_o = ((ev == 1) & (ev_c == 0)).astype(int)
        d2 = d.assign(_ev_other=ev_o)
        b_c = cox_beta(d2, time, death_cause, cov, weight, strata=[exposure])
        b_o = cox_beta(d2, time, "_ev_other", cov, weight, strata=[exposure])
        lp_c, lp_o = Xc @ b_c.values, Xc @ b_o.values
        n = len(w)
        for a in (0, 1):
            m = a_ == a
            Hc = np.outer(np.exp(lp_c), breslow_cumhaz(t[m], ev_c[m], w[m], lp_c[m], grid))
            Ho = np.outer(np.exp(lp_o), breslow_cumhaz(t[m], ev_o[m], w[m], lp_o[m], grid))
            S = np.exp(-(Hc + Ho))
            S_prev = np.hstack([np.ones((n, 1)), S[:, :-1]])
            dHc = np.diff(np.hstack([np.zeros((n, 1)), Hc]), axis=1)
            out[f"CIF{a}"] = (w[:, None] * np.cumsum(S_prev * dHc, axis=1)).sum(axis=0) / w.sum()
    return pd.DataFrame(out, index=pd.Index(grid, name="t"))


def step_value(grid, values, at, start_value: float):
    """Value of a right-continuous step function (values[k] on [grid[k], grid[k+1])) at time(s) `at`."""
    grid, values = np.asarray(grid, float), np.asarray(values, float)
    pos = np.searchsorted(grid, np.atleast_1d(np.asarray(at, float)), side="right")
    return np.concatenate([[start_value], values])[pos]


def restricted_mean(grid, values, tau: float, start_value: float) -> float:
    """Integral over [0, tau] of a right-continuous step function equal to `start_value` on
    [0, grid[0]) and values[k] on [grid[k], grid[k+1]). With survival -> RMST; with cumulative
    incidence (start 0) -> restricted mean time lost (RMTL)."""
    g, v = np.asarray(grid, float), np.asarray(values, float)
    keep = g < tau
    knots = np.concatenate([[0.0], g[keep], [tau]])
    vals = np.concatenate([[start_value], v[keep]])
    return float(np.sum(vals * np.diff(knots)))


def harrell_c(time, event, risk, weight=None, tau: Optional[float] = None) -> float:
    """(Survey-)weighted Harrell's C for a risk score (higher = worse prognosis).

    Usable pairs: the subject with the shorter observed time had the event (subjects censored at
    that same time count as surviving longer); pairs with tied event times are not used. Pair
    weight = w_i * w_j; ties in the risk score count 1/2. With `tau`, follow-up is truncated at tau
    (events after tau are treated as censored at tau). O(n log n) via a Fenwick tree."""
    t = np.asarray(time, float).copy()
    e = np.asarray(event).astype(bool).copy()
    r = np.asarray(risk, float)
    w = np.ones(len(t)) if weight is None else np.asarray(weight, float)
    if tau is not None:
        e &= t <= tau
        t = np.minimum(t, tau)
    _, rank = np.unique(r, return_inverse=True)
    K = int(rank.max()) + 1 if len(rank) else 0
    tree = np.zeros(K + 1)

    def add(i, val):
        i += 1
        while i <= K:
            tree[i] += val
            i += i & (-i)

    def prefix(i):                                  # total weight of ranks 0 .. i-1
        s = 0.0
        while i > 0:
            s += tree[i]
            i -= i & (-i)
        return s

    order = np.argsort(-t, kind="mergesort")
    ts = t[order]
    num = den = total = 0.0
    i, n = 0, len(t)
    while i < n:
        j = i
        while j < n and ts[j] == ts[i]:
            j += 1
        idx = order[i:j]
        for k in idx[~e[idx]]:                      # censored at this time: still at risk for events at this time
            add(rank[k], w[k])
            total += w[k]
        evs = idx[e[idx]]
        for k in evs:
            below = prefix(rank[k])
            equal = prefix(rank[k] + 1) - below
            num += w[k] * (below + 0.5 * equal)
            den += w[k] * total
        for k in evs:
            add(rank[k], w[k])
            total += w[k]
        i = j
    return num / den if den > 0 else np.nan


def pool_fixed_effect(estimates, ses) -> Dict[str, float]:
    """Inverse-variance fixed-effect pooling (e.g. log hazard ratios from two cohorts) with
    Cochran's Q test and I^2 for between-cohort heterogeneity."""
    from scipy import stats as _st
    b, s = np.asarray(estimates, float), np.asarray(ses, float)
    wt = 1.0 / s ** 2
    bp = float(np.sum(wt * b) / np.sum(wt))
    sp = float(np.sqrt(1.0 / np.sum(wt)))
    q = float(np.sum(wt * (b - bp) ** 2))
    k = len(b)
    return {"estimate": bp, "se": sp, "lo95": bp - 1.959964 * sp, "hi95": bp + 1.959964 * sp,
            "z_p": float(2 * _st.norm.sf(abs(bp / sp))), "Q": q, "Q_df": k - 1,
            "Q_p": float(_st.chi2.sf(q, k - 1)) if k > 1 else np.nan,
            "I2": float(max(0.0, (q - (k - 1)) / q)) if q > 0 else 0.0}


def se_from_ci(lo: float, hi: float, log: bool = True) -> float:
    """Standard error recovered from a symmetric 95% CI (on the log scale for ratios)."""
    return float((np.log(hi) - np.log(lo)) / (2 * 1.959964)) if log else float((hi - lo) / (2 * 1.959964))


def absolute_risk_suite(d: pd.DataFrame, time: str, death_all: str, death_cause: str, exposure: str,
                        covariates: Sequence[str], weight: str, horizons: Sequence[float] = (5, 10),
                        taus: Sequence[float] = (10,), c_horizon: Optional[float] = 10.0,
                        curve_step: float = 0.25, curve_max: Optional[float] = None) -> pd.Series:
    """Every absolute-scale estimand for one set of weights (full sample or one jackknife replicate).

    Returned as a flat Series (risks in %, restricted mean times in months):
      all_F{a}_{h}, all_RD_{h}           standardised cumulative all-cause mortality and difference
      all_RMST{a}_{tau}, all_dRMST_{tau} restricted mean survival time and difference (a=1 minus a=0)
      cause_CIF{a}_{h}, cause_RD_{h}     standardised cumulative incidence of cause-specific death
      cause_RMTL{a}_{tau}, cause_dRMTL_{tau}  restricted mean time lost to the cause and difference
      C_all_base / C_all_plus / dC_all   weighted Harrell's C at c_horizon without / with exposure
      C_cause_base / ... / dC_cause      same for cause-specific death (other deaths censored)
      curve_all_RD_{t}, curve_cause_RD_{t}  risk differences on a regular grid (for figures)
      curve_all_F{a}_{t}, curve_cause_CIF{a}_{t}
    The analysed sample must already be restricted to complete cases for `covariates`."""
    cov = list(covariates)
    t_last = max(list(horizons) + list(taus) + ([curve_max] if curve_max else []))
    ev_times = np.unique(d.loc[d[death_all] == 1, time].values)
    extra = np.arange(curve_step, t_last + 1e-9, curve_step)
    grid = np.unique(np.concatenate([ev_times[ev_times <= t_last], np.asarray(horizons, float),
                                     np.asarray(taus, float), extra]))
    cur = gformula_curves(d, time, death_all, death_cause, exposure, cov, weight, grid)
    g = cur.index.values
    out = {}
    for h in horizons:
        f0, f1 = (1 - step_value(g, cur[f"S{a}"].values, h, 1.0)[0] for a in (0, 1))
        c0, c1 = (step_value(g, cur[f"CIF{a}"].values, h, 0.0)[0] for a in (0, 1))
        out.update({f"all_F0_{h:g}": 100 * f0, f"all_F1_{h:g}": 100 * f1, f"all_RD_{h:g}": 100 * (f1 - f0),
                    f"cause_CIF0_{h:g}": 100 * c0, f"cause_CIF1_{h:g}": 100 * c1, f"cause_RD_{h:g}": 100 * (c1 - c0)})
    for tau in taus:
        r0, r1 = (12 * restricted_mean(g, cur[f"S{a}"].values, tau, 1.0) for a in (0, 1))
        l0, l1 = (12 * restricted_mean(g, cur[f"CIF{a}"].values, tau, 0.0) for a in (0, 1))
        out.update({f"all_RMST0_{tau:g}": r0, f"all_RMST1_{tau:g}": r1, f"all_dRMST_{tau:g}": r1 - r0,
                    f"cause_RMTL0_{tau:g}": l0, f"cause_RMTL1_{tau:g}": l1, f"cause_dRMTL_{tau:g}": l1 - l0})
    pts = np.round(np.arange(0.0, t_last + 1e-9, curve_step), 4)          # rounded values name the estimands
    # v9.1: evaluate the right-continuous curves at the exact multiples of curve_step (plus 1e-9), so that a jump
    # at exactly that time is included. v9 evaluated at the rounded values, which fall just before k/12 for one in
    # three monthly points and returned the previous month's value there (horizon estimands, RMST/RMTL and C
    # were not affected).
    pts_eval = np.arange(len(pts)) * curve_step + 1e-9
    S0, S1 = (step_value(g, cur[f"S{a}"].values, pts_eval, 1.0) for a in (0, 1))
    C0, C1 = (step_value(g, cur[f"CIF{a}"].values, pts_eval, 0.0) for a in (0, 1))
    for k, p_ in enumerate(pts):
        out.update({f"curve_all_F0_{p_:g}": 100 * (1 - S0[k]), f"curve_all_F1_{p_:g}": 100 * (1 - S1[k]),
                    f"curve_all_RD_{p_:g}": 100 * (S0[k] - S1[k]),
                    f"curve_cause_CIF0_{p_:g}": 100 * C0[k], f"curve_cause_CIF1_{p_:g}": 100 * C1[k],
                    f"curve_cause_RD_{p_:g}": 100 * (C1[k] - C0[k])})
    if c_horizon:
        tc = np.minimum(d[time].values, c_horizon)
        dc = d.assign(_t=tc,
                      _e_all=((d[death_all].values == 1) & (d[time].values <= c_horizon)).astype(int),
                      _e_cause=((d[death_cause].values == 1) & (d[time].values <= c_horizon)).astype(int))
        X = dc[cov].values.astype(float)
        for lab, evc in (("all", "_e_all"), ("cause", "_e_cause")):
            b0 = cox_beta(dc, "_t", evc, cov, weight)
            b1 = cox_beta(dc, "_t", evc, cov + [exposure], weight)
            r0 = X @ b0.values
            r1 = X @ b1.values[:-1] + dc[exposure].values * b1.values[-1]
            c_0 = harrell_c(tc, dc[evc].values, r0, dc[weight].values)
            c_1 = harrell_c(tc, dc[evc].values, r1, dc[weight].values)
            out.update({f"C_{lab}_base": c_0, f"C_{lab}_plus": c_1, f"dC_{lab}": c_1 - c_0})
    return pd.Series(out, dtype=float)
