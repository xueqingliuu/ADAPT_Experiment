# %%
from __future__ import annotations

import pandas as pd
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# %%
# ---------- CONFIG ----------
BASE_DIR = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPT_MRT/")   # e.g. ".../exports"
ROOT_DIR = BASE_DIR / "rawdata"
OUT_DIR = BASE_DIR / "Xueqing"   # where combined JSON/CSV will go

DATE_START = "2024-11-01"
DATE_END = "2026-07-03"
FOLDER_FMT = "%Y-%m-%d"

# Combine only files that end with _YYYYMMDD.json or _YYYYMMDD-YYYYMMDD.json
DATE_SUFFIX_RE = re.compile(r"_(\d{8}(?:-\d{8})?)\.json$", re.IGNORECASE)

# Non-dated JSONs that appear inside each date folder and should be combined
FIXED_NAME_DATASETS = {
    "ProjectDeviceData_cleaned.json",
    "filtered_activities-steps.json",
    # "filtered_hrv.json",
    "filtered_activities-heart.json",
    # add more if needed, e.g. "Manifest.json"
}

SKIP_DATASETS = {"FitbitIntradayCombined", "FitbitRestingHeartRates", "DeletedSurveyResults"}  # add more if needed

# Which keys to look for when a JSON file is a dict that contains a list of records
LIST_KEYS_CANDIDATES = ("data", "events", "records", "items", "rows", "results")


# ---------- HELPERS ----------
def base_dataset_name(filename: str) -> str | None:
    """
    Turns:
      AnalyticsEvents_NotificationSent_20250912-20250913.json -> AnalyticsEvents_NotificationSent
      FitbitDevices_20250913.json -> FitbitDevices
    Returns None if it doesn't match the dated pattern.
    """
    m = DATE_SUFFIX_RE.search(filename)
    if not m:
        return None
    return filename[: m.start()]



def iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    """
    Read either:
    - standard JSON array/object, or
    - JSONL (one JSON object per line).
    Skips empty/malformed files with warning.
    """
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            # find first non-whitespace character
            first = ""
            while True:
                c = f.read(1)
                if c == "":
                    print(f"[skip empty] {path}")
                    return
                if not c.isspace():
                    first = c
                    break

            f.seek(0)

            # Try standard JSON first if file starts with { or [
            if first in "{[":
                try:
                    obj = json.load(f)
                    if isinstance(obj, list):
                        for r in obj:
                            if isinstance(r, dict):
                                yield r
                        return
                    if isinstance(obj, dict):
                        # Common export shape: {"data":[...]} or similar list container.
                        for key in LIST_KEYS_CANDIDATES:
                            maybe_rows = obj.get(key)
                            if isinstance(maybe_rows, list):
                                for r in maybe_rows:
                                    if isinstance(r, dict):
                                        yield r
                                return
                        yield obj
                        return
                except json.JSONDecodeError:
                    # fall through to JSONL parsing
                    f.seek(0)

            # JSONL fallback
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                    if isinstance(rec, dict):
                        yield rec
                except json.JSONDecodeError:
                    # skip bad line instead of crashing whole run
                    continue

    except Exception as e:
        print(f"[skip unreadable] {path} ({e})")
        return
    

def add_provenance(records: list[dict[str, Any]], src: Path) -> list[dict[str, Any]]:
    """
    Add source metadata so you can trace rows back later.
    """
    folder = src.parent.name
    fname = src.name
    for r in records:
        r["_source_folder"] = folder
        r["_source_file"] = fname
    return records

def flatten_record(x: Any, parent: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(x, dict):
        for k, v in x.items():
            key = f"{parent}{sep}{k}" if parent else str(k)
            out.update(flatten_record(v, key, sep=sep))
        return out
    if isinstance(x, list):
        out[parent] = json.dumps(x, ensure_ascii=False)
        return out
    out[parent] = x
    return out


import csv

def combine_dataset_streaming(dataset: str, paths: list[Path]) -> None:
    out_json = OUT_DIR / f"{dataset}.json"
    out_csv  = OUT_DIR / f"{dataset}.csv"

    # Always overwrite (and avoid any accidental append artifacts)
    if out_json.exists():
        out_json.unlink()
    if out_csv.exists():
        out_csv.unlink()

    # PASS 1: collect CSV columns (without keeping all rows)
    columns: set[str] = set()
    total = 0
    for p in paths:
        for rec in iter_json_records(p):
            recs = add_provenance([rec], p)  # re-use your provenance logic
            flat = flatten_record(recs[0], sep=".")
            columns.update(flat.keys())
            total += 1

    if total == 0:
        out_json.write_text("[]\n", encoding="utf-8")
        out_csv.write_text("", encoding="utf-8")
        print(f"[SKIP] {dataset}: 0 records")
        return

    header = sorted(columns)

    # PASS 2: stream-write JSON array + CSV
    with out_json.open("w", encoding="utf-8") as jf, out_csv.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=header, extrasaction="ignore")
        writer.writeheader()

        jf.write("[\n")
        first = True

        written = 0
        for p in paths:
            for rec in iter_json_records(p):
                rec = add_provenance([rec], p)[0]

                # JSON array item (comma-separated)
                if not first:
                    jf.write(",\n")
                jf.write(json.dumps(rec, ensure_ascii=False))
                first = False

                # CSV row
                flat = flatten_record(rec, sep=".")
                writer.writerow({k: flat.get(k, "") for k in header})

                written += 1

        jf.write("\n]\n")

    print(f"[OK] {dataset}: {written} records -> {out_json.name}, {out_csv.name}")

# %%
PATTERN = "FitbitIntradayCombined_*.json"


def iter_jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    # FitbitIntradayCombined rows look like JSONL: one JSON object per line
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line_no, ln in enumerate(f, start=1):
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError as e:
                print(f"[skip bad line] {path}:{line_no} ({e})")
                continue
            yield obj if isinstance(obj, dict) else {"value": obj}


def find_first_intraday_file() -> Path | None:
    for d in pd.date_range(DATE_START, DATE_END, freq="D"):
        folder = ROOT_DIR / d.strftime(FOLDER_FMT)
        if not folder.exists():
            continue
        files = sorted(folder.glob(PATTERN))
        if files:
            return files[0]
    return None


def list_types_from_one_file() -> set[str]:
    p = find_first_intraday_file()
    if p is None:
        raise FileNotFoundError(f"No files matching {PATTERN} found in date range.")
    types = set()
    for rec in iter_jsonl_records(p):
        t = rec.get("Type")
        if t is not None:
            types.add(str(t))
    print(f"Using sample file: {p}")
    print("Distinct Types:")
    for t in sorted(types):
        print("  -", t)
    return types

# --- Run ---
# types = list_types_from_one_file()

# %%
TARGET_TYPES = {
    "activities-steps": "filtered_activities-steps.json",
    # "hrv": "filtered_hrv.json",
    "activities-heart": "filtered_activities-heart.json",
}

# Raw Fitbit input files (must match the earlier cell, or this cell can run alone)
PATTERN = "FitbitIntradayCombined_*.json"


def _value_dict(rec: dict) -> dict:
    v = rec.get("Value")
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip().startswith("{"):
        try:
            o = json.loads(v)
            return o if isinstance(o, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _activities_heart_datetimes(rec: dict) -> tuple[Any, Any]:
    """Return raw DateTime and InsertedDate from top-level or nested Value."""
    vd = _value_dict(rec)
    dt = rec.get("DateTime") or rec.get("dateTime") or vd.get("DateTime") or vd.get("dateTime")
    ins = rec.get("InsertedDate") or vd.get("InsertedDate")
    return dt, ins


def _to_minute_iso(val: Any) -> str | None:
    if val is None:
        return None

    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            pass
        else:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            return parsed.replace(second=0, microsecond=0).isoformat()

    ts = pd.to_datetime(val, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts.floor("min").isoformat()


def _heart_rate_value(rec: dict) -> Any:
    """Heart rate (bpm) from top-level Value — Fitbit intraday is usually a number."""
    v = rec.get("Value")
    if isinstance(v, dict):
        return v.get("value") or v.get("bpm") or v.get("Value")
    if v is None:
        return None
    num = pd.to_numeric(v, errors="coerce")
    if pd.notna(num):
        return int(num) if float(num).is_integer() else float(num)
    return v


def _slim_activities_heart_record(rec: dict) -> dict[str, Any] | None:
    """One raw row: minute-rounded times + HR (used to aggregate many seconds into one minute)."""
    dt_raw, ins_raw = _activities_heart_datetimes(rec)
    dt_m = _to_minute_iso(dt_raw)
    ins_m = _to_minute_iso(ins_raw)
    if dt_m is None:
        return None
    pid = rec.get("ParticipantIdentifier")
    hr = _heart_rate_value(rec)
    return {
        "ParticipantIdentifier": pid,
        "Value": hr,
        "DateTime": dt_m,
        "InsertedDate": ins_m,
    }


def extract_types_per_date_folder() -> None:
    date_range = pd.date_range(start=DATE_START, end=DATE_END, freq="D")
    output_names = set(TARGET_TYPES.values())

    for d in date_range:
        folder = ROOT_DIR / d.strftime(FOLDER_FMT)
        if not folder.exists():
            continue

        # Input pattern should target only original files
        input_files = sorted(
            p for p in folder.glob(PATTERN)
            if p.is_file() and p.name not in output_names and not p.name.startswith("_ONLY_")
        )
        if not input_files:
            continue

        # activities-heart: average Value per (ParticipantIdentifier, DateTime minute)
        heart_agg: dict[tuple[Any, str], dict[str, Any]] = {}

        out_handles: dict[str, Any] = {}
        try:
            for t, out_name in TARGET_TYPES.items():
                if t == "activities-heart":
                    continue
                out_handles[t] = (folder / out_name).open("w", encoding="utf-8")

            kept_counts = {t: 0 for t in TARGET_TYPES}

            for p in input_files:
                for rec in iter_jsonl_records(p):
                    t = rec.get("Type")
                    if t not in TARGET_TYPES:
                        continue
                    if t == "activities-heart":
                        slim = _slim_activities_heart_record(rec)
                        if slim is None:
                            continue
                        hr = slim.get("Value")
                        if hr is None:
                            continue
                        try:
                            hr_f = float(hr)
                        except (TypeError, ValueError):
                            continue
                        pid = slim.get("ParticipantIdentifier")
                        dt_m = slim["DateTime"]
                        ins_m = slim.get("InsertedDate")
                        key = (pid, dt_m)
                        if key not in heart_agg:
                            heart_agg[key] = {"sum": 0.0, "n": 0, "ins_max": None}
                        b = heart_agg[key]
                        b["sum"] += hr_f
                        b["n"] += 1
                        if ins_m is not None:
                            if b["ins_max"] is None or ins_m > b["ins_max"]:
                                b["ins_max"] = ins_m
                    else:
                        out_handles[t].write(json.dumps(rec, ensure_ascii=False) + "\n")
                        kept_counts[t] += 1

            heart_path = folder / TARGET_TYPES["activities-heart"]
            with heart_path.open("w", encoding="utf-8") as hf:
                for (pid, dt_m) in sorted(
                    heart_agg.keys(), key=lambda k: (str(k[0]) if k[0] is not None else "", k[1])
                ):
                    b = heart_agg[(pid, dt_m)]
                    n = b["n"]
                    avg = b["sum"] / n if n else 0.0
                    val_out = int(round(avg)) if abs(avg - round(avg)) < 1e-9 else round(avg, 2)
                    row = {
                        "ParticipantIdentifier": pid,
                        "Value": val_out,
                        "DateTime": dt_m,
                        "InsertedDate": b["ins_max"],
                    }
                    hf.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept_counts["activities-heart"] = len(heart_agg)

            print(f"[OK] {folder.name}")
            for t in TARGET_TYPES:
                print(f"  {t}: {kept_counts[t]} rows -> {TARGET_TYPES[t]}")

        finally:
            for h in out_handles.values():
                h.close()


# %%
# ---------- MAIN ----------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date_range = pd.date_range(start=DATE_START, end=DATE_END, freq="D")
    groups: dict[str, list[Path]] = {}

    for d in date_range:
        folder = ROOT_DIR / d.strftime(FOLDER_FMT)
        if not folder.exists():
            continue

        for p in folder.glob("*.json"):
            base = base_dataset_name(p.name)

            if base is None and p.name in FIXED_NAME_DATASETS:
                base = p.stem  # fixed-name dataset grouped across dates

            if base in SKIP_DATASETS:
                continue

            if base is None:
                continue

            groups.setdefault(base, []).append(p)

    if not groups:
        print("No JSON files found. Check ROOT_DIR and FOLDER_FMT.")
        return

    for dataset, paths in sorted(groups.items()):
        combine_dataset_streaming(dataset, sorted(paths))

# %%
# Extract data from ProjectDeviceData_clean
# timeZone, phase, participantidentifier, utcOffset, Value-timestamp


# 1) Read JSON
# - Use lines=True if file is NDJSON (one JSON object per line)
# - If this errors, try removing lines=True
# df = pd.read_json(OUT_DIR / "ProjectDeviceData_cleaned.json", lines=True)

# 2) try normal JSON first, then NDJSON, then manual line parsing
def load_json_flex(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8-sig").strip()  # handles BOM
    if not text:
        raise ValueError("File is empty.")
    # A) whole-file JSON (array/object)
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            return pd.DataFrame([obj])
    except json.JSONDecodeError:
        pass
    # B) pandas NDJSON
    try:
        return pd.read_json(path, lines=True)
    except Exception:
        pass
    # C) manual NDJSON parse (skip bad lines)
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            rec = json.loads(s)
            if isinstance(rec, dict):
                rows.append(rec)
        except json.JSONDecodeError:
            continue
    if not rows:
        raise ValueError("Could not parse JSON file as array/object or line-delimited JSON.")
    return pd.DataFrame(rows)
def parse_value(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return {}
    return {}


def _project_device_paths() -> list[Path]:
    paths: list[Path] = []
    for d in pd.date_range(start=DATE_START, end=DATE_END, freq="D"):
        p = ROOT_DIR / d.strftime(FOLDER_FMT) / "ProjectDeviceData_cleaned.json"
        if p.exists():
            paths.append(p)

    if paths:
        return paths

    combined_path = OUT_DIR / "ProjectDeviceData_cleaned.json"
    if combined_path.exists():
        return [combined_path]

    raise FileNotFoundError(
        f"Missing ProjectDeviceData_cleaned.json under {ROOT_DIR} and {combined_path}."
    )


def _to_date_iso(val: Any) -> str | None:
    if val is None:
        return None

    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            pass
        else:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            return parsed.date().isoformat()

    ts = pd.to_datetime(val, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts.date().isoformat()


def extract_project_device_fields() -> pd.DataFrame:
    seen: set[tuple[Any, Any, Any, Any, Any]] = set()
    source_count = 0

    for path in _project_device_paths():
        for r in iter_json_records(path):
            source_count += 1
            value_obj = parse_value(r.get("Value"))

            user = value_obj.get("user", {}) if isinstance(value_obj.get("user"), dict) else {}
            demographics = user.get("demographics", {}) if isinstance(user.get("demographics"), dict) else {}
            custom_fields = user.get("customFields", {}) if isinstance(user.get("customFields"), dict) else {}

            # timestamp under Value (fallback: Value.event.timestamp)
            timestamp = value_obj.get("timestamp")
            if timestamp is None and isinstance(value_obj.get("event"), dict):
                timestamp = value_obj["event"].get("timestamp")

            participantidentifier = r.get("ParticipantIdentifier")
            if pd.isna(participantidentifier) or participantidentifier is None:
                participantidentifier = user.get("participantIdentifier")

            seen.add((
                demographics.get("timeZone"),
                custom_fields.get("Phase"),
                participantidentifier,
                demographics.get("utcOffset"),
                _to_date_iso(timestamp),
            ))

    rows = sorted(seen, key=lambda row: tuple("" if v is None else str(v) for v in row))
    out = pd.DataFrame(rows, columns=[
        "timeZone", "phase", "participantidentifier", "utcOffset", "date"
    ])
    print(f"ProjectDeviceData source records: {source_count}")
    print(out.shape)

    out.to_csv(OUT_DIR / "ProjectDeviceData_selected_fields_combined.csv", index=False)
    return out


def run_pipeline() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extract_types_per_date_folder()
    main()
    extract_project_device_fields()


if __name__ == "__main__":
    run_pipeline()
