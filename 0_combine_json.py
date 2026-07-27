# %%
from __future__ import annotations

import csv
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# %%
# ---------- CONFIG ----------
BASE_DIR = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPT_MRT/")   # e.g. ".../exports"
ROOT_DIR = BASE_DIR / "rawdata"
OUT_DIR = BASE_DIR / "Xueqing"   # where combined JSON/CSV will go

DATE_START = "2024-11-01"
DATE_END = "2026-07-26"
FOLDER_FMT = "%Y-%m-%d"

# Parallelism for per-day Fitbit extraction (I/O-heavy; keep modest on Dropbox)
EXTRACT_WORKERS = max(1, min(6, (os.cpu_count() or 2) - 1))

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

_IO_BUFFER = 1024 * 1024


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


def _first_non_ws_char(f) -> str:
    """Read ahead in a buffered chunk instead of one byte at a time."""
    while True:
        chunk = f.read(4096)
        if not chunk:
            return ""
        for c in chunk:
            if not c.isspace():
                return c


def iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    """
    Read either:
    - standard JSON array/object, or
    - JSONL (one JSON object per line).
    Skips empty/malformed files with warning.
    """
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", buffering=_IO_BUFFER) as f:
            first = _first_non_ws_char(f)
            if not first:
                print(f"[skip empty] {path}")
                return

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
                if not ln or ln.isspace():
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


def add_provenance(rec: dict[str, Any], folder: str, fname: str) -> dict[str, Any]:
    """Add source metadata so you can trace rows back later (mutates and returns rec)."""
    rec["_source_folder"] = folder
    rec["_source_file"] = fname
    return rec


def flatten_record(x: Any, parent: str = "", sep: str = ".", out: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flatten nested dicts into dotted keys; lists become JSON strings."""
    if out is None:
        out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            key = f"{parent}{sep}{k}" if parent else str(k)
            flatten_record(v, key, sep=sep, out=out)
        return out
    if isinstance(x, list):
        out[parent] = json.dumps(x, ensure_ascii=False)
        return out
    out[parent] = x
    return out


def combine_dataset_streaming(dataset: str, paths: list[Path]) -> None:
    """Single-pass combine: write JSON while collecting flats, then write CSV."""
    out_json = OUT_DIR / f"{dataset}.json"
    out_csv = OUT_DIR / f"{dataset}.csv"

    # Always overwrite (and avoid any accidental append artifacts)
    if out_json.exists():
        out_json.unlink()
    if out_csv.exists():
        out_csv.unlink()

    columns: set[str] = set()
    flats: list[dict[str, Any]] = []
    written = 0

    with out_json.open("w", encoding="utf-8", buffering=_IO_BUFFER) as jf:
        jf.write("[\n")
        first = True

        for p in paths:
            folder = p.parent.name
            fname = p.name
            for rec in iter_json_records(p):
                add_provenance(rec, folder, fname)

                if not first:
                    jf.write(",\n")
                jf.write(json.dumps(rec, ensure_ascii=False))
                first = False

                flat = flatten_record(rec, sep=".")
                columns.update(flat.keys())
                flats.append(flat)
                written += 1

        jf.write("\n]\n")

    if written == 0:
        out_json.write_text("[]\n", encoding="utf-8")
        out_csv.write_text("", encoding="utf-8")
        print(f"[SKIP] {dataset}: 0 records")
        return

    header = sorted(columns)
    with out_csv.open("w", encoding="utf-8", newline="", buffering=_IO_BUFFER) as cf:
        writer = csv.DictWriter(cf, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for flat in flats:
            writer.writerow({k: flat.get(k, "") for k in header})

    print(f"[OK] {dataset}: {written} records -> {out_json.name}, {out_csv.name}")


# %%
PATTERN = "FitbitIntradayCombined_*.json"


def iter_jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    # FitbitIntradayCombined rows look like JSONL: one JSON object per line
    with path.open("r", encoding="utf-8-sig", errors="replace", buffering=_IO_BUFFER) as f:
        for line_no, ln in enumerate(f, start=1):
            if not ln or ln.isspace():
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

# Substrings used to skip json.loads for irrelevant JSONL lines (huge win on multi-GB files)
_TARGET_TYPE_NEEDLES = tuple(TARGET_TYPES.keys())


def _value_dict(rec: dict) -> dict:
    v = rec.get("Value")
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        s = v.lstrip()
        if s.startswith("{"):
            try:
                o = json.loads(v)
                return o if isinstance(o, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def _activities_heart_datetimes(rec: dict) -> tuple[Any, Any]:
    """Return raw DateTime and InsertedDate from top-level or nested Value."""
    dt = rec.get("DateTime") or rec.get("dateTime")
    ins = rec.get("InsertedDate")
    if dt is not None and ins is not None:
        return dt, ins
    vd = _value_dict(rec)
    if dt is None:
        dt = vd.get("DateTime") or vd.get("dateTime")
    if ins is None:
        ins = vd.get("InsertedDate")
    return dt, ins


def _to_minute_iso(val: Any) -> str | None:
    """Floor timestamp to minute in UTC ISO. Fast path for common Fitbit string shapes."""
    if val is None:
        return None

    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None

        # Date-only: YYYY-MM-DD
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            return f"{s}T00:00:00+00:00"

        # Naive or Zulu ISO datetime: YYYY-MM-DDTHH:MM...
        if (
            len(s) >= 16
            and s[4] == "-"
            and s[7] == "-"
            and s[10] in "T "
            and s[13] == ":"
        ):
            # Non-UTC numeric offsets need full parse + UTC convert
            body = s[:-1] if s[-1] in "Zz" else s
            if "+" not in body[11:] and "-" not in body[11:]:
                return f"{s[:10]}T{s[11:16]}:00+00:00"

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

    if isinstance(val, datetime):
        parsed = val
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.replace(second=0, microsecond=0).isoformat()

    if isinstance(val, (int, float)):
        try:
            parsed = datetime.fromtimestamp(float(val), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
        return parsed.replace(second=0, microsecond=0).isoformat()

    # Rare fallback (non-string / odd types)
    ts = pd.to_datetime(val, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts.floor("min").isoformat()


def _heart_rate_float(v: Any) -> float | None:
    """Parse HR bpm without pandas overhead."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for key in ("value", "bpm", "Value"):
            if key in v and v[key] is not None:
                return _heart_rate_float(v[key])
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _heart_rate_value(rec: dict) -> Any:
    """Heart rate (bpm) from top-level Value — Fitbit intraday is usually a number."""
    v = rec.get("Value")
    if isinstance(v, dict):
        return v.get("value") or v.get("bpm") or v.get("Value")
    hr = _heart_rate_float(v)
    if hr is None:
        return v
    return int(hr) if hr.is_integer() else hr


def _slim_activities_heart_record(rec: dict) -> dict[str, Any] | None:
    """One raw row: minute-rounded times + HR (used to aggregate many seconds into one minute)."""
    dt_raw, ins_raw = _activities_heart_datetimes(rec)
    dt_m = _to_minute_iso(dt_raw)
    if dt_m is None:
        return None
    return {
        "ParticipantIdentifier": rec.get("ParticipantIdentifier"),
        "Value": _heart_rate_value(rec),
        "DateTime": dt_m,
        "InsertedDate": _to_minute_iso(ins_raw),
    }


def _line_may_match_target(line: str) -> bool:
    """Cheap reject before json.loads for non-target Fitbit types."""
    for needle in _TARGET_TYPE_NEEDLES:
        if needle in line:
            return True
    return False


def _process_one_date_folder(folder_str: str) -> dict[str, Any]:
    """Worker: filter FitbitIntradayCombined JSONL for one date folder."""
    folder = Path(folder_str)
    output_names = set(TARGET_TYPES.values())

    input_files = sorted(
        p for p in folder.glob(PATTERN)
        if p.is_file() and p.name not in output_names and not p.name.startswith("_ONLY_")
    )
    if not input_files:
        return {"folder": folder.name, "counts": None, "skipped": True}

    # activities-heart: average Value per (ParticipantIdentifier, DateTime minute)
    # value layout: [sum, n, ins_max]
    heart_agg: dict[tuple[Any, str], list[Any]] = {}
    kept_counts = {t: 0 for t in TARGET_TYPES}
    dumps = json.dumps

    out_handles: dict[str, Any] = {}
    try:
        for t, out_name in TARGET_TYPES.items():
            if t == "activities-heart":
                continue
            out_handles[t] = (folder / out_name).open(
                "w", encoding="utf-8", buffering=_IO_BUFFER
            )

        for p in input_files:
            with p.open("r", encoding="utf-8-sig", errors="replace", buffering=_IO_BUFFER) as f:
                for line_no, ln in enumerate(f, start=1):
                    if not _line_may_match_target(ln):
                        continue
                    s = ln.strip()
                    if not s:
                        continue
                    try:
                        rec = json.loads(s)
                    except json.JSONDecodeError as e:
                        print(f"[skip bad line] {p}:{line_no} ({e})")
                        continue
                    if not isinstance(rec, dict):
                        continue

                    t = rec.get("Type")
                    if t == "activities-heart":
                        dt_raw = rec.get("DateTime") or rec.get("dateTime")
                        if dt_raw is None:
                            vd = _value_dict(rec)
                            dt_raw = vd.get("DateTime") or vd.get("dateTime")
                            ins_raw = rec.get("InsertedDate") or vd.get("InsertedDate")
                        else:
                            ins_raw = rec.get("InsertedDate")
                            if ins_raw is None:
                                ins_raw = _value_dict(rec).get("InsertedDate")

                        dt_m = _to_minute_iso(dt_raw)
                        if dt_m is None:
                            continue
                        hr_f = _heart_rate_float(rec.get("Value"))
                        if hr_f is None:
                            continue

                        ins_m = _to_minute_iso(ins_raw)
                        key = (rec.get("ParticipantIdentifier"), dt_m)
                        b = heart_agg.get(key)
                        if b is None:
                            heart_agg[key] = [hr_f, 1, ins_m]
                        else:
                            b[0] += hr_f
                            b[1] += 1
                            if ins_m is not None and (b[2] is None or ins_m > b[2]):
                                b[2] = ins_m

                    elif t in out_handles:
                        out_handles[t].write(dumps(rec, ensure_ascii=False))
                        out_handles[t].write("\n")
                        kept_counts[t] += 1

        heart_path = folder / TARGET_TYPES["activities-heart"]
        with heart_path.open("w", encoding="utf-8", buffering=_IO_BUFFER) as hf:
            for (pid, dt_m) in sorted(
                heart_agg.keys(),
                key=lambda k: ("" if k[0] is None else str(k[0]), k[1]),
            ):
                b = heart_agg[(pid, dt_m)]
                n = b[1]
                avg = b[0] / n if n else 0.0
                val_out = int(round(avg)) if abs(avg - round(avg)) < 1e-9 else round(avg, 2)
                row = {
                    "ParticipantIdentifier": pid,
                    "Value": val_out,
                    "DateTime": dt_m,
                    "InsertedDate": b[2],
                }
                hf.write(dumps(row, ensure_ascii=False))
                hf.write("\n")
        kept_counts["activities-heart"] = len(heart_agg)

    finally:
        for h in out_handles.values():
            h.close()

    return {"folder": folder.name, "counts": kept_counts, "skipped": False}


def extract_types_per_date_folder() -> None:
    date_range = pd.date_range(start=DATE_START, end=DATE_END, freq="D")
    folders = [
        str(ROOT_DIR / d.strftime(FOLDER_FMT))
        for d in date_range
        if (ROOT_DIR / d.strftime(FOLDER_FMT)).exists()
    ]
    if not folders:
        print("No date folders found for Fitbit extraction.")
        return

    workers = EXTRACT_WORKERS
    print(f"Extracting Fitbit types from {len(folders)} folders using {workers} worker(s)...")

    results: list[dict[str, Any]]
    if workers == 1:
        results = [_process_one_date_folder(f) for f in folders]
    else:
        try:
            results = []
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_process_one_date_folder, f): f for f in folders}
                for fut in as_completed(futures):
                    results.append(fut.result())
            results.sort(key=lambda r: r["folder"])
        except Exception as e:
            # Interactive / notebook contexts often cannot spawn process pools.
            print(f"Parallel extract unavailable ({e}); falling back to sequential.")
            results = [_process_one_date_folder(f) for f in folders]

    for r in results:
        if r.get("skipped"):
            continue
        print(f"[OK] {r['folder']}")
        counts = r["counts"] or {}
        for t in TARGET_TYPES:
            print(f"  {t}: {counts.get(t, 0)} rows -> {TARGET_TYPES[t]}")


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
        # Fast path: leading YYYY-MM-DD
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            return s[:10]
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

    if isinstance(val, datetime):
        parsed = val
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.date().isoformat()

    if isinstance(val, date):
        return val.isoformat()

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

            user = value_obj.get("user") if isinstance(value_obj.get("user"), dict) else {}
            demographics = user.get("demographics") if isinstance(user.get("demographics"), dict) else {}
            custom_fields = user.get("customFields") if isinstance(user.get("customFields"), dict) else {}

            # timestamp under Value (fallback: Value.event.timestamp)
            timestamp = value_obj.get("timestamp")
            if timestamp is None:
                event = value_obj.get("event")
                if isinstance(event, dict):
                    timestamp = event.get("timestamp")

            participantidentifier = r.get("ParticipantIdentifier")
            if participantidentifier is None or (
                isinstance(participantidentifier, float) and pd.isna(participantidentifier)
            ):
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
