"""Generate hourly behavioral reports from the minute-level perception outputs.

For a stay, groups the per-minute perception outputs (both camera views) into
local-time hour blocks, pulls the medication events of that hour from the
stay's EHR event list, and calls the report model with the report prompt
(prompts/report_hourly_v1.yaml) to produce the six-section report.

Reports are saved to <TRACER_DATA_ROOT>/reports/<pat_id>/hour_<YYYY-MM-DD>_<HH>.md,
which is where the report tools read them.

The report model is called with no generation configuration (API defaults).
``--model mock:report`` produces a deterministic template report offline
(tracer.backends.mock) for the demo.

Usage:
    python -m tracer.reports.generate_reports --pat-id stay_A \
        --perception-dir data/perception/<model>/<prompt>/run_0 [--model gemini-3.1-flash-lite]
        [--events data/ehr/stay_A_events.json] [--dry-run] [--save-prompt FILE] [--skip-existing]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from .. import config, data
from ..perception.build_timeline import parse_output_filename
from ..perception.prompts import load_prompt

DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_PROMPT = "report_hourly_v1"

# Medication event types included in the report context.
MED_EVENT_TYPES = {"med_sedation", "med_cns_adjunct"}

STOP_MAR_ACTIONS = {"Stopped", "Completed", "Paused", "Medication Hold", "Patch Removed"}


def group_outputs_by_hour(output_files: list[Path], room: str | None = None) -> dict:
    """(local date, hour) -> {'lt': [(minute, parsed)], 'rt': [...], start_local, end_local}."""
    tz = config.local_tz()
    hourly = defaultdict(lambda: {"lt": [], "rt": []})
    for path in output_files:
        parsed = parse_output_filename(path.name)
        if parsed is None or (room and parsed["room"] != room):
            continue
        local_dt = parsed["utc"].astimezone(tz)
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        pr = d.get("parsed_response")
        if pr is None:
            raw = d.get("raw_response", "")
            try:
                pr = json.loads(raw) if raw else None
            except (json.JSONDecodeError, TypeError):
                pr = None
            if pr is None:
                continue
        hourly[(local_dt.strftime("%Y-%m-%d"), local_dt.hour)][parsed["view"]].append((local_dt.minute, pr))
    result = {}
    for key, views in hourly.items():
        for cam in ("lt", "rt"):
            views[cam].sort(key=lambda x: x[0])
        all_minutes = [m for cam in ("lt", "rt") for m, _ in views[cam]]
        if not all_minutes:
            continue
        date_str, hour = key
        y, mo, d = (int(x) for x in date_str.split("-"))
        base_dt = datetime(y, mo, d, hour, tzinfo=tz)
        views["start_local"] = base_dt + timedelta(minutes=min(all_minutes))
        views["end_local"] = base_dt + timedelta(minutes=max(all_minutes))
        views["hour_key"] = key
        result[key] = views
    return result


def _event_type(e: dict) -> str:
    return e.get("type") or e.get("event_type") or ""


def _event_time(e: dict):
    return e.get("time") or e.get("timestamp")


def load_medications(events: list[dict]) -> list[dict]:
    meds = [e for e in events if _event_type(e) in MED_EVENT_TYPES]
    meds.sort(key=lambda e: _event_time(e) or "")
    return meds


def get_meds_for_hour(all_meds: list[dict], hour_start, hour_end) -> list[dict]:
    """Meds administered during the hour, plus continuous IV infusions started
    in the previous 24 h whose last action was not a stop."""
    tz = config.local_tz()
    parsed = []
    for m in all_meds:
        try:
            ts = datetime.fromisoformat(str(_event_time(m)))
            ts = ts.replace(tzinfo=tz) if ts.tzinfo is None else ts.astimezone(tz)
        except (ValueError, TypeError):
            continue
        parsed.append((ts, m))
    hour_meds = [_format_med(ts, m) for ts, m in parsed if hour_start <= ts <= hour_end]
    lookback_start = hour_start - timedelta(hours=24)
    infusion_state = {}
    for ts, m in parsed:
        if ts >= hour_start:
            break
        if ts < lookback_start:
            continue
        d = m.get("data", {})
        if d.get("route") != "Intravenous" or d.get("infusion_rate") is None:
            continue
        infusion_state[(d.get("medication_id", d.get("display_name")), d.get("route"))] = (ts, d.get("mar_action", ""), m)
    for _key, (ts, action, m) in infusion_state.items():
        if action not in STOP_MAR_ACTIONS:
            f = _format_med(ts, m)
            f["context"] = f"Continuous infusion started at {ts.strftime('%H:%M')} (still running)"
            hour_meds.append(f)
    return hour_meds


def _format_med(ts, med_event: dict) -> dict:
    d = med_event.get("data", {})
    result = {"time": ts.strftime("%Y-%m-%d %H:%M"),
              "drug": d.get("display_name") or d.get("med_description", "Unknown"),
              "action": d.get("mar_action", "Unknown"),
              "dose": f"{d.get('sig', '?')} {d.get('dose_unit', '')}".strip(),
              "route": d.get("route", "Unknown"),
              "class": d.get("thera_class_name", ""), "subclass": d.get("pharm_class_name", "")}
    if d.get("infusion_rate"):
        result["infusion_rate"] = f"{d.get('infusion_rate')} {d.get('mar_inf_rate_unit', '')}".strip()
    return result


def build_prompt_text(prompt_template, hour_data: dict, hour_meds: list, hour_start, hour_end) -> str:
    lt_by_min = {m: pr for m, pr in hour_data["lt"]}
    rt_by_min = {m: pr for m, pr in hour_data["rt"]}
    all_minutes = set(lt_by_min) | set(rt_by_min)
    entries = []
    for minute in sorted(all_minutes):
        entry = {"minute": minute, "time": f"{hour_start.strftime('%H')}:{minute:02d}"}
        lt, rt = lt_by_min.get(minute), rt_by_min.get(minute)
        entry["left_view"] = {"labels": lt.get("labels", {}), "summary": lt.get("summary", "")} if lt else "NOT AVAILABLE"
        entry["right_view"] = {"labels": rt.get("labels", {}), "summary": rt.get("summary", "")} if rt else "NOT AVAILABLE"
        entries.append(entry)
    both = sum(1 for m in all_minutes if m in lt_by_min and m in rt_by_min)
    lt_only = sum(1 for m in all_minutes if m in lt_by_min and m not in rt_by_min)
    rt_only = sum(1 for m in all_minutes if m not in lt_by_min and m in rt_by_min)
    coverage_note = (f"Camera coverage: {len(all_minutes)} minutes total. {both} with both views, "
                     f"{lt_only} left-only, {rt_only} right-only.")
    return (f"{prompt_template.prompt}\n\n---\n\n{prompt_template.user_prompt}\n\n"
            f"Observation period: {hour_start.strftime('%Y-%m-%d %H:%M')} to "
            f"{hour_end.strftime('%Y-%m-%d %H:%M')} (local time)\n\n"
            f"{coverage_note}\n\n"
            f"ACTIVITY_TIMELINE = {json.dumps(entries, indent=2)}\n\n"
            f"MEDICATION_EVENTS = "
            f"{json.dumps(hour_meds, indent=2) if hour_meds else 'No medications were administered or actively infusing during this hour.'}")


def call_report_model(prompt_text: str, model: str) -> str:
    if model.startswith("mock:"):
        from ..backends import mock
        return mock.generate_report_text(prompt_text)
    from ..backends.gemini import call_gemini
    return call_gemini(prompt_text, model=model)


def generate_for_stay(pat_id: str, perception_dir: Path, *, model: str = DEFAULT_MODEL,
                      prompt_name: str = DEFAULT_PROMPT, events: list | None = None,
                      reports_root: Path | None = None, dry_run: bool = False, max_hours: int | None = None,
                      skip_existing: bool = False, save_prompt: Path | None = None, log=print) -> dict:
    prompt_template = load_prompt(prompt_name)
    files = sorted(Path(perception_dir).glob("*.json"))
    room = data.room_for(pat_id) or None
    hourly = group_outputs_by_hour(files, room)
    if events is None:
        tl_path = data.timeline_json(pat_id)
        events = json.loads(tl_path.read_text(encoding="utf-8")).get("events", []) if tl_path.exists() else []
    all_meds = load_medications(events)
    out_dir = (reports_root or data.reports_dir()) / pat_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[{pat_id}] {len(files)} perception files -> {len(hourly)} hour blocks; {len(all_meds)} medication events")
    generated = skipped = 0
    for hour_key in sorted(hourly)[: max_hours] if max_hours else sorted(hourly):
        hd = hourly[hour_key]
        date_str, hour = hour_key
        report_path = out_dir / f"hour_{date_str}_{hour:02d}.md"
        if skip_existing and report_path.exists():
            skipped += 1
            continue
        hour_meds = get_meds_for_hour(all_meds, hd["start_local"], hd["end_local"])
        if dry_run:
            log(f"  [{date_str} {hour:02d}:00] would generate: lt={len(hd['lt'])} rt={len(hd['rt'])} meds={len(hour_meds)}")
            generated += 1
            continue
        prompt_text = build_prompt_text(prompt_template, hd, hour_meds, hd["start_local"], hd["end_local"])
        if save_prompt:
            save_prompt.parent.mkdir(parents=True, exist_ok=True)
            save_prompt.write_text(prompt_text, encoding="utf-8")
            log(f"  saved example prompt to {save_prompt} ({len(prompt_text)} chars)")
            return {"generated": 0, "skipped": skipped, "saved_prompt": str(save_prompt)}
        t0 = time.time()
        try:
            report = call_report_model(prompt_text, model)
        except Exception as e:
            log(f"  [{date_str} {hour:02d}:00] ERROR: {e}")
            continue
        report_path.write_text(report, encoding="utf-8")
        generated += 1
        log(f"  [{date_str} {hour:02d}:00] wrote {report_path.name} ({time.time() - t0:.1f}s, {len(report)} chars)")
    return {"generated": generated, "skipped": skipped, "reports_dir": str(out_dir)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate hourly behavioral reports")
    ap.add_argument("--pat-id", required=True)
    ap.add_argument("--perception-dir", required=True, help="run directory with the stay's per-clip JSON outputs")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="report model (mock:report for the offline template)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--events", default=None, help="JSON list of EHR events (default: the stay's timeline JSON)")
    ap.add_argument("--reports-dir", default=None, help="default: <TRACER_DATA_ROOT>/reports")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-hours", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--save-prompt", default=None, metavar="FILE", help="write the first prompt to FILE and exit")
    args = ap.parse_args(argv)
    events = json.loads(Path(args.events).read_text(encoding="utf-8")) if args.events else None
    info = generate_for_stay(args.pat_id, Path(args.perception_dir), model=args.model, prompt_name=args.prompt,
                             events=events, reports_root=Path(args.reports_dir) if args.reports_dir else None,
                             dry_run=args.dry_run, max_hours=args.max_hours, skip_existing=args.skip_existing,
                             save_prompt=Path(args.save_prompt) if args.save_prompt else None)
    print(json.dumps(info, indent=2))
    if args.save_prompt:
        sys.exit(0)


if __name__ == "__main__":
    main()
