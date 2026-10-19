#!/usr/bin/env python3
"""Sequential, resumable entity extraction. Python 3.9+, no dependencies.

Edit CSV_FILE, DEFAULT_URL, and USER_ID in the .env beside this script.
Run: python3 mix-exxtract-script.py
Or:  python3 mix-exxtract-script.py --limit 100
Completed rows in the selected method output column are skipped. --limit counts new rows.
Each successful response is atomically saved to the original CSV. Keep a backup
if original CSV formatting matters; CSV quoting/line endings may be normalized.
Do not edit the CSV or run another copy of this script while it is processing.
"""

import argparse
import csv
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import urllib.error
import urllib.request


ENV_FILE = Path(__file__).resolve().with_name(".env")


def load_config():
    """Read simple KEY=VALUE settings from the .env beside this script."""
    settings = {}
    if ENV_FILE.exists():
        for number, line in enumerate(ENV_FILE.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Invalid .env line {number}: expected KEY=VALUE")
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if value.startswith(("'", '"')):
                if len(value) < 2 or value[-1] != value[0]:
                    raise ValueError(f"Unclosed quote on .env line {number}")
                value = value[1:-1]
            settings[key] = value
    # Environment variables override .env; command-line arguments override both.
    return {key: os.environ.get(key, settings.get(key))
            for key in ("CSV_FILE", "DEFAULT_URL", "USER_ID")}


FIELDS = ("recipientName", "recipientPhone", "amount", "currency", "commission", "purpose")


OUTPUT_COLUMNS = {
    "userMix": "mix_output",
    "userGemini": "gemnini_output",
    "userFinetunned": "finetunned_output",
}


def parse_entities(payload, user_id="userMix"):
    """Accept a direct entity object or OpenAI-style choices[0].message.content."""
    value = payload
    for _ in range(8):
        if isinstance(value, dict):
            if "choices" in value:
                try:
                    value = value["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise ValueError("Response has no usable choices[0].message.content") from exc
                continue
            break
        if not isinstance(value, str):
            raise ValueError("Response does not contain an entity object")
        value = value.strip()
        if value.startswith("```") and value.endswith("```"):
            lines = value.splitlines()
            value = "\n".join(lines[1:-1]).strip()
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Entity content is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Response does not contain an entity object")
    if user_id not in OUTPUT_COLUMNS:
        raise ValueError(f"Unsupported user_id: {user_id}")
    if user_id == "userFinetunned":
        correctness = {"amount": "amountCorrectness", "currency": "currencyCorrectness"}
    elif user_id == "userMix":
        correctness = {"amount": "amount_correctness"}
    else:
        # Gemini may return entities directly; honor correctness flags if supplied.
        correctness = {}
        for field, flags in (("amount", ("amount_correctness", "amountCorrectness")),
                             ("currency", ("currency_correctness", "currencyCorrectness"))):
            for flag in flags:
                if flag in value:
                    correctness[field] = flag
                    break
    missing = set(FIELDS).union(correctness.values()) - value.keys()
    if missing:
        raise ValueError("Response missing required fields: " + ", ".join(sorted(missing)))
    result = {key: value[key] for key in FIELDS}
    for field, flag in correctness.items():
        if type(value[flag]) is not bool:
            raise ValueError(f"{flag} must be JSON true or false")
        if not value[flag]:
            result[field] = None
    for key, item in result.items():
        if isinstance(item, (dict, list, bool)):
            raise ValueError(f"Invalid entity value for {key}")
    if all(item is None or (isinstance(item, str) and not item.strip())
           for item in result.values()):
        raise ValueError("No entities found in response")
    # Reject nonstandard JSON numeric values such as NaN or Infinity.
    json.dumps(result, allow_nan=False)
    return result


def request_entities(message, args):
    print(f"\n[INPUT MESSAGE]\n{message}", flush=True)
    body = {"user_id": args.user_id, "prompt_name": args.prompt_name,
            "messages": [{"role": "user", "content": message}], "temperature": 0}
    request = urllib.request.Request(
        args.url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            raw_response = response.read().decode("utf-8")
            payload = json.loads(raw_response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Adaptor returned HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Adaptor connection failed: {exc.reason}") from exc
    entities = parse_entities(payload, args.user_id)
    print("[EXTRACTED ENTITIES]", flush=True)
    print(json.dumps(entities, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return entities


def save_csv(path, headers, rows, encoding, dialect):
    """Replace only after a complete file is written in the same directory."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding=encoding, newline="",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            os.chmod(temp_path, stat.S_IMODE(path.stat().st_mode))
            writer = csv.DictWriter(stream, fieldnames=headers, dialect=dialect,
                                    doublequote=True, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def nonnegative(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("Must be zero or greater")
    return value


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_file", nargs="?", default=config["CSV_FILE"],
                        help="Optional CSV path override; defaults to CSV_FILE in .env")
    parser.add_argument("--limit", type=nonnegative, help="New rows to process; 0 means all")
    parser.add_argument("--message-column", default="input")
    parser.add_argument("--url", default=config["DEFAULT_URL"])
    parser.add_argument("--user-id", default=config["USER_ID"], choices=OUTPUT_COLUMNS)
    parser.add_argument("--prompt-name", default="extract-entities-main")
    parser.add_argument("--timeout", type=float, default=120, help="Request timeout in seconds")
    args = parser.parse_args()
    for key, value in (("CSV_FILE", args.csv_file), ("DEFAULT_URL", args.url),
                       ("USER_ID", args.user_id)):
        if not value or not value.strip():
            parser.error(f"Set {key} in {ENV_FILE} or provide its command-line override")
    if args.user_id not in OUTPUT_COLUMNS:
        parser.error("USER_ID must be one of: " + ", ".join(OUTPUT_COLUMNS))
    output_column = OUTPUT_COLUMNS[args.user_id]
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    filename = args.csv_file
    if not filename:
        raise ValueError("A CSV file path is required")
    path = Path(filename).expanduser().resolve(strict=True)
    if args.limit is None:
        entered = input("How many new messages? [Enter for all]: ").strip()
        args.limit = nonnegative(entered) if entered else 0

    raw = path.read_bytes()
    encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    content = raw.decode(encoding)
    try:
        dialect = csv.Sniffer().sniff(content[:65536], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    # Message/JSON apostrophes can mislead Sniffer's quote detection.
    dialect.quotechar = '"'
    dialect.doublequote = True
    reader = csv.DictReader(io.StringIO(content, newline=""), dialect=dialect)
    headers = reader.fieldnames
    if not headers or args.message_column not in headers:
        raise ValueError(f"CSV must contain column {args.message_column!r}")
    if len(headers) != len(set(headers)):
        raise ValueError("CSV contains duplicate column names")
    rows = list(reader)
    for number, row in enumerate(rows, 1):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed CSV data row {number}: wrong number of columns")
    if output_column not in headers:
        headers.append(output_column)
        for row in rows:
            row[output_column] = ""
    pending = [(number, row) for number, row in enumerate(rows, 1)
    if not row[output_column].strip()]
    if args.limit:
        pending = pending[:args.limit]
    print(f"Processing {len(pending)} pending message(s) for {args.user_id} → {output_column}. Ctrl+C stops safely.", flush=True)
    saved = 0
    for number, row in pending:
        try:
            message = row[args.message_column]
            if not message.strip():
                raise ValueError("Message is empty")
            entities = request_entities(message, args)
            row[output_column] = json.dumps(entities, ensure_ascii=False, allow_nan=False)
            save_csv(path, headers, rows, encoding, dialect)
        except Exception as exc:
            raise RuntimeError(f"Stopped at CSV data row {number}: {exc}. "
                               "Earlier saved results are retained; rerun to resume.") from exc
        saved += 1
        print(f"Saved data row {number} ({saved}/{len(pending)}).", flush=True)
    print(f"Done. Saved {saved} new result(s) to {path}.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Saved results are retained; rerun to resume.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
