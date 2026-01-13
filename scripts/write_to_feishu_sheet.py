import requests
import json
import time
import os
import argparse
import glob
import re
import pandas as pd
from datetime import datetime, date
import subprocess

# date/branch/commit/avg_latency/avg_throughput/peak_used/peak_reserved
META_COLS=7
HEADER_ROWS=5
HEADER_COLS="W"

class FeishuSheetHandler:
    """Feishu Sheet Handler for retrieving and writing sheet data"""

    def __init__(self, app_id, app_secret):
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = "https://open.feishu.cn/open-apis"
        self.access_token = None
        self.token_expire_time = 0
        self.get_access_token()

    def get_access_token(self):
        """Get and cache tenant_access_token"""
        if self.access_token and time.time() < self.token_expire_time:
            return self.access_token

        url = f"{self.base_url}/auth/v3/tenant_access_token/internal"
        resp = requests.post(url, json={"app_id": self.app_id, "app_secret": self.app_secret}, timeout=10)
        if resp.status_code != 200:
            print("Failed to get token: HTTP error", resp.status_code)
            return None

        data = resp.json()
        if data.get("code") != 0:
            print(f"Failed to get token: {data.get('msg')}")
            return None

        self.access_token = data.get("tenant_access_token")
        self.token_expire_time = time.time() + data.get("expire", 7200) - 600
        return self.access_token

    def _feishu_request(self, method, endpoint, **kwargs):
        """Unified Feishu API request wrapper"""
        token = self.get_access_token()
        if not token:
            return None

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8"
        }

        url = f"{self.base_url}{endpoint}"
        resp = requests.request(method, url, headers=headers, timeout=15, **kwargs)

        if resp.status_code != 200:
            print(f"Request failed: HTTP {resp.status_code}")
            return None

        data = resp.json()
        if data.get("code") != 0:
            print(f"Feishu returned error: {data.get('msg')}")
            return None

        return data

    def get_all_sheet_ids(self, spreadsheet_token):
        """Get list of all sheets"""
        # API reference：https://open.feishu.cn/document/server-docs/docs/sheets-v3/spreadsheet-sheet/query
        data = self._feishu_request("GET", f"/sheets/v3/spreadsheets/{spreadsheet_token}/sheets/query")
        if not data:
            return []
        sheets = [
            {"sheet_id": s["sheet_id"], "title": s["title"], "index": s.get("index", 0)}
            for s in data["data"]["sheets"]
        ]
        print(f"Retrieved  {len(sheets)} sheets")
        return sheets

    def prepend_data(self, spreadsheet_token, sheet_id, data):
        """Insert data after the header"""
        # API reference：https://open.feishu.cn/document/server-docs/docs/sheets-v3/data-operation/prepend-data
        payload = {"valueRange": {"range": f"{sheet_id}!A{HEADER_ROWS}:Z", "values": data}}
        data = self._feishu_request("POST", f"/sheets/v2/spreadsheets/{spreadsheet_token}/values_prepend", json=payload)
        if data:
            print(f"Successfully inserted 5 rows into {sheet_id}")
            return True
        return False

    def get_sheet_row_count(self, spreadsheet_token, sheet_id):
        """Get total row count of the sheet"""
        # API reference：https://open.feishu.cn/document/server-docs/docs/sheets-v3/spreadsheet-sheet/get
        data = self._feishu_request("GET", f"/sheets/v3/spreadsheets/{spreadsheet_token}/sheets/{sheet_id}")
        if data:
            return data["data"]["sheet"]["grid_properties"]["row_count"]
        return 0

    def set_style(self, spreadsheet_token, sheet_id, entry_index):
        """Set cell style for a given range"""
        # API reference：https://open.feishu.cn/document/server-docs/docs/sheets-v3/data-operation/batch-set-cell-style
        base_style = {"hAlign": 1, "vAlign": 1}
        if entry_index % 2 == 0:
            base_style["backColor"] = "#EFFAFF"

        start = HEADER_ROWS
        end = HEADER_ROWS + 4
        payload = {
            "data": [
                {"ranges": [f"{sheet_id}!A{start}:{HEADER_COLS}{end}"], "style": base_style},
                {"ranges": [f"{sheet_id}!A{start}:A{end}"], "style": {"formatter": "yyyy/MM/dd"}}
            ]
        }
        return self._feishu_request("PUT", f"/sheets/v2/spreadsheets/{spreadsheet_token}/styles_batch_update", json=payload) is not None

    def merge_columns(self, spreadsheet_token, sheet_id):
        """Merge columns A5:G9"""
        # API reference：https://open.feishu.cn/document/server-docs/docs/sheets-v3/data-operation/merge-cells
        start = HEADER_ROWS
        end = HEADER_ROWS + 4
        payload = {"range": f"{sheet_id}!A{start}:G{end}", "mergeType": "MERGE_COLUMNS"}
        return self._feishu_request("POST", f"/sheets/v2/spreadsheets/{spreadsheet_token}/merge_cells", json=payload) is not None

    def write_cmd_args_to_header(self, spreadsheet_token, cmd_args, sheet_id):
        """Write command args to A1:W1"""
        def col_letter_to_idx(letter: str) -> int:
            """A->1, Z->26, AA->27 ..."""
            letter = letter.strip().upper()
            idx = 0
            for ch in letter:
                idx = idx * 26 + (ord(ch) - ord('A') + 1)
            return idx

        data = [cmd_args] + [""] * (col_letter_to_idx(HEADER_COLS) - 1)
        payload = {"valueRange": {"range": f"{sheet_id}!A1:W1", "values": [data]}}
        return self._feishu_request("PUT", f"/sheets/v2/spreadsheets/{spreadsheet_token}/values", json=payload) is not None

    def create_sheet_for_testcase(self, spreadsheet_token, sheet_title, template_sheet_id):
        """Create a sheet from template given a specific title"""
        payload = {
            "requests": [
                {
                    "copySheet": {
                        "source": {"sheetId": template_sheet_id},
                        "destination": {"title": sheet_title}
                    }
                }
            ]
        }
        resp = self._feishu_request("POST", f"/sheets/v2/spreadsheets/{spreadsheet_token}/sheets_batch_update", json=payload)
        if resp:
            try:
                new_sheet_id = resp["data"]["replies"][0]["copySheet"]["properties"]["sheetId"]
                return new_sheet_id
            except Exception:
                print("Unexpected copySheet response:", resp)
                return None
        else:
            return None

    def sort_sheets_by_title(self, spreadsheet_token, template_title = "模板") -> bool:
        sheets = self.get_all_sheet_ids(spreadsheet_token)

        template = None
        normal = []
        for s in sheets:
            if s["title"] == template_title:
                template = s
            else:
                normal.append(s)

        def natural_key(s: str):
            return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', s)]

        normal.sort(key=lambda x: natural_key(x["title"]))
        ordered = normal + ([template] if template else [])

        requests_ = []
        for new_index, s in enumerate(ordered):
            sheet_id = s["sheet_id"]
            requests_.append({
                "updateSheet": {
                    "properties": {
                        "sheetId": sheet_id,
                        "index": new_index
                    }
                }
            })

        if not requests_:
            return True 

        payload = {"requests": requests_}
        return self._feishu_request("POST", f"/sheets/v2/spreadsheets/{spreadsheet_token}/sheets_batch_update", json=payload) is not None


    def post_process(self, spreadsheet_token, sheet_id):
        """Post-processing: set styles and merge cells"""
        row_count = self.get_sheet_row_count(spreadsheet_token, sheet_id)
        if row_count == 0:
            print("Unable to get total row count, skip post-processing")
            return False

        self.set_style(spreadsheet_token, sheet_id, (row_count - 3) // 5)

        return self.merge_columns(spreadsheet_token, sheet_id)

    @staticmethod
    def convert_to_feishu_date(dt):
        """Convert date to Feishu numeric date"""
        # Feishu uses the 1900 date system, same as Excel
        # Dates are represented as number of days since January 1, 1900
        # Manually add 2 days:
        #     1. +1 since index of 1900-01-01 is 1
        #     2. +1 due to leap year problem: https://en.wikipedia.org/wiki/Leap_year_problem
        if isinstance(dt, str):
            dt = datetime.strptime(dt, "%Y/%m/%d").date()
        elif isinstance(dt, datetime):
            dt = dt.date()
        base_date = date(1900, 1, 1)
        return (dt - base_date).days + 2


def load_config(config_file):
    """Load configuration from JSON file"""
    if not os.path.exists(config_file):
        print(f"Config file {config_file} does not exist")
        return None

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError:
        print(f"Config file {config_file} is not valid JSON file")
        return None

    required_keys = ["APP_ID", "APP_SECRET", "MODEL_SPREADSHEET_TOKEN"]
    for key in required_keys:
        if key not in config:
            print(f"Config file missing required key: {key}")
            return None

    if not isinstance(config["MODEL_SPREADSHEET_TOKEN"], dict) or not config["MODEL_SPREADSHEET_TOKEN"]:
        print("MODEL_SPREADSHEET_TOKEN must be a non-empty dictionary")
        return None

    return config

def parse_command_args(log_content: str, start_flag="--dtype"):
    """Parse command-line arguments from [COMMAND] line"""
    for line in log_content.splitlines():
        if line.startswith("[COMMAND]"):
            idx = line.find(start_flag)
            if idx != -1:
                return line[idx:].strip()
            return None
    return None

def parse_training_log(log_content):
    """Parse training log to extract avg latency and throughput from step >= 2 and peak mem usage during whole time"""
    pattern_with_peak = (
        r"step\s+(\d+)/\d+\s+\|.*?\|\s+\(\s*"
        r"(\d+(?:\.\d+)?)\s*ms\s*\|\s*"
        r"(\d+(?:\.\d+)?)\s*tok/s\s*\|\s*"
        r"peak used:\s*(\d+)\s*MB\s*\|\s*"
        r"peak reserved:\s*(\d+)\s*MB"
    )

    # NOTE(zbl): This is for compatibility reasons
    pattern_no_peak = (
        r"step\s+(\d+)/\d+\s+\|.*?\|\s+\(\s*"
        r"(\d+(?:\.\d+)?)\s*ms\s*\|\s*"
        r"(\d+(?:\.\d+)?)\s*tok/s"
    )

    matches = re.findall(pattern_with_peak, log_content)
    has_peak = True
    if not matches:
        matches = re.findall(pattern_no_peak, log_content)
        has_peak = False

    filtered = [m for m in matches if int(m[0]) > 1]
    if not filtered:
        print("No valid step data found in log")
        return None

    latencies = [float(m[1]) for m in filtered]
    throughputs = [int(m[2]) for m in filtered]

    avg_latency = round(sum(latencies) / len(latencies), 2)
    avg_throughput = round(sum(throughputs) / len(throughputs), 2)

    peak_used_max = None
    peak_reserved_max = None
    if has_peak:
        peak_used = [int(m[3]) for m in filtered]
        peak_reserved = [int(m[4]) for m in filtered]
        peak_used_max = max(peak_used)
        peak_reserved_max = max(peak_reserved)

    return [avg_latency, avg_throughput, peak_used_max, peak_reserved_max]


def parse_profile_report(profile_content):
    """Parse performance report and return DataFrame or None"""
    sort_columns = ['Device Total(us)', 'Avg Device(us)', 'Host Total(us)', 'Avg Host(us)']
    lines = profile_content.splitlines()

    # Locate Tag: Step_9 line
    step9_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Tag: Step_9"):
            step9_index = i
            break

    if step9_index is None:
        print("Tag: Step_9 not found in performance report")
        return None

    table_lines = []
    start_collecting = False

    # Collect until next Tag
    for line in lines[step9_index + 1:]:
        stripped = line.strip()
        if stripped.startswith("Tag:"):
            break
        if not start_collecting and "Peak Device Memory Usage:" in stripped:
            start_collecting = True
            continue
        if start_collecting and stripped:
            table_lines.append(stripped)

    if not table_lines:
        print("No table data after Tag: Step_9")
        return None

    headers = [h for h in re.split(r'\s{2,}', table_lines[0]) if h]
    data_rows = []

    for line in table_lines[1:]:
        row = [c for c in re.split(r'\s{2,}', line.strip()) if c]
        if len(row) == len(headers):
            data_rows.append(row)

    if not data_rows:
        print("No valid data rows in table")
        return None

    df = pd.DataFrame(data_rows, columns=headers)
    for col in headers[1:]:
        df[col] = df[col].replace('%', '', regex=True).apply(pd.to_numeric)

    # Concatenate top-5 sorted results horizontally
    dfs = []
    for col in sort_columns:
        if col not in df.columns:
            dfs.append(pd.DataFrame())
            continue
        sorted_df = df.sort_values(by=col, ascending=False).head(5)
        right_col = "Host %" if 'Host' in col else "Device %"
        required_cols = ["Name", col, right_col, "Count"] if right_col in sorted_df.columns else ["Name", col, "Count"]

        formatted_df = sorted_df[[c for c in required_cols if c in sorted_df.columns]].copy()
        if 'Count' in formatted_df.columns:
            formatted_df['Count'] = pd.to_numeric(formatted_df['Count'], errors='coerce').astype('Int64')
        dfs.append(formatted_df.reset_index(drop=True))

    merged_df = pd.concat(dfs, axis=1)
    if merged_df.shape[0] >= 5 and merged_df.shape[1] >= 16:
        return merged_df.head(5).iloc[:, :16]
    return None

def discover_testcases(model_name: str, log_dir="logs"):
    """Get all test case id from local log dir"""
    pattern = os.path.join(log_dir, f"{model_name}_*.log")
    files = glob.glob(pattern)
    testcases = []
    prefix = f"{model_name}_"
    for path in files:
        base = os.path.basename(path)
        if not base.startswith(prefix) or base.endswith("_profile.log") or not base.endswith(".log"):
            continue
        testcase = base[len(prefix):-len(".log")]
        if testcase:
            testcases.append(testcase)
    return sorted(set(testcases))

def get_git_branch():
    """Get current git branch"""
    try:
        result = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True)
        return result.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_git_commit_id():
    """Get current git commit id (first 7 chars)"""
    try:
        result = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        return result.strip()[:7]
    except subprocess.CalledProcessError:
        return "unknown"


def get_model_data(model_name, sheet_title):
    """Construct 2D list for writing to Feishu"""
    log_file_path = f"logs/{model_name}_{sheet_title}.log"
    profile_file_path = f"profile_logs/{model_name}_{sheet_title}_profile_{model_name}.report.rank0"

    avg_latency, avg_throughput, peak_used_max, peak_reserved_max = None, None, None, None

    # Read training log
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            result = parse_training_log(content)
            if result:
                avg_latency, avg_throughput, peak_used_max, peak_reserved_max = result
            cmd_args = parse_command_args(content)
    else:
        print(f"Training log does not exist: {log_file_path}")

    # Read performance report
    report_df = None
    if os.path.exists(profile_file_path):
        with open(profile_file_path, 'r', encoding='utf-8') as f:
            report_df = parse_profile_report(f.read())
    else:
        print(f"Performance report does not exist: {profile_file_path}")

    if report_df is None:
        return []

    # Insert $META_COLS empty columns at the front
    new_data = [["" for _ in range(META_COLS)] for _ in range(5)]
    new_df = pd.DataFrame(new_data, index=report_df.index)
    combined_df = pd.concat([new_df, report_df], axis=1)

    # Fill first row's first $META_COLS columns with info
    combined_df.iloc[0, 0] = FeishuSheetHandler.convert_to_feishu_date(datetime.now().date())
    combined_df.iloc[0, 1] = get_git_branch()
    combined_df.iloc[0, 2] = get_git_commit_id()
    if avg_latency is not None:
        combined_df.iloc[0, 3] = avg_latency
    if avg_throughput is not None:
        combined_df.iloc[0, 4] = avg_throughput
    if peak_used_max is not None:
        combined_df.iloc[0, 5] = peak_used_max
    if peak_reserved_max is not None:
        combined_df.iloc[0, 6] = peak_reserved_max

    return cmd_args, combined_df.values.tolist()


def main():
    parser = argparse.ArgumentParser(description='Script to write training metrics to Feishu sheets')
    parser.add_argument('config_file', help='Path to JSON config file (e.g. token.json)')
    args = parser.parse_args()

    config = load_config(args.config_file)
    if not config:
        print("Failed to load config file, exiting")
        return

    print(f"Successfully loaded config file: {args.config_file}")
    print(f"Found {len(config['MODEL_SPREADSHEET_TOKEN'])} models to process")

    handler = FeishuSheetHandler(
        app_id=config["APP_ID"],
        app_secret=config["APP_SECRET"]
    )

    for model_name, spreadsheet_token in config["MODEL_SPREADSHEET_TOKEN"].items():
        print(f"\n=== Start processing {model_name} ===")
        model_name = model_name.lower()

        testcases = discover_testcases(model_name)
        if not testcases:
            print(f"No local testcases found under logs/ for model={model_name}, skipping")
            continue
        print(f"Discovered {len(testcases)} local testcases: {testcases}")

        remote_sheets = handler.get_all_sheet_ids(spreadsheet_token)
        remote_by_title = {s["title"]: s["sheet_id"] for s in remote_sheets}

        if "模板" not in remote_by_title:
            print(f"No template sheets retrieved for {model_name}, skipping")
            continue
        template_sheet_id = remote_by_title["模板"]

        sort_sheets = False

        for testcase in testcases:
            print("\n-------")
            sheet_id = remote_by_title.get(testcase)
            write_cmd = False

            if not sheet_id:
                print(f"Sheet for '{testcase}' not found, creating from template...")
                sheet_id = handler.create_sheet_for_testcase(spreadsheet_token, sheet_title=testcase, template_sheet_id=template_sheet_id)
                if not sheet_id:
                    print(f"Failed to create sheet '{testcase}', skipping")
                    continue
                remote_by_title[testcase] = sheet_id
                sort_sheets = True
                write_cmd = True
                print(f"Created sheet '{testcase}' with id={sheet_id}")
            
            print(f"Processing testcase '{testcase}' -> sheet_id={sheet_id}")

            cmd_args, sheet_data = get_model_data(model_name=model_name, sheet_title=testcase)

            if not sheet_data:
                print("No valid data generated, skipping")
                continue

            if write_cmd and cmd_args:
                handler.write_cmd_args_to_header(spreadsheet_token, cmd_args, sheet_id)

            if handler.prepend_data(spreadsheet_token, sheet_id, sheet_data):
                handler.post_process(spreadsheet_token, sheet_id)

        if sort_sheets:
            handler.sort_sheets_by_title(spreadsheet_token, "模板")

    print("\n=== All models and sheets processed ===")


if __name__ == "__main__":
    main()
