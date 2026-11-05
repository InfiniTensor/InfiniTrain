"""Machine-readable and human-readable regression results."""

import collections
import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

from execution import atomic_json


def write_reports(output, plan, results, metadata):
    output = Path(output)
    results = list(results)
    expected = [case["id"] for case in plan["cases"]]
    observed = [item["id"] for item in results if item.get("kind") == "training"]
    complete = (
        bool(expected)
        and len(expected) == len(set(expected))
        and len(observed) == len(set(observed))
        and set(observed) == set(expected)
    )
    if not complete:
        results.append(
            {
                "id": "infrastructure/result-integrity",
                "kind": "infrastructure",
                "status": "ERROR",
                "reasons": [
                    "Result IDs must exactly and uniquely cover the nonempty resolved plan"
                ],
            }
        )
    counts = dict(collections.Counter(r["status"] for r in results))
    accepted = {"PASS"} | (set() if metadata.get("strict") else {"RECORDED"})
    success = complete and all(r["status"] in accepted for r in results)
    summary = {
        "schema_version": 1,
        "suite": plan["suite"],
        "success": success,
        "complete": complete,
        "comparison_complete": complete and all(
            r["status"] == "PASS" for r in results if r.get("kind") == "training"
        ),
        "counts": counts,
        "metadata": metadata,
        "results": results,
        "plan": str(output / "resolved_plan.json"),
    }
    coverage = dict(plan["coverage"])
    passed = {r["id"] for r in results if r["status"] == "PASS"}
    coverage["executed_case_ids"] = [r["id"] for r in results if r.get("stages")]
    coverage["passed_case_ids"] = sorted(
        passed.intersection(c["id"] for c in plan["cases"])
    )
    coverage["failed_or_unverified_case_ids"] = [
        c["id"] for c in plan["cases"] if c["id"] not in passed
    ]
    import planning

    matrices = []
    for matrix in coverage.get("matrices", []):
        matrix = dict(matrix)
        rows = [
            c["factors"]
            for c in plan["cases"]
            if c["id"] in passed
            and "factors" in c
            and c["scenario"] == matrix["scenario"]
            and c["model"] == matrix["model"]
            and "reference" not in c["tags"]
        ]
        actual = set().union(*(planning.pair_keys(r) for r in rows)) if rows else set()
        matrix["verified_pairs"] = sorted(actual)
        matrix["unverified_pairs"] = sorted(set(matrix["required_pairs"]) - actual)
        matrices.append(matrix)
    coverage["matrices"] = matrices
    coverage["recorded_case_ids"] = sorted(
        result["id"]
        for result in results
        if result.get("kind") == "training" and result["status"] == "RECORDED"
    )
    atomic_json(output / "coverage.json", coverage)
    completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_manifest = {
        "schema_version": 1,
        "run_id": output.name,
        "started_at": metadata.get("started_at"),
        "completed_at": completed_at,
        "suite": plan["suite"],
        "backend": metadata.get("environment", {}).get("backend"),
        "source": metadata.get("source", {}),
        "environment": metadata.get("environment", {}),
        "selection_scope": metadata.get("selection_scope"),
        "alignment": metadata.get("alignment"),
        "build_provenance": metadata.get("build_provenance"),
        "success": success,
        "summary": str(output / "summary.json"),
        "cases": [
            {
                "case_id": result["id"],
                "status": result["status"],
                "reproduce": result.get("reproduce"),
                "commands": [
                    {
                        "reference": bool(stage.get("reference")),
                        "repeat": stage.get("repeat"),
                        "argv": stage.get("process", {}).get("argv"),
                        "cwd": stage.get("process", {}).get("cwd"),
                        "log": stage.get("process", {}).get("log"),
                        "elapsed_seconds": stage.get("process", {}).get("elapsed_seconds"),
                    }
                    for stage in result.get("stages", [])
                ],
            }
            for result in results
            if result.get("kind") == "training"
        ],
    }
    metadata["completed_at"] = completed_at
    atomic_json(output / "run.json", run_manifest)
    root = ET.Element(
        "testsuite",
        name="InfiniTrain " + plan["suite"],
        tests=str(len(results)),
        failures=str(sum(r["status"] == "FAIL" for r in results)),
        errors=str(
            sum(
                (r["status"] == "RECORDED" and metadata.get("strict", False)) or r["status"]
                in (
                    "ERROR",
                    "INCONCLUSIVE",
                    "BLOCKED",
                    "NOT_COMPARABLE",
                )
                for r in results
            )
        ),
        skipped=str(sum(r["status"] == "SKIP" or (r["status"] == "RECORDED" and not metadata.get("strict")) for r in results)),
    )
    lines = [
        "# Regression {}\n".format(plan["suite"]),
        "Result: **{}**.\n".format(
            "RECORDED" if success and counts.get("RECORDED") else "PASS" if success else "NOT PASSED"
        ),
        "Scope: {} selected training cases out of {}; partial selection: {}.\n".format(
            len(plan["cases"]),
            plan["coverage"].get("suite_cases", len(plan["cases"])),
            plan["coverage"].get("partial", False),
        ),
        "Unit tests explicitly omitted: {}.\n".format(
            metadata.get("unit_tests_omitted", False)
        ),
        "| Case | Status | Reason |",
        "|---|---|---|",
    ]
    for result in results:
        case = ET.SubElement(
            root,
            "testcase",
            name=result["id"],
            classname=result.get("kind", "training"),
            time=str(result.get("elapsed_seconds", 0)),
        )
        reasons = result.get("reasons", [])
        reason = "; ".join(reasons)
        status = result["status"]
        if status != "PASS":
            tag = (
                "failure"
                if status == "FAIL"
                else "skipped"
                if status == "SKIP" or (status == "RECORDED" and not metadata.get("strict"))
                else "error"
            )
            ET.SubElement(case, tag, type=status, message=reason).text = reason
        ET.SubElement(case, "system-out").text = result.get("reproduce", "")
        lines.append(
            "| {} | {} | {} |".format(
                result["id"].replace("|", "\\|"),
                status,
                reason.replace("|", "\\|").replace("\n", " "),
            )
        )
    ET.ElementTree(root).write(
        str(output / "junit.xml"), encoding="utf-8", xml_declaration=True
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    # Publish completion only after all reports exist; concurrent runs use this marker.
    atomic_json(output / "summary.json", summary)
    return summary
