"""The alert rules against the metrics they read (NFR-10, DEC-22, S3-7b), hermetically.

promtool tests the rules' logic in CI's deploy-config job (``deploy/prometheus/rules.test.yml``),
but on series its test file writes by hand. These tests hold both files to what
:data:`rag_qa.observability.metrics.REGISTRY` exposes: every metric they name exists, and every
``le`` is a bucket the histogram has, spelled as prometheus_client spells it (``"2.0"``, never
``"2"``). A metric renamed in ``metrics.py`` would otherwise leave the rules silent and their
tests green. And every alert has a test that fires it and one that does not.
"""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from prometheus_client import generate_latest

from rag_qa.observability import metrics

DEPLOY = Path(__file__).resolve().parent.parent / "deploy"
RULES = DEPLOY / "prometheus" / "rules.yml"
TESTS = DEPLOY / "prometheus" / "rules.test.yml"
PROMETHEUS = DEPLOY / "prometheus" / "prometheus.yml"
ALERTMANAGER = DEPLOY / "alertmanager" / "alertmanager.yml"

#: A metric's name, then its labels, in a PromQL expression or a promtool series.
SELECTOR = re.compile(r"\b(rag_[a-z_]+|up)\b(\{[^}]*\})?")
LE = re.compile(r'le="([^"]*)"')


def load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def exposed() -> tuple[set[str], dict[str, set[str]]]:
    """The names the registry exposes, families and samples, and each histogram's ``le``
    values as written in the exposition."""
    names: set[str] = set()
    buckets: dict[str, set[str]] = {}
    for line in generate_latest(metrics.REGISTRY).decode().splitlines():
        if line.startswith("# TYPE "):
            names.add(line.split()[2])
        elif line and not line.startswith("#"):
            name = re.split(r"[{ ]", line, maxsplit=1)[0]
            names.add(name)
            for le in LE.findall(line):
                buckets.setdefault(name, set()).add(le)
    return names, buckets


def rules() -> list[dict[str, Any]]:
    return [rule for group in load(RULES)["groups"] for rule in group["rules"]]


def selectors() -> list[tuple[str, str, str]]:
    """Every metric selector in the rules' expressions and the tests' input series:
    ``(where, name, labels)``."""
    found = [
        ("rules.yml", name, labels)
        for rule in rules()
        for name, labels in SELECTOR.findall(rule["expr"])
    ]
    for test in load(TESTS)["tests"]:
        for series in test.get("input_series", []):
            found += [
                ("rules.test.yml", n, labels) for n, labels in SELECTOR.findall(series["series"])
            ]
    return found


def test_every_metric_the_rules_and_their_tests_name_is_exposed() -> None:
    names, _ = exposed()
    missing = {
        (where, name) for where, name, _ in selectors() if name != "up" and name not in names
    }
    assert missing == set()


def test_every_le_is_a_bucket_of_its_histogram_as_it_is_exposed() -> None:
    _, buckets = exposed()
    wrong = [
        (where, name, le)
        for where, name, labels in selectors()
        for le in LE.findall(labels)
        if le not in buckets.get(name, set())
    ]
    assert wrong == []


@pytest.mark.parametrize(
    ("histogram", "objective"),
    [
        ("rag_retrieval_seconds", "0.3"),
        ("rag_sources_seconds", "2.0"),
        ("rag_ttft_seconds", "300.0"),
    ],
)
def test_each_latency_slo_reads_its_objective_s_bucket(histogram: str, objective: str) -> None:
    # NFR-1, NFR-2a, NFR-2b: the rules' threshold is the objective, not a bucket beside it.
    read = {
        le
        for where, name, labels in selectors()
        if where == "rules.yml" and name == f"{histogram}_bucket"
        for le in LE.findall(labels)
    }
    assert read == {objective}


def alert_cases() -> dict[str, tuple[bool, bool]]:
    """For each alert the tests name: whether one test fires it and one keeps it quiet."""
    cases: dict[str, tuple[bool, bool]] = {}
    for test in load(TESTS)["tests"]:
        for case in test.get("alert_rule_test", []):
            fires, quiet = cases.get(case["alertname"], (False, False))
            cases[case["alertname"]] = (
                fires or bool(case["exp_alerts"]),
                quiet or not case["exp_alerts"],
            )
    return cases


def test_every_alert_has_a_test_that_fires_it_and_one_that_does_not() -> None:
    alerts = {rule["alert"] for rule in rules() if "alert" in rule}
    cases = alert_cases()
    assert {alert: cases.get(alert, (False, False)) for alert in alerts} == {
        alert: (True, True) for alert in alerts
    }
    assert set(cases) <= alerts  # no test of an alert that does not exist


def test_the_files_agree_on_the_job_the_rule_files_and_the_receiver() -> None:
    prometheus = load(PROMETHEUS)
    (job,) = prometheus["scrape_configs"]
    assert job["job_name"] == "rag"
    assert prometheus["rule_files"] == ["rules.yml"] == load(TESTS)["rule_files"]
    assert {'up{job="rag"}'} <= {f"{name}{labels}" for _, name, labels in selectors()}
    alertmanager = load(ALERTMANAGER)
    assert alertmanager["route"]["receiver"] == "null"
    assert [receiver["name"] for receiver in alertmanager["receivers"]] == ["null"]
