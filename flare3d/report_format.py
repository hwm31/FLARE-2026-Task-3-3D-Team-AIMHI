"""Shared region schema + output normaliser for the zero-shot 3D adapters.

The FLARE26 3D report task is scored per anatomical region, and both baselines' GREEN
drivers parse a report by walking it LINE BY LINE with

    re.match(r"([^:–\\-]+)[:–\\-]\\s*(.+)", line)

A model that answers "liver: ...; spleen: ...; pancreas: ..." on a single line is
therefore read as ONE region called "liver" whose text swallows every other region.

Med3DVLM (FLARE-tuned) already emits one region per line, so it needs nothing. The
zero-shot models (CT-CHAT, RadFM) are prompted by us and frequently answer on one line,
so we re-split their output on the known region names before writing the CSV. This is
output formatting inside our own adapter -- the scorer stays byte-identical to the
baselines'.
"""

from __future__ import annotations

import re

REGIONS = [
    "liver", "biliary system", "spleen", "pancreas", "kidneys",
    "endocrine system", "lymphatic system", "gastrointestinal tract",
    "abdominal cavity and peritoneum", "blood vessels", "musculoskeletal system",
    "lungs and pleura", "respiratory tract", "heart", "mediastinum", "esophagus",
    "breast tissue", "diaphragm", "urinary system",
]

REPORT_PROMPT = (
    "You are a radiologist reporting a CT scan. Describe the findings for each "
    "anatomical region below. Answer with exactly one line per region in the form "
    "'<region>: <findings>'.\nRegions: " + "; ".join(REGIONS[:18]) + "\n"
)

# longest-first so "lungs and pleura" wins over a bare "lungs"-like prefix
_REGION_RE = re.compile(
    r"(?i)(?:(?<=^)|(?<=[\s;.,]))(" + "|".join(re.escape(r) for r in
                                               sorted(REGIONS, key=len, reverse=True)) + r")\s*:")


def normalize_report(text: str) -> str:
    """Put each 'region: findings' pair on its own line.

    No-ops on text that is already one-region-per-line, and on free prose that never
    names a region (that case legitimately scores 0 as a parse failure).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    hits = list(_REGION_RE.finditer(text))
    if len(hits) < 2:
        return text.strip()

    # already one region per line? then leave it alone
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) >= len(hits):
        return text.strip()

    out = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        region = m.group(1).strip().lower()
        body = text[m.end():end].strip().strip(";").strip()
        if body:
            out.append(f"{region}: {body}")
    return "\n".join(out) if out else text.strip()
