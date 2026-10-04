"""Fictional inputs isolated from both development and holdout questions."""

from __future__ import annotations

import json

from eval.common import ROOT, sha256
from eval.evaluate import CoverageOutput, PairwiseOutput, QualityOutput

SCHEMAS = {
    "coverage": CoverageOutput,
    "quality": QualityOutput,
    "pairwise": PairwiseOutput,
}
QUESTION = "仅根据虚构研究笔记，比较青岚图书馆的星桥和雨舟配送方案：容量、每趟费用、适用时段及试验局限分别是什么？"
REQUIREMENTS = [
    "说明星桥与雨舟两种方案各自的每趟容量。",
    "说明两种方案各自的每趟费用，并指出低费用不等于全面更好。",
    "说明两种方案各自适用的配送时段。",
    "说明试验路线和条件不同，不能据此断言哪种方案更安全或普遍更可靠。",
]
NOTES = [
    "虚构笔记 S1：星桥每趟可装20箱，费用120元，适用白天。一次100趟试验中95趟完成配送。",
    "虚构笔记 S2：雨舟每趟可装12箱，费用80元，适用夜间。另一次100趟试验中90趟完成配送。",
    "虚构笔记 S3：两次试验使用不同路线和条件；没有安全事故统计，不能比较安全性，也不能直接推出普遍可靠性。",
]
ANSWERS = {
    "A": "星桥每趟20箱、120元，适用于白天；雨舟每趟12箱、80元，适用于夜间。雨舟每趟费用较低，但容量也较小，不能因此认定全面更好。星桥和雨舟各100趟试验分别完成95和90趟，路线和条件不同，不能直接推广或判断普遍可靠性。笔记没有安全数据，不能断言哪一种更安全。",
    "B": "星桥每趟可配送20箱。",
    "C": "星桥每趟20箱、120元；雨舟每趟12箱、80元。雨舟单趟费用较低，但容量较小，并不说明全面更好。星桥适合白天。雨舟更安全，因此值得优先推广。",
    "A_paraphrase": "白天可采用星桥：每次20箱，120元。夜间可采用雨舟：每次12箱，80元。雨舟花费少同时运量小，不能说所有方面都更优。两组100次试验完成次数为星桥95、雨舟90，但采用不同路线和条件，不能得出一般可靠性高低。安全事故数据没有提供，因此两者的安全性无法排序。",
}


def fixtures() -> dict:
    """Return frozen synthetic case content and expected qualitative directions."""
    return {
        "version": "synthetic-judge-v1",
        "question": QUESTION,
        "requirements": REQUIREMENTS,
        "synthetic_evidence_notes": NOTES,
        "answers": ANSWERS,
        "expected": {
            "coverage_A": ["covered"] * 4,
            "coverage_B": ["covered", "not_covered", "not_covered", "not_covered"],
            "quality_A_above_B": True,
            "unsupported_C_evidence_support_below_5": True,
            "obvious_pair_logical_winner": "A",
            "near_pair_expected": "tie",
        },
    }


def payload(
    kind: str, case: str = "C", swapped: bool = False, long: bool = False
) -> dict:
    """Build mode-independent synthetic Judge inputs with the existing schemas."""
    if kind == "pairwise":
        pair = ["A", "B"] if case == "obvious" else ["A", "A_paraphrase"]
        if swapped:
            pair.reverse()
        result = {
            "question": QUESTION,
            "requirements": REQUIREMENTS,
            "answer_A": ANSWERS[pair[0]],
            "answer_B": ANSWERS[pair[1]],
        }
    else:
        report = ANSWERS[case]
        notes = [
            {
                "text": text,
                "context_kind": "retrieved_research_notes",
                "source_urls": [f"https://synthetic.invalid/note-{index}"],
                "truncated": False,
            }
            for index, text in enumerate(NOTES)
        ]
        if long:
            # Twenty packet items of 1000 characters; relevant facts occur early, midway and late.
            notes = []
            for index in range(20):
                fact = (
                    NOTES[0]
                    if index == 0
                    else NOTES[1]
                    if index == 10
                    else NOTES[2]
                    if index == 19
                    else "虚构补充背景：本段不引入新的配送容量、费用、时段或安全结论。"
                )
                text = fact + (f"背景段{index:02}仅供长度验收。" * 100)
                notes.append(
                    {
                        "text": text[:1000].ljust(1000, "。"),
                        "context_kind": "retrieved_research_notes",
                        "source_urls": [f"https://synthetic.invalid/long-{index}"],
                        "truncated": False,
                    }
                )
            report = (
                ANSWERS["A"] + "\n" + "补充叙述不改变上述试验事实和限定。" * 2000
            )[:20000].ljust(20000, "。")
        result = {
            "question": QUESTION,
            "requirements": REQUIREMENTS,
            "final_report": report,
        }
        if kind == "quality" or long:
            result.update(
                rubric=json.loads(
                    (ROOT / "rubric_v1.json").read_text(encoding="utf-8")
                ),
                evidence_context_kind="retrieved_research_notes",
                evidence_context=notes,
            )
    # Identical format guidance across all methods, including json_mode's explicit JSON requirement.
    result["output_json_schema"] = SCHEMAS[kind].model_json_schema()
    return result


def snapshot() -> dict[str, str]:
    """Hash protected files without parsing or invoking any benchmark question."""
    manifest = json.loads((ROOT / "manifest_v1.json").read_text(encoding="utf-8"))
    paths = ["questions_v1.jsonl", "manifest_v1.json", *manifest["artifacts_sha256"]]
    protected = {f"eval/{path}": sha256((ROOT / path).read_bytes()) for path in paths}
    project = ROOT.parent
    for path in [*project.glob("backend/src/**/*.py"), ROOT / "evaluate.py"]:
        protected[path.relative_to(project).as_posix()] = sha256(path.read_bytes())
    if (project / ".env").exists():
        protected[".env"] = sha256((project / ".env").read_bytes())
    return protected


def acceptance_hashes() -> dict:
    """Hash fixtures, schemas, and unchanged Judge prompts used in acceptance."""

    def encode(value: dict) -> bytes:
        return json.dumps(value, sort_keys=True, ensure_ascii=False).encode()

    return {
        "fixture_sha256": sha256(encode(fixtures())),
        "schema_sha256": {
            kind: sha256(encode(schema.model_json_schema()))
            for kind, schema in SCHEMAS.items()
        },
        "prompt_sha256": {
            kind: sha256((ROOT / f"prompts/{kind}_judge_v1.txt").read_bytes())
            for kind in SCHEMAS
        },
        "rubric_sha256": sha256((ROOT / "rubric_v1.json").read_bytes()),
    }
