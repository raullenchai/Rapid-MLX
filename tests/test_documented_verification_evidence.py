from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTED_ARTIFACTS = {
    "scripts/README.qwen38-converter.md": (
        "scripts/qwen38_streaming_convert.py",
        "scripts/synthetic_qwen38_fixture.py",
        "scripts/verify_synth_conversion.py",
        "scripts/test_fail_closed_guards.py",
    ),
    "evals/results/SUFFIX_POC_REPORT.md": (
        "rapid_mlx/speculative/suffix_decoding.py",
        "scripts/bench_suffix_decoding.py",
        "evals/results/suffix_poc_sweep.json",
    ),
    "reports/benchmarks/readme-refresh/summary.md": (
        "results-20260606-152047.json",
        "results-20260606-152654.json",
        "results-20260606-153334.json",
        "results-20260609-070403.json",
    ),
}


def test_documented_verification_evidence_exists() -> None:
    missing = []

    for document, references in DOCUMENTED_ARTIFACTS.items():
        document_path = REPO_ROOT / document
        document_text = document_path.read_text()

        for reference in references:
            if reference not in document_text:
                missing.append(f"{document} no longer documents {reference}")
                continue

            artifact = (
                document_path.parent / reference
                if "/" not in reference
                else REPO_ROOT / reference
            )
            if not artifact.is_file():
                missing.append(f"{document} references missing artifact {reference}")

    assert not missing, "\n".join(missing)
