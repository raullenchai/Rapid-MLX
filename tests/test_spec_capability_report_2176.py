from vllm_mlx.model_aliases import AliasProfile, list_profiles
from vllm_mlx.spec_decode.capability import REGISTERED_METHODS, assess_method
from vllm_mlx.spec_decode.report import _quantization, build_report


def test_report_covers_every_alias_and_method():
    report = build_report()
    assert set(report) == set(list_profiles())
    assert all(
        set(entry["methods"]) == set(REGISTERED_METHODS) for entry in report.values()
    )
    assert all(
        {"hf_path", "checkpoint_format", "quantization", "modality"}
        <= set(entry["model"])
        for entry in report.values()
    )


def test_4bit_is_recommendation_evidence_not_capability_ban():
    profile = AliasProfile(hf_path="user/Qwen3.5-9B-abliterated-4bit")
    for method in ("mtp", "dflash", "ddtree", "dspark"):
        assessment = assess_method(profile, method)
        assert assessment.recommendation == "experimental"
        assert assessment.explicit_opt_in is True
        assert assessment.capable is not False


def test_true_verifier_incompatibility_remains_hard_failure():
    hybrid = AliasProfile(hf_path="user/hybrid", is_hybrid=True)
    suffix = assess_method(hybrid, "suffix")
    assert suffix.capable is False
    assert suffix.recommendation == "incompatible"


def test_uncurated_non_hybrid_suffix_is_experimental():
    profile = AliasProfile(hf_path="user/custom-text-model")
    suffix = assess_method(profile, "suffix")
    assert suffix.capable is None
    assert suffix.recommendation == "experimental"
    assert suffix.explicit_opt_in is True


def test_verified_entries_do_not_require_experimental_opt_in():
    profile = AliasProfile(
        hf_path="user/verified-8bit",
        supports_spec_decode=True,
        suffix_decoding_tier="verified",
        supports_dflash=True,
        supports_ddtree=True,
        dflash_draft_model="user/dflash-drafter",
        ddtree_draft_model="user/ddtree-drafter",
        ddtree_speculative_tokens=16,
        ddtree_tree_budget=24,
        mtp_draft_model="user/mtp-head",
    )
    for method in ("suffix", "dflash", "ddtree", "mtp"):
        assessment = assess_method(profile, method)
        assert assessment.recommendation == "verified"
        assert assessment.explicit_opt_in is False
        if method != "mtp":
            assert assessment.capable is True


def test_incomplete_registry_flags_are_not_reported_verified():
    profile = AliasProfile(
        hf_path="user/incomplete-8bit",
        supports_dflash=True,
        supports_ddtree=True,
    )
    for method in ("dflash", "ddtree"):
        assessment = assess_method(profile, method)
        assert assessment.recommendation == "experimental"
        assert assessment.explicit_opt_in is True


def test_quantized_curated_flags_remain_experimental():
    for hf_path in ("user/target-4-bit", "user/target_4bit"):
        profile = AliasProfile(
            hf_path=hf_path,
            supports_dflash=True,
            dflash_draft_model="user/drafter",
            mtp_draft_model="user/mtp-head",
        )
        for method in ("dflash", "mtp"):
            assessment = assess_method(profile, method)
            assert assessment.recommendation == "experimental"
            assert assessment.explicit_opt_in is True


def test_quantization_recognizes_hyphenated_spellings():
    assert _quantization("user/model-4-bit") == "4bit"
    assert _quantization("user/model-8-bit") == "8bit"
    assert _quantization("user/paint4bitmaps") == "not-encoded-in-repo-name"
    assert _quantization("user/model-4bit-instruct") == "4bit"
