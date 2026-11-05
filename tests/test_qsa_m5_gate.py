from __future__ import annotations

import pytest

from rapid_mlx.kernels.qsa_m5_gate import architecture_supported, envelope_supported


@pytest.mark.parametrize("arch", ["applegpu_g17g", "applegpu_g17d", "applegpu_g18p"])
@pytest.mark.parametrize("release", ["26.2", "26.5.2", "27.0"])
def test_tensor_unit_architectures(arch, release):
    assert architecture_supported(arch, release)


@pytest.mark.parametrize(
    "arch",
    [
        "",
        "unknown",
        "Apple M5",
        "applegpu_g15d",
        "applegpu_g14s",
        "applegpu_g16g",
        "applegpu_g17p",
        "applegpu_g17g-extra",
    ],
)
def test_reject_pre_tensor_units_and_unknown_architectures(arch):
    assert not architecture_supported(arch, "26.5.2")


@pytest.mark.parametrize("release", ["", "26", "26.1", "15.6", "invalid", "26.beta"])
def test_reject_unsupported_os(release):
    assert not architecture_supported("applegpu_g17g", release)


@pytest.mark.parametrize(
    "batch,length,training,expected",
    [
        (1, 64, False, True),
        (1, 512, False, True),
        (1, 63, False, False),
        (1, 1, False, False),
        (2, 512, False, False),
        (0, 512, False, False),
        (1, 512, True, False),
    ],
)
def test_prefill_envelope(batch, length, training, expected):
    assert envelope_supported(length, batch_size=batch, training=training) == expected


@pytest.mark.parametrize(
    "enabled,hardware,gpu,expected",
    [
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
        (True, True, True, True),
    ],
)
def test_complete_dispatch_gate(enabled, hardware, gpu, expected):
    from rapid_mlx.kernels.qsa_m5_gate import admit

    assert (
        admit(
            512,
            batch_size=1,
            training=False,
            enabled=enabled,
            hardware=lambda: hardware,
            on_gpu=lambda: gpu,
        )
        == expected
    )


def test_disabled_and_other_batches_never_probe_or_compile():
    from rapid_mlx.kernels.qsa_m5_gate import admit

    def fail():
        pytest.fail("declined dispatch touched device runtime")

    assert not admit(
        512, batch_size=1, training=False, enabled=False, hardware=fail, on_gpu=fail
    )
    assert not admit(
        512, batch_size=2, training=False, enabled=True, hardware=fail, on_gpu=fail
    )
    assert not admit(
        1, batch_size=1, training=False, enabled=True, hardware=fail, on_gpu=fail
    )
    assert not admit(
        512, batch_size=1, training=True, enabled=True, hardware=fail, on_gpu=fail
    )
    assert not admit(
        512,
        batch_size=1,
        training=False,
        enabled=True,
        hardware=lambda: False,
        on_gpu=fail,
    )
