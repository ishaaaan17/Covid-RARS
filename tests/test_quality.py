from covid_rars.quality import assign_quality_flag


def test_short_cough_flag():
    flag, reasons = assign_quality_flag(
        modality="cough",
        duration_sec=0.2,
        silence_ratio=0.1,
        clipping_ratio=0.0,
    )

    assert flag == "short"
    assert "duration" in reasons


def test_mostly_silence_flag():
    flag, reasons = assign_quality_flag(
        modality="speech",
        duration_sec=2.0,
        silence_ratio=0.9,
        clipping_ratio=0.0,
    )

    assert flag == "mostly_silence"
    assert "silence" in reasons


def test_ok_flag():
    flag, reasons = assign_quality_flag(
        modality="breath",
        duration_sec=2.0,
        silence_ratio=0.2,
        clipping_ratio=0.0,
    )

    assert flag == "ok"
    assert reasons == []
