from watermark_detector.predict import build_parser, verdict


def test_verdict_against_watermarked_reference():
    assert verdict(0.9, "watermarked") == "watermarked"
    assert verdict(0.1, "watermarked") == "clean"


def test_verdict_against_clean_reference():
    assert verdict(0.9, "clean") == "clean"
    assert verdict(0.1, "clean") == "watermarked"


def test_parser_requires_reference():
    args = build_parser().parse_args(["a.jpg", "--reference", "r.jpg"])
    assert args.reference_label == "watermarked"
