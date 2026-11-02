from pathlib import Path

from watermark_detector.train import build_parser, split_records


def _records(n_sources):
    recs = []
    for i in range(n_sources):
        recs.append((Path(f"clean/s{i}.png"), 0))
        recs.append((Path(f"watermarked/s{i}.png"), 1))
    return recs


def test_split_keeps_each_source_in_one_split():
    splits = split_records(_records(20), seed=1)
    stems = [{p.stem for p, _ in recs} for recs in splits.values()]
    assert stems[0].isdisjoint(stems[1]) and stems[0].isdisjoint(stems[2])
    assert stems[1].isdisjoint(stems[2])
    assert sum(len(r) for r in splits.values()) == 40


def test_split_is_deterministic_and_balanced():
    a = split_records(_records(20), seed=3)
    b = split_records(_records(20), seed=3)
    assert a == b
    for recs in a.values():
        labels = [lab for _, lab in recs]
        assert labels.count(0) == labels.count(1) > 0


def test_cli_defaults_follow_config():
    args = build_parser().parse_args([])
    assert args.epochs == 20 and args.batch_size == 32 and args.dataset_size == 450
