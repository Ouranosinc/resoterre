import datetime
import json
import tempfile
from pathlib import Path

from resoterre import snakemake_utils


def test_merge_manifests():
    with tempfile.TemporaryDirectory() as tmp_dir:
        with Path(tmp_dir, "manifest1.txt").open("w") as f:
            f.write("file1\nfile2\nfile3\n")
        with Path(tmp_dir, "manifest2.txt").open("w") as f:
            f.write("file3\nfile4\n\nfile5\n")
        output_manifest = Path(tmp_dir, "merged_manifest.txt")
        snakemake_utils.merge_manifests(
            inputs=[Path(tmp_dir, "manifest1.txt"), Path(tmp_dir, "manifest2.txt")], output=output_manifest
        )
        with Path(output_manifest).open() as f:
            lines = f.read().splitlines()
        assert lines == ["file1", "file2", "file3", "file4", "file5"]


def test_merge_logs():
    with tempfile.TemporaryDirectory() as tmp_dir:
        with Path(tmp_dir, "log1.txt").open("w") as f:
            f.write("[DEBUG] (something) message\n[INFO] (something) message\n[WARNING] (something) message\n")
        with Path(tmp_dir, "log2.txt").open("w") as f:
            f.write("[DEBUG] (something else) message\n[INFO] (something else) message\n")
        output_manifest = Path(tmp_dir, "merged_manifest.txt")
        snakemake_utils.merge_logs(
            inputs=[Path(tmp_dir, "log1.txt"), Path(tmp_dir, "log2.txt")],
            output=output_manifest,
            search_patterns=["[WARNING]"],
            purge=True,
        )
        with Path(output_manifest).open() as f:
            lines = f.read().splitlines()
        assert not Path(tmp_dir, "log2.txt").exists()
        assert lines[2] == "[WARNING] (something) message"


def test_merge_logs_from_json_manifest():
    with tempfile.TemporaryDirectory() as tmp_dir:
        with Path(tmp_dir, "log1.txt").open("w") as f:
            f.write("[DEBUG] (something) message\n[INFO] (something) message\n[WARNING] (something) message\n")
        with Path(tmp_dir, "log2.txt").open("w") as f:
            f.write("[DEBUG] (something else) message\n[INFO] (something else) message\n")
        with Path(tmp_dir, "manifest_log1.json").open("w") as f:
            json.dump({"log_file": str(Path(tmp_dir, "log1.txt"))}, f)
        with Path(tmp_dir, "manifest_log2.json").open("w") as f:
            json.dump({"log_file": str(Path(tmp_dir, "log2.txt"))}, f)
        output_manifest = Path(tmp_dir, "merged_manifest.txt")
        snakemake_utils.merge_logs(
            inputs=[Path(tmp_dir, "manifest_log1.json"), Path(tmp_dir, "manifest_log2.json")],
            output=output_manifest,
            search_patterns=["[WARNING]"],
            purge=True,
            from_json_manifest=True,
        )
        with Path(output_manifest).open() as f:
            lines = f.read().splitlines()
        assert not Path(tmp_dir, "log2.txt").exists()
        assert lines[2] == "[WARNING] (something) message"


def test_decode_period_string():
    start_datetime, end_datetime = snakemake_utils.decode_period_string("2026010100_2026013123")
    assert start_datetime.year == 2026
    assert start_datetime.month == 1
    assert start_datetime.day == 1
    assert start_datetime.hour == 0
    assert end_datetime.year == 2026
    assert end_datetime.month == 1
    assert end_datetime.day == 31
    assert end_datetime.hour == 23


def test_split_period():
    start_datetime = datetime.datetime(2026, 1, 1, 0)
    end_datetime = datetime.datetime(2026, 1, 31, 23)
    period_strings = snakemake_utils.split_period(
        start_datetime=start_datetime, end_datetime=end_datetime, batch_size=32, datetime_format="%Y%m%d%H", hours=12
    )
    assert len(period_strings) == 2
    assert period_strings[0] == "2026010100_2026011612"
    assert period_strings[1] == "2026011700_2026020112"


def test_split_glob():
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(5):
            with Path(tmp_dir, f"file{i}.txt").open("w") as f:
                f.write(f"file {i}\n")
        file_paths = snakemake_utils.split_glob(
            search_path=Path(tmp_dir),
            glob_pattern="file*.txt",
            batch_size=2,
            output_directory=Path(tmp_dir, "manifests"),
            manifest_prefix="test",
        )
        assert len(file_paths) == 3
        assert set(file_paths[0]) == {Path(tmp_dir, "file0.txt"), Path(tmp_dir, "file1.txt")}
        assert set(file_paths[1]) == {Path(tmp_dir, "file2.txt"), Path(tmp_dir, "file3.txt")}
        assert set(file_paths[2]) == {Path(tmp_dir, "file4.txt")}
        assert Path(tmp_dir, "manifests", "test_00000000.txt").is_file()
        assert Path(tmp_dir, "manifests", "test_00000001.txt").is_file()
        assert Path(tmp_dir, "manifests", "test_00000002.txt").is_file()
