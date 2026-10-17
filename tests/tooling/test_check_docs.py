import subprocess

from check_docs import check_markdown, destinations, heading_anchors, main, markdown_files


def test_relative_root_paths_decoding_and_heading_links(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "source file.py").write_text("pass\n")
    (tmp_path / "target.md").write_text("# Model `API`\n\n## Loss\n\n## Loss\n")
    source = tmp_path / "docs/README.md"
    source.write_text(
        "# Start\n[relative](../target.md#model-api)\n[root](/target.md#loss-1)\n"
        "[file](../source%20file.py)\n[directory](../docs/)\n[self](#start)\n"
        "[reference][target]\n\n[target]: ../target.md#loss\n"
    )
    assert check_markdown(source, tmp_path) == []


def test_missing_file_anchor_and_escaping_path_are_reported_with_lines(tmp_path):
    (tmp_path / "target.md").write_text("# Real heading\n")
    source = tmp_path / "README.md"
    source.write_text(
        "# Home\n[missing](missing.md)\n[anchor](target.md#wrong)\n[escape](../outside.md)\n"
    )
    errors = check_markdown(source, tmp_path)
    assert len(errors) == 3
    assert "README.md:2: missing.md: destination does not exist" in errors[0]
    assert "heading anchor does not exist" in errors[1]
    assert "escapes repository" in errors[2]


def test_fenced_examples_and_remote_destinations_are_ignored(tmp_path):
    source = tmp_path / "README.md"
    source.write_text(
        "# Home\n```markdown\n[example](missing.md)\n```\n"
        "~~~\n[other](also-missing.md)\n~~~\n"
        "[remote](https://example.invalid/missing)\n[email](mailto:me@example.invalid)\n"
        "[network](//example.invalid/file)\n"
    )
    assert check_markdown(source, tmp_path) == []


def test_images_reference_destinations_and_angle_paths():
    markdown = '![image](plot.png)\n[link](<some file.md> "Title")\n[ref]: target.md "Title"\n'
    assert list(destinations(markdown)) == [(1, "plot.png"), (2, "some file.md"), (3, "target.md")]


def test_heading_slugs_include_duplicates_and_ignore_code_examples():
    markdown = "# A *model*\n## A model\n```\n# Fake\n```\n## [Loss](loss.md)\n"
    assert heading_anchors(markdown) == {"a-model", "a-model-1", "loss"}


def test_cli_uses_git_inventory_and_ignores_local_artifacts(tmp_path, capsys):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("build/\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build/README.md").write_text("[broken](missing)\n")
    source = tmp_path / "README.md"
    source.write_text("# Home\n")
    assert markdown_files(tmp_path) == [source]
    assert main(["--root", str(tmp_path)]) == 0
    assert "Checked 1 Markdown files" in capsys.readouterr().out
    source.write_text("[broken](missing.md)\n")
    assert main(["--root", str(tmp_path)]) == 1
    assert "README.md:1" in capsys.readouterr().out
