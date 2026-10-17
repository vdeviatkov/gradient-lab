# Repository-tool tests

[All tests](../README.md) · [Scripts](../../scripts/README.md)

```bash
python -m pytest tests/tooling
```

[Documentation-checker tests](test_check_docs.py) use temporary Markdown trees to verify relative
links, heading anchors, reference destinations, URL decoding, code-fence exclusion, and rejection
of missing or out-of-repository paths. They do not access the network.
