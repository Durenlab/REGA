# docs/

This directory will contain the Sphinx documentation source for REGA.

Setup is deferred to Phase D. When configured, it will use:
- [Sphinx](https://www.sphinx-doc.org/) for HTML generation
- [MyST-Parser](https://myst-parser.readthedocs.io/) for Markdown support
- [sphinx-rtd-theme](https://sphinx-rtd-theme.readthedocs.io/) for styling

To build the docs (once Phase D is complete):

```bash
pip install ".[docs]"
sphinx-build -b html docs/ docs/_build/html
```
