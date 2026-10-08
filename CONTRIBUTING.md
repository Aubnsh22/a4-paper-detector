# Contributing

Use a branch for changes and open a pull request describing the observed
problem, the change, and the validation performed.

Install the development dependencies and run the tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m build
```

For detector changes, include a regression case with known expected behavior.
Prefer a synthetic image when it can reproduce the problem. Share real images
only when you have permission, and remove personal metadata before upload.

Report Python, OpenCV, and NumPy versions, image dimensions, the returned
reasons, and a minimal reproduction when filing an issue. Avoid changing a
threshold solely to make one image pass; also check ambiguity, missing paper,
cropped corners, and difficult backgrounds.

Keep public examples small and executable. Document changes to coordinate
conventions, acceptance rules, dependencies, and the JSON schema in the
changelog. Contributions are distributed under the repository's MIT license.
