# Changelog

## 0.2.0 - 2026-10-31

- Update the A4 recovery, pose checks, and acceptance policy to the current application pipeline.
- Add a folder command that saves annotated photos and per-image comparison results.
- Document the standalone detector workflow and its limits.

## 0.1.0 - 2026-10-08

- Extract the OpenCV A4 detector and reference checks from the foot-measurement application.
- Include colour-boundary recovery for pale and patterned backgrounds.
- Add an application API, JSON serialization, a CLI, and a plane homography helper.
- Package the A4 policy with the Python wheel.
- Include synthetic examples, regression tests, and Linux/Windows CI.
- Document input conventions, default limits, and measurement assumptions.
