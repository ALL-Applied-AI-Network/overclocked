# CubeVisionStack-v1.1.2-blue-best-candidate

This revision targets blue 10 mm cubes only. It allows multiple blue detections and selects the cube whose live
position is most strongly confirmed across three table-camera observations.

Candidate ranking is deterministic:

1. Lowest three-frame positional spread.
2. Highest mean fitted-silhouette confidence.
3. Greatest separation from another blue candidate for grasp clearance.
4. X/Y ordering only as a final exact-tie breaker.

The placement file is never used for motion targeting or candidate selection. Non-blue objects are ignored by the
colour gate. Candidates below the existing `0.55` confidence threshold, outside the safe table source region, too
close to the bin, or beyond the conservative radial limit cannot be selected. Multiple blue cubes alone are not a
reason to hold.

The physically validated grasp height (`-333 mm`), release height (`-255 mm`), motion speeds, camera calibration,
workspace validation, source-removal verification, and one-retry limit remain unchanged.

## Test locally from WSL

```bash
cd '/mnt/c/Users/glnva/OneDrive/Documents/NVIDIA+Rockwell/overclocked/CubeVisionStack-v1.1.2-blue-best-candidate'
uv sync
uv run python -m unittest -v test_blue_best_candidate.py
uv run run.py --mock --trials 10 --trial-s 3
```

This is a separate package. It contains no `.request.json` and has not been submitted or booked. Do not reuse the
position-validation request file; a new task description and request are required if this revision is submitted.
