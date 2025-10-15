# fca
From Captions to Answers: Enhancing Prophet with Question-Aware Captioning for Knowledge-Based VQA

## Third-party code: Prophet (vendored)

We vendor the upstream Prophet repository under `third_party/prophet` to reuse most of its implementation and extend it with question‑aware captioning.

- Upstream: https://github.com/MILVLG/prophet (default branch: `main`)
- Vendored path: `third_party/prophet`
- License: see `third_party/prophet/LICENSE` (ensure attribution and license notices are preserved when distributing)

### Update Prophet to latest upstream

Use Git subtree to pull updates while keeping all code in this repository.

Option A: one‑off command

```
git subtree pull --prefix third_party/prophet https://github.com/MILVLG/prophet main --squash
```

Option B: via helper script

```
scripts/vendor_sync_prophet.sh # pulls from main by default
```

You can specify a different branch or tag:

```
scripts/vendor_sync_prophet.sh v1.0.0
scripts/vendor_sync_prophet.sh prophet++
```

Notes:

- Squashing keeps the repo small while still allowing future pulls.
- If you make local modifications to `third_party/prophet`, document them clearly in commit messages to ease future merges.
