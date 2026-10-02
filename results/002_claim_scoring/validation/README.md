# 002 judge validation material

- `pilot.jsonl` (20) and `validation.jsonl` (60): readouts with shuffled candidate lists, drawn by
  `scripts/002_claim_scoring/make_sets.py` (seed 20261004), stratified by reader × arm (validation).
- `key.jsonl`: each readout's table trajectory (start, z, answer). Kept separate so coding is blind.
- `author_codes.jsonl`: the author's (Claude's) role codes for the 60 validation readouts, written from the text and
  candidates only, before any judge output on them was seen.
  - Coding was strict: a role given different colours in different places (e.g. across an Oracle sample's bullet
    answers) is "conflicting", even when one colour dominates.
  - A role given only non-candidate values (e.g. "blue" when blue is not a candidate, or "0x01", "Monday") is "none".
  - "cannot tell" means the text states no value for the role.
