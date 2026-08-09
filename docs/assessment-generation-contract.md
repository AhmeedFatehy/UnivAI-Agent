# Assessment generation contract

## Answer options

- A weekly quiz MCQ has exactly 4 unique, non-blank choices: A-D.
- A midterm or final MCQ has exactly 6 unique, non-blank choices: A-F.
- The correct answer is stored as its letter and must be valid for that paper.
- All distractors must be plausible and grounded in the supplied evidence.

The schema checks these rules after parsing the model response. A malformed
response enters the existing bounded repair attempt and is never published as
valid content.

## Scoring boundary

The Agent authors questions but never calculates a learner's mark. The Exam
system owns the authoritative `+1 / -1 / 0`, floor-at-zero calculation so a
client or generated artifact cannot supply a trusted score.
