# Agent instructions

Read README.md, assignment15.py, assignment13.py, test.py, and inputs.yml.
Before editing, propose a bounded plan covering inheritance, x-fastest indexing,
face neighbors, units, shapes, boundary half-distances, and 1D regression checks.
Wait for the student's approval. Implement one method at a time, then show its diff.
Edit only compute_transmissibility, compute_accumulation, and fill_matrices
in assignment15.py. The parser, boundary helper, and assignment13.py are supplied.
Protect tests, data, figures, instructions, environment, and submission infrastructure.
Reuse the inherited time integrator and preserve the course theta convention.
Run python -m unittest -v test.py and git diff --check. Check 1D compatibility,
2D connectivity, and conservation; report failures honestly and never weaken tests.
Compare the final diff with the approved plan. Ask the student to inspect the diff
and rerun checks independently. Stop on conflicting instructions or unexpected changes.
Propose instruction changes for review; never silently change instructions.
No edits during submission. When the user says submit assignment 15, load
.github/skills/submit-assignment/SKILL.md. Submit only assignment15.py.
