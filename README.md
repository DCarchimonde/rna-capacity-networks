# RNA capacity networks

Code and data for **When pairwise expression is insufficient for RNA capacity networks**
by Siyuan Tong and Yuxuan Wang.

This repository contains the final analysis code, count projections, reference results
and figure inputs for the manuscript. The analysis directory names match the paths
cited in the paper and Supplementary Information.

## Run the analyses

Use Python 3.12 on Linux or WSL, with a C++17 compiler supporting OpenMP. Numerical
kernels compile locally on first use. No GPU is required.

```bash
git clone https://github.com/DCarchimonde/rna-capacity-networks.git
cd rna-capacity-networks
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --stage integrity
python reproduce.py --stage checks
python reproduce.py --stage figures
```

The integrity check compares the supplied files with `MANIFEST_SHA256.json`; run it
before regenerating outputs. The numerical checks include mathematical identities,
independent optimisation comparisons and consistency checks against the manuscript.
The figures stage rebuilds all 12 final figures from the supplied results.

One large reference archive is stored as two binary parts in `data/reference_parts/`.
The entry point automatically joins and verifies them before running an analysis or
integrity check. The restored NPZ is byte-identical to the validated reference file;
no separate download or additional package is needed.

For the complete computational replay from the supplied count projections:

```bash
python reproduce.py --stage full
```

The full workflow runs 43 ordered steps and can take tens of minutes or longer.
Allow at least 8 GB RAM and several GB of working space. All required count
projections and reference outputs are included. No network connection is needed
after installing dependencies.

Additional modes, individual commands and the scope of each calculation are
documented in [REPRODUCE.txt](REPRODUCE.txt). To inspect the full execution order:

```bash
python reproduce.py --stage full --list
```

## Repository contents

| Location | Contents |
| --- | --- |
| `reproduce.py` | Entry point for integrity checks, analyses and figures |
| `analyses/` | Experimental modules, specifications, inputs and results |
| `analysis/` and `integration/` | Cross-module diagnostics and final numerical refinement |
| `data/` | Count projections and final figure inputs |
| `source_data/` | Manuscript-facing numerical tables |
| `figures/` | Eight main and four supplementary figures |
| `review/` | Mathematical, numerical and data validation records |
| [DATA_CODE_MAP.txt](DATA_CODE_MAP.txt) | Maps manuscript analyses and figures to code |
| [SOURCE_DATA_INDEX.txt](SOURCE_DATA_INDEX.txt) | Table paths, row counts and field names |
| [FIELD_GUIDE.txt](FIELD_GUIDE.txt) | Units, missing values and shared field definitions |
| [VALIDATION.txt](VALIDATION.txt) | Reproduction checks completed for this release |

The final code and scientific outputs are retained in one directory tree. Diagnostic
analyses discussed in the Supplementary Information are included alongside the main
analyses.

## Data and reproducibility

The public datasets are [GSE178341](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178341)
and [GSE166555](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE166555).
The supplied projections retain raw integer gene counts and complete RNA depths.
They are not normalised expression matrices.

The full replay starts from these projections. It does not download or re-extract
the original 1.2 GB atlas HDF5 or the second cohort's GEO matrices. Source accessions,
checksums and optional extraction commands are in [DATA_SOURCES.txt](DATA_SOURCES.txt).

The full computational workflow has been checked against the manuscript's reference
results. All 2,048 paired permutation p-values and decisions match, and all 12 final
figures match the reference rendering. Runtime measurements and file metadata can
differ across machines. See [VALIDATION.txt](VALIDATION.txt) and
[review/reproduction_validation.json](review/reproduction_validation.json) for the
checks, numerical tolerances and scope limits.

The paired permutation test uses fixed integer category labels to preserve the
seeded allocation order when floating-point calculations differ at roundoff level.
The labels, source lookup digests and implementation details are documented in
[REPRODUCE.txt](REPRODUCE.txt).

## Citation and licence

Software citation metadata are supplied in [CITATION.cff](CITATION.cff). Use the
specific release or commit used in your analysis when citing the repository.

Original analysis code is available under the [MIT licence](LICENSE). The unchanged
CS-CORE implementation retains its original
[MIT licence](analyses/strong_comparators/vendor/LICENSE).
Public source data retain their original terms and attribution; see
[DATA_SOURCES.txt](DATA_SOURCES.txt).
