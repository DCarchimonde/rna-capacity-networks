SUPPLEMENTARY CODE AND DATA
When pairwise expression is insufficient for RNA capacity networks
Siyuan Tong and Yuxuan Wang

This archive contains the analysis code, count projections, numerical results
and figure inputs supporting the manuscript and Supplementary Information.
It is one submission release. The analysis directories cover different experiments;
their names match the paths cited in the manuscript.

START HERE
1. Use Python 3.12 on Linux or WSL and a C++17 compiler supporting OpenMP.
2. Install dependencies: python -m pip install -r requirements.txt
3. Verify the supplied files: python reproduce.py --stage integrity
4. Run numerical checks: python reproduce.py --stage checks
5. Rebuild final figures: python reproduce.py --stage figures

In the GitHub distribution, the entry point automatically restores one large
reference archive from the verified binary parts in data/reference_parts/.
No separate data download is required.

Complete computational replay from supplied count projections:
    python reproduce.py --stage full

The replay compiles numerical kernels locally, uses the recorded seeds, and
updates analysis outputs in this directory. No GPU, paid service or network
connection is needed after installing dependencies. Use a fresh extracted copy
to compare a replay with the supplied reference results.

CONTENTS
REPRODUCE.txt              Commands, execution order and scope of each stage.
DATA_CODE_MAP.txt          Analysis modules and figure-to-code mapping.
SOURCE_DATA_INDEX.txt      Table paths, row counts and column names.
source_data_map.json       Maps table copies to canonical analysis results.
FIELD_GUIDE.txt            Common fields, units and missing values.
DATA_SOURCES.txt           Accessions, extraction and data-use information.
requirements.txt          Tested Python dependency versions.
MANIFEST_SHA256.json       Sizes and SHA-256 digests of the release files.
VALIDATION.txt             Checks performed for this submission release.
data/                     Count caches and final figure inputs.
analyses/                 Scientific implementations, specifications and results.
source_data/              Manuscript-facing numeric tables.
figures/                  Eight main and four supplementary figures.
review/                   Numerical, data and comparator-coverage checks.
integration/              Propagation of the final numerical refinement.

Count projections retain raw integer gene counts and complete RNA depths.
They are not normalized expression matrices. The original large public files
are identified in the provenance records; DATA_SOURCES.txt explains optional
re-extraction. No manuscript drafts, intermediate project copies or manuscript
editing scripts are included.
