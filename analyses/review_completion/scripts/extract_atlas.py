"""Project the public author count matrix; retain all cells before depth filtering."""

from pathlib import Path
import hashlib, json, time
import h5py
import numpy as np
import pandas as pd
from h5_stream import StreamedVector

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "inputs"
alpha = [f"ITGA{i}" for i in range(1, 12)] + [
    "ITGAV",
    "ITGAL",
    "ITGAM",
    "ITGAX",
    "ITGAD",
    "ITGA2B",
    "ITGAE",
]
genes = alpha + [f"ITGB{i}" for i in range(1, 9)]
edges = (
    [(a, "ITGB1") for a in alpha[:12]]
    + [(a, "ITGB2") for a in ["ITGAL", "ITGAM", "ITGAX", "ITGAD"]]
    + [
        ("ITGAV", "ITGB3"),
        ("ITGA2B", "ITGB3"),
        ("ITGA6", "ITGB4"),
        ("ITGAV", "ITGB5"),
        ("ITGAV", "ITGB6"),
        ("ITGA4", "ITGB7"),
        ("ITGAE", "ITGB7"),
        ("ITGAV", "ITGB8"),
    ]
)
u = pd.read_csv(IN / "uniprot_alpha_subunits.tsv", sep="\t").set_index("Gene Names (primary)")
assert all(a in u.index for a in alpha)
pd.DataFrame(
    [
        dict(
            alpha=a,
            beta=b,
            alpha_uniprot=u.loc[a, "Entry"],
            source_url="https://www.uniprot.org/uniprotkb/" + u.loc[a, "Entry"] + "/entry",
            subunit_annotation=u.loc[a, "Subunit structure"],
        )
        for a, b in edges
    ]
).to_csv(IN / "integrin_edges_with_provenance.csv", index=False)
meta = pd.read_csv(IN / "GSE178341_metatables.csv.gz").set_index("cellID")
cl = pd.read_csv(IN / "GSE178341_cluster.csv.gz").set_index("sampleID")
meta.index = meta.index.str.strip()
cl.index = cl.index.str.strip()
start = time.perf_counter()
with h5py.File(IN / "GSE178341_full.h5") as f:
    m = f["matrix"]
    names = m["features/name"][:].astype(str)
    ids = m["barcodes"][:].astype(str)
    ids = np.char.strip(ids)
    assert len(np.unique(ids)) == len(ids)
    meta = meta.loc[ids].join(cl[["clTopLevel", "clMidwayPr", "cl295v11SubShort"]])
    print(meta.groupby(["SPECIMEN_TYPE", "clMidwayPr"]).size().to_string(), flush=True)
    row = np.array([np.flatnonzero(names == g)[0] for g in genes])
    assert all(np.sum(names == g) == 1 for g in genes)
    lookup = np.full(len(names), -1, dtype=int)
    lookup[row] = np.arange(len(genes))
    ptr = m["indptr"][:]
    X = np.zeros((len(ids), len(genes)), dtype=np.int64)
    L = np.zeros(len(ids), dtype=np.int64)
    data = StreamedVector(m["data"], IN / "GSE178341_full.h5", ROOT.parents[2] / "review_cache")
    indices = StreamedVector(
        m["indices"], IN / "GSE178341_full.h5", ROOT.parents[2] / "review_cache"
    )
    for lo in range(0, len(ids), 4000):
        hi = min(len(ids), lo + 4000)
        s, e = ptr[lo], ptr[hi]
        ix = indices[s:e]
        v = data[s:e]
        assert np.all(v >= 0) and np.all(v == np.floor(v)), "Count layer is not nonnegative integer"
        ci = np.repeat(np.arange(hi - lo), np.diff(ptr[lo : hi + 1]))
        L[lo:hi] = np.bincount(ci, weights=v, minlength=hi - lo).astype(np.int64)
        j = lookup[ix]
        keep = j >= 0
        np.add.at(X, (lo + ci[keep], j[keep]), v[keep].astype(np.int64))
        if lo % 40000 == 0:
            print("cells", hi, "seconds", round(time.perf_counter() - start, 1), flush=True)
    assert np.all(X.sum(1) <= L)
    np.savez_compressed(
        IN / "atlas_integrins_all_cells.npz",
        counts=X,
        depth=L,
        genes=np.array(genes),
        cell_id=ids,
        patient=meta.PID.to_numpy(str),
        tissue=meta.SPECIMEN_TYPE.to_numpy(str),
        celltype=meta.clMidwayPr.to_numpy(str),
        subtype=meta.cl295v11SubShort.to_numpy(str),
        MMR=meta.MMRStatus.to_numpy(str),
    )
    meta.reset_index().rename(columns={"cellID": "cell_id"}).to_csv(
        IN / "atlas_joined_metadata.csv.gz", index=False
    )
    digest = hashlib.file_digest(open(IN / "GSE178341_full.h5", "rb"), "sha256").hexdigest()
    (IN / "atlas_projection_provenance.json").write_text(
        json.dumps(
            dict(
                source="GSE178341",
                url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE178nnn/GSE178341/suppl/GSE178341_crc10x_full_c295v4_submit.h5",
                sha256=digest,
                bytes=(IN / "GSE178341_full.h5").stat().st_size,
                cells=len(ids),
                features=len(names),
                integrin_genes=genes,
                edges=edges,
                all_nonzero_values_integer=True,
                denominator="sum of all 43113 author feature rows; no normalization",
                extraction_seconds=time.perf_counter() - start,
            ),
            indent=2,
        )
        + "\n"
    )
    print("DONE", X.shape, flush=True)
