"""List or copy selected GEO supplementary TSV members without reading scores."""

import argparse
import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd
from prepare_counts import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir")
    args = p.parse_args()
    manifest = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    names = manifest.loc[manifest.selected == "1", "file_name"].tolist()
    with tarfile.open(args.archive, "r:*") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        lookup = {}
        for member in members:
            lookup.setdefault(Path(member.name).name, []).append(member)
        inventory = [{"member": m.name, "bytes": m.size} for m in members]
        print(json.dumps({"archive_sha256": sha(args.archive), "inventory": inventory}, indent=2))
        if args.out_dir:
            absent = [name for name in names if len(lookup.get(name, [])) != 1]
            if absent:
                raise ValueError(
                    "Selected filenames absent or ambiguous; audit inventory before updating manifest: "
                    + ", ".join(absent)
                )
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            for name in names:
                with archive.extractfile(lookup[name][0]) as src, (out / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            (out / "archive_inventory.json").write_text(
                json.dumps(
                    {
                        "archive_sha256": sha(args.archive),
                        "selected_files": names,
                        "inventory": inventory,
                    },
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
