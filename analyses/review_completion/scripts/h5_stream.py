"""Read unusually large shuffled-deflate HDF5 chunks within bounded RAM.

GEO's data chunk decompresses to ~4 GiB. Stream its DEFLATE bytes to a
temporary byte-plane file, then undo HDF5 shuffle only for requested slices.
The HDF5 filter pipeline and full decompressed byte length are verified.
"""

from pathlib import Path
import zlib, numpy as np


class StreamedVector:
    def __init__(self, dataset, source, cache):
        self.dtype = dataset.dtype
        self.n = dataset.shape[0]
        self.chunk = dataset.chunks[0]
        self.parts = []
        filters = [
            dataset.id.get_create_plist().get_filter(i)[0]
            for i in range(dataset.id.get_create_plist().get_nfilters())
        ]
        assert filters == [2, 1], filters
        Path(cache).mkdir(exist_ok=True, parents=True)
        for i in range(dataset.id.get_num_chunks()):
            info = dataset.id.get_chunk_info(i)
            assert info.filter_mask == 0
            target = Path(cache) / (dataset.name.rsplit("/", 1)[-1] + f"_{i}.shuffled")
            expected = self.chunk * self.dtype.itemsize
            if not target.exists() or target.stat().st_size != expected:
                decoder = zlib.decompressobj()
                left = info.size
                with open(source, "rb") as inp, open(target, "wb") as out:
                    inp.seek(info.byte_offset)
                    while left:
                        b = inp.read(min(1024 * 1024, left))
                        left -= len(b)
                        while b:
                            out.write(decoder.decompress(b, 4 * 1024 * 1024))
                            b = decoder.unconsumed_tail
                    out.write(decoder.flush())
                    assert decoder.eof
                assert target.stat().st_size == expected, (target.stat().st_size, expected)
                print("streamed", target.name, expected, flush=True)
            self.parts.append(
                np.memmap(target, mode="r", dtype=np.uint8, shape=(self.dtype.itemsize, self.chunk))
            )

    def __getitem__(self, sl):
        start, stop, step = sl.indices(self.n)
        assert step == 1
        out = []
        while start < stop:
            c = start // self.chunk
            end = min(stop, (c + 1) * self.chunk)
            a = (
                self.parts[c][:, start - c * self.chunk : end - c * self.chunk]
                .T.copy()
                .reshape(-1)
                .view(self.dtype)
            )
            out.append(a)
            start = end
        return np.concatenate(out) if len(out) > 1 else out[0]
