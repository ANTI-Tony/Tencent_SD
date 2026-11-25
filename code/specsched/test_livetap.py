"""CPU unit test of the live-tap ring buffer: writer/reader round trip, wrap-around, lapping, chunk decode."""
import os
import struct
import tempfile

import numpy as np

from livetap import CHUNK_HDR, K_CHUNK, K_FINISH, Ring, _id_payload, decode_chunk, decode_id


def chunk(rid, start, plen, tok, aux):
    rb = rid.encode()
    hdr = CHUNK_HDR.pack(len(rb), len(tok), start, plen, aux.shape[1], 0) + rb + bytes((-len(rb)) % 8)
    return [np.frombuffer(hdr, dtype=np.uint8), tok.view(np.uint8).reshape(-1), aux.reshape(-1).view(np.uint8)]


def test_roundtrip_and_wrap():
    path = os.path.join(tempfile.mkdtemp(), "ring")
    w = Ring(path, capacity=64 * 1024, create=True)
    r = Ring(path)
    hidden = 48
    got = []
    for i in range(300):   # ~300 * (16 + 48 + 20*4 + 20*48*2) B = ~600 KB  >> 64 KB capacity: wraps ~9 times
        tok = np.arange(i, i + 20, dtype=np.int32)
        aux = np.full((20, hidden), i, dtype=np.int16)
        assert w.write(K_CHUNK, chunk(f"req-{i}", 0, 5, tok, aux))
        assert w.write(K_FINISH, _id_payload(f"req-{i}"))
        if i % 7 == 0:
            for kind, payload in r.read():
                got.append((kind, payload))
    for kind, payload in r.read():
        got.append((kind, payload))
    assert r.lapped == 0, "reader kept up but was lapped"
    chunks = [decode_chunk(p) for k, p in got if k == K_CHUNK]
    fins = [decode_id(p) for k, p in got if k == K_FINISH]
    assert len(chunks) == 300 and len(fins) == 300
    for i, (rid, n, start, plen, hid, tok, aux) in enumerate(chunks):
        assert rid == f"req-{i}" and n == 20 and start == 0 and plen == 5 and hid == hidden
        assert tok.tolist() == list(range(i, i + 20)) and int(aux[3, 7]) == i
        assert fins[i] == rid
    # a slow reader is lapped and resynchronises at the oldest intact record
    r2 = Ring(path)
    r2.rewind_to_oldest()
    before = r2.rseq
    for i in range(300, 600):
        w.write(K_CHUNK, chunk(f"req-{i}", 0, 5, np.zeros(20, np.int32), np.zeros((20, hidden), np.int16)))
    out = list(r2.read(max_records=10_000))
    assert r2.lapped >= 1 and r2.rseq == w.write_seq and before < r2.rseq
    assert all(seq_ok for seq_ok in [k == K_CHUNK for k, _ in out])
    print("ring ok: records", w.records, "bytes", w.nbytes, "lapped", r2.lapped, "read-after-lap", len(out))


if __name__ == "__main__":
    test_roundtrip_and_wrap()
