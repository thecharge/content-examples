"""Single-host transactional events and immutable, deduplicated tensor snapshots.

Committed snapshots, rather than repeated floating-point training, recover state.
SQLite keeps objects, roots, evidence and the event in the same durable commit.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sqlite3

import torch


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class StorageLimits:
    max_bytes: int = 256 * 1024 * 1024
    max_events: int = 10000
    max_objects: int = 100000
    max_event_bytes: int = 1024 * 1024
    retained_revisions: int = 8

    def __post_init__(self):
        if min(asdict(self).values()) < 1:
            raise ValueError("Storage limits must be positive")


class Ledger:
    def __init__(self, path, limits=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limits = limits or StorageLimits()
        self.db = sqlite3.connect(path, isolation_level=None, timeout=30)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS objects(hash TEXT PRIMARY KEY, kind TEXT NOT NULL,
                data BLOB NOT NULL, size INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS events(revision INTEGER PRIMARY KEY,
                id TEXT UNIQUE NOT NULL, request TEXT NOT NULL, kind TEXT NOT NULL,
                payload TEXT NOT NULL, snapshot TEXT NOT NULL, valid INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE IF NOT EXISTS dependencies(event TEXT NOT NULL, support TEXT NOT NULL,
                PRIMARY KEY(event,support));
            CREATE TABLE IF NOT EXISTS roots(name TEXT PRIMARY KEY, snapshot TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        """)
        saved = self.db.execute(
            "SELECT value FROM metadata WHERE key='limits'"
        ).fetchone()
        config = canonical(asdict(self.limits))
        if saved and saved[0] != config:
            raise ValueError("Reopen with the original storage limits")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES('limits',?)", (config,))

    @property
    def revision(self):
        return self.db.execute(
            "SELECT COALESCE(MAX(revision),0) FROM events"
        ).fetchone()[0]

    def _put(self, kind, data):
        identity = digest(kind.encode() + b"\0" + data)
        self.db.execute(
            "INSERT OR IGNORE INTO objects VALUES(?,?,?,?)",
            (identity, kind, data, len(data)),
        )
        return identity

    def _get(self, identity, kind=None):
        row = self.db.execute(
            "SELECT kind,data FROM objects WHERE hash=?", (identity,)
        ).fetchone()
        if row is None or digest(row[0].encode() + b"\0" + row[1]) != identity:
            raise ValueError("Missing or corrupt artifact: " + identity)
        if kind is not None and row[0] != kind:
            raise ValueError("Artifact kind mismatch")
        return row

    def encode(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            # Include dtype and shape in the address; equal bytes alone are insufficient.
            header = canonical(
                dict(dtype=str(value.dtype), shape=list(value.shape))
            ).encode()
            raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
            return {
                "tensor": self._put(
                    "tensor", len(header).to_bytes(4, "big") + header + raw
                )
            }
        if isinstance(value, dict):
            return {
                "dict": [[self.encode(k), self.encode(v)] for k, v in value.items()]
            }
        if isinstance(value, (tuple, list)):
            return {
                "tuple" if isinstance(value, tuple) else "list": [
                    self.encode(v) for v in value
                ]
            }
        if isinstance(value, float) and not __import__("math").isfinite(value):
            return {"float": repr(value)}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise ValueError("Unsupported snapshot type: " + type(value).__name__)

    def decode(self, value):
        if not isinstance(value, dict):
            return value
        if "tensor" in value:
            _, blob = self._get(value["tensor"], "tensor")
            size = int.from_bytes(blob[:4], "big")
            header = json.loads(blob[4 : 4 + size])
            dtype = getattr(torch, header["dtype"].removeprefix("torch."))
            raw = bytearray(blob[4 + size :])
            if not raw:
                return torch.empty(header["shape"], dtype=dtype)
            # Every decode owns its tensor. Deduplication does not tie mutable state.
            return torch.frombuffer(raw, dtype=dtype).clone().reshape(header["shape"])
        if "dict" in value:
            return {self.decode(k): self.decode(v) for k, v in value["dict"]}
        if "float" in value:
            return float(value["float"])
        key = "tuple" if "tuple" in value else "list"
        result = [self.decode(v) for v in value[key]]
        return tuple(result) if key == "tuple" else result

    def result(self, event_id, request):
        row = self.db.execute(
            "SELECT request,payload FROM events WHERE id=?", (event_id,)
        ).fetchone()
        if row is None:
            return None
        if row[0] != digest(canonical(request).encode()):
            raise ValueError("Event identity reused with different input")
        return json.loads(row[1])

    def commit(
        self,
        event_id,
        kind,
        request,
        state,
        expected,
        *,
        result=None,
        supports=(),
        invalidates=(),
    ):
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Stable nonempty event identity required")
        request_hash = digest(canonical(request).encode())
        payload = canonical(
            dict(
                request=request, result=result, supports=sorted(set(supports)), schema=1
            )
        )
        if len(payload.encode()) > self.limits.max_event_bytes:
            raise ValueError("Event byte ceiling exceeded")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            old = self.result(event_id, request)
            if old is not None:
                self.db.execute("ROLLBACK")
                return old
            if self.revision != expected:
                raise ValueError("Stale model revision")
            if self.revision >= self.limits.max_events:
                raise ValueError(
                    "Event ceiling exhausted; rotate into a new retained branch"
                )
            for support in set(supports):
                found = self.db.execute(
                    "SELECT valid FROM events WHERE id=?", (support,)
                ).fetchone()
                if not found or not found[0]:
                    raise ValueError("Missing or invalidated provenance: " + support)
            affected = set()
            for identity in invalidates:
                affected.update(self.invalidate(identity))
            if affected:
                payload = canonical(
                    dict(
                        request=request,
                        result=result,
                        supports=sorted(set(supports)),
                        invalidates=sorted(affected),
                        schema=1,
                    )
                )
            if len(payload.encode()) > self.limits.max_event_bytes:
                raise ValueError("Correction exceeds the event byte ceiling")
            snapshot = self._put("manifest", canonical(self.encode(state)).encode())
            revision = expected + 1
            self.db.execute(
                "INSERT INTO events(revision,id,request,kind,payload,snapshot) VALUES(?,?,?,?,?,?)",
                (revision, event_id, request_hash, kind, payload, snapshot),
            )
            self.db.executemany(
                "INSERT INTO dependencies VALUES(?,?)",
                [(event_id, s) for s in set(supports)],
            )
            self.db.execute(
                "INSERT OR REPLACE INTO roots VALUES('serving',?)", (snapshot,)
            )
            # Recent snapshots are automatic roots. Explicit pins can retain any older one.
            self.collect(transaction=True)
            count, size = self.db.execute(
                "SELECT COUNT(*),COALESCE(SUM(size),0) FROM objects"
            ).fetchone()
            pages = self.db.execute("PRAGMA page_count").fetchone()[0]
            page_size = self.db.execute("PRAGMA page_size").fetchone()[0]
            if (
                count > self.limits.max_objects
                or size + len(payload) > self.limits.max_bytes
                or pages * page_size > self.limits.max_bytes
            ):
                raise ValueError("Storage ceiling exhausted")
            self.db.execute("COMMIT")
            return json.loads(payload)
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def recover(self, revision=None):
        revision = self.revision if revision is None else revision
        row = self.db.execute(
            "SELECT snapshot FROM events WHERE revision=?", (revision,)
        ).fetchone()
        if row is None:
            raise ValueError("Unknown recovery revision")
        return self.decode(json.loads(self._get(row[0], "manifest")[1]))

    def pin(self, name, revision):
        if not name or name == "serving":
            raise ValueError("Use a non-serving pin name")
        row = self.db.execute(
            "SELECT snapshot FROM events WHERE revision=?", (revision,)
        ).fetchone()
        if row is None:
            raise ValueError("Unknown pin revision")
        self._get(row[0], "manifest")
        self.db.execute("INSERT OR REPLACE INTO roots VALUES(?,?)", (name, row[0]))

    def unpin(self, name):
        if name == "serving":
            raise ValueError("Cannot unpin serving state")
        self.db.execute("DELETE FROM roots WHERE name=?", (name,))

    def collect(self, *, transaction=False):
        if not transaction:
            self.db.execute("BEGIN IMMEDIATE")
        try:
            roots = {r[0] for r in self.db.execute("SELECT snapshot FROM roots")}
            roots.update(
                r[0]
                for r in self.db.execute(
                    "SELECT snapshot FROM events ORDER BY revision DESC LIMIT ?",
                    (self.limits.retained_revisions,),
                )
            )
            live = set(roots)

            def walk(value):
                if isinstance(value, dict):
                    if "tensor" in value:
                        self._get(value["tensor"], "tensor")
                        live.add(value["tensor"])
                    else:
                        for item in value.values():
                            walk(item)
                elif isinstance(value, list):
                    for item in value:
                        walk(item)

            for root in roots:
                walk(json.loads(self._get(root, "manifest")[1]))
            dead = [
                r[0]
                for r in self.db.execute("SELECT hash FROM objects")
                if r[0] not in live
            ]
            self.db.executemany(
                "DELETE FROM objects WHERE hash=?", [(h,) for h in dead]
            )
            if not transaction:
                self.db.execute("COMMIT")
            return dict(removed=len(dead), retained=len(live))
        except BaseException:
            if not transaction and self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def invalidate(self, event_id):
        """Used inside a correction transaction; descendants remain audit records."""
        if not self.db.in_transaction:
            raise ValueError("Invalidation requires a revision transaction")
        if not self.db.execute(
            "SELECT 1 FROM events WHERE id=?", (event_id,)
        ).fetchone():
            raise ValueError("Unknown correction target")
        pending, affected = [event_id], set()
        while pending:
            current = pending.pop()
            if current in affected:
                continue
            affected.add(current)
            pending.extend(
                r[0]
                for r in self.db.execute(
                    "SELECT event FROM dependencies WHERE support=?", (current,)
                )
            )
        self.db.executemany(
            "UPDATE events SET valid=0 WHERE id=?", [(e,) for e in affected]
        )
        return sorted(affected)

    def close(self):
        self.db.close()
