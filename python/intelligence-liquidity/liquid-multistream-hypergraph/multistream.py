"""One mutable liquid graph, private stream memories, and room-level release gates.

A single owner calls this coordinator. Logical streams are not OS threads or GPU
streams. Batches contain at most one next observation from each logical stream.
"""

from collections import Counter, deque
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path

import torch

from adaptation import ReplayBoard, create, pack, restore
from batching import Backend
from graph import GraphState
from runtime import Runtime, metrics, pack_state, unpack_state


@dataclass(frozen=True)
class StreamLimits:
    maximum_streams: int = 36
    maximum_rooms: int = 6
    maximum_pending_per_stream: int = 8
    maximum_queue: int = 72
    batch_size: int = 8
    history_per_room: int = 96
    maximum_deduplication: int = 4096
    maximum_resident_bytes: int = 1073741824
    trial_timeout_predictions: int = 20736

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("Stream bounds must be positive integers")
        if self.maximum_rooms > self.maximum_streams:
            raise ValueError("Room ceiling exceeds stream ceiling")


class MultiRuntime(Runtime):
    """Causal feedback is ordered within streams; shared changes require room checks."""

    def __init__(self, graph, *, stream_limits=None, device="cpu", **kwargs):
        super().__init__(graph, **kwargs)
        self.stream_limits = stream_limits or StreamLimits()
        if self.stream_limits.batch_size > graph.max_batch:
            raise ValueError("Scheduler batch exceeds graph envelope")
        if (
            self.stream_limits.history_per_room * self.stream_limits.maximum_rooms
            > self.settings.history
        ):
            raise ValueError("Room histories exceed the shared history reservation")
        self.backend = Backend(device)
        # Contextual replay and correction belong to the stream, not the trunk.
        self.contextual = None
        self.streams = {}
        self.queue = deque()
        self.room_history = {}
        self.rooms = {}
        self.evidence = {}
        self.version = 0
        self.order = 0
        self.protected_rooms = []
        self.monitor_rooms = []
        self.trial_started = 0
        self.active_token = None
        self.late_feedback = 0
        self.evidence_floor = {}
        self.suspended_pending = {}

    def open(self, stream_id, room):
        if (
            not isinstance(stream_id, str)
            or not stream_id
            or not isinstance(room, str)
            or not room
        ):
            raise ValueError("Nonempty stream and room IDs required")
        if stream_id in self.streams:
            if self.streams[stream_id]["room"] != room:
                raise ValueError("A stream cannot change room; open a new identity")
            return
        if len(self.streams) >= self.stream_limits.maximum_streams:
            raise ValueError("Stream ceiling exhausted")
        if (
            room not in self.room_history
            and len(self.room_history) >= self.stream_limits.maximum_rooms
        ):
            raise ValueError("Room ceiling exhausted")
        if (
            self.reserved_bytes(self.graph, len(self.streams) + 1)
            > self.stream_limits.maximum_resident_bytes
        ):
            raise ValueError("Opening stream would exceed its memory reservation")
        if room not in self.rooms:
            self.rooms[room] = dict(
                correction=self.graph.cores["b0"].correction.detach().clone(),
                learner=create(self.graph, self.settings, stream=True)
                if self.settings.contextual
                else None,
            )
        shared = self.rooms[room]
        self.streams[stream_id] = dict(
            room=room,
            state=GraphState(),
            shadow_state=GraphState(),
            fallback_state=GraphState(),
            correction=shared["correction"],
            last_prediction=-math.inf,
            last_reveal=-math.inf,
            learner=shared["learner"],
        )
        self.room_history.setdefault(
            room, deque(maxlen=self.stream_limits.history_per_room)
        )
        if self.shadow is not None and room not in self.protected_rooms:
            self.protected_rooms.append(room)
        if self.fallback is not None and room not in self.monitor_rooms:
            self.monitor_rooms.append(room)

    def close(self, stream_id):
        if any(r["stream_id"] == stream_id for r in self.pending.values()) or any(
            r["stream_id"] == stream_id for r in self.queue
        ):
            raise ValueError("Drain or discard stream work before closing")
        del self.streams[stream_id]

    def reset_stream(self, stream_id):
        slot = self.streams[stream_id]
        if any(r["stream_id"] == stream_id for r in self.pending.values()):
            raise ValueError("Cannot reset a stream with pending feedback")
        slot.update(
            state=GraphState(), shadow_state=GraphState(), fallback_state=GraphState()
        )

    def _request(self, item):
        item = dict(item)
        sid, room, stamp = item["stream_id"], item["room"], item["stamp"]
        if sid not in self.streams:
            raise ValueError("Open the stream before submitting observations")
        slot = self.streams[sid]
        if (
            room != slot["room"]
            or not math.isfinite(stamp)
            or stamp <= max(slot["last_prediction"], slot["last_reveal"])
        ):
            raise ValueError("Room identity and increasing per-stream clock required")
        if not isinstance(item.get("evidence_id"), str) or not item["evidence_id"]:
            raise ValueError("Stable underlying observation evidence_id required")
        for name, value in item["values"].items():
            if not isinstance(name, str) or not name or name.startswith("parent:"):
                raise ValueError("Invalid feature identity")
            if value is not None and (
                math.isinf(float(value))
                or abs(float(value)) > self.controller.guards.max_abs_input
            ):
                raise ValueError("Input magnitude guard exceeded")
        item["values"] = {
            k: (None if v is None or math.isnan(float(v)) else float(v))
            for k, v in item["values"].items()
        }
        return item

    def enqueue(self, **item):
        item = self._request(item)
        if len(self.queue) >= self.stream_limits.maximum_queue:
            raise ValueError("Ingress queue exhausted")
        previous = [
            r["stamp"] for r in self.queue if r["stream_id"] == item["stream_id"]
        ]
        if previous and item["stamp"] <= previous[-1]:
            raise ValueError("Queued clocks must increase within each stream")
        self.queue.append(item)

    def drain(self):
        """Round-robin batches; no room must supply a sample for another to run."""
        outputs = []
        while self.queue:
            selected, held, seen = [], deque(), set()
            for item in self.queue:
                sid = item["stream_id"]
                full = (
                    sum(r["stream_id"] == sid for r in self.pending.values())
                    >= self.stream_limits.maximum_pending_per_stream
                )
                if (
                    sid in seen
                    or full
                    or len(selected) >= self.stream_limits.batch_size
                ):
                    held.append(item)
                else:
                    selected.append(item)
                    seen.add(sid)
            if (
                not selected
                or len(self.pending) + len(selected) > self.settings.maximum_pending
            ):
                break
            result = self.predict_batch(selected)
            self.queue = held
            outputs.extend(result)
        return outputs

    def predict(
        self, stamp, values, *, stream_id="default", room="default", evidence_id=None
    ):
        if stream_id not in self.streams:
            self.open(stream_id, room)
        return self.predict_batch(
            [
                dict(
                    stream_id=stream_id,
                    room=room,
                    stamp=stamp,
                    values=values,
                    evidence_id=evidence_id or f"{room}:{stamp}",
                )
            ]
        )[0]

    @torch.no_grad()
    def predict_batch(self, items):
        if not items or len(items) > self.stream_limits.batch_size:
            raise ValueError("Batch must fit the configured ceiling")
        items = [self._request(r) for r in items]
        if len({r["stream_id"] for r in items}) != len(items):
            raise ValueError("A batch may contain only one next sample per stream")
        if len(self.pending) + len(items) > self.settings.maximum_pending:
            raise ValueError("Global pending-label budget exhausted")
        for r in items:
            if (
                sum(p["stream_id"] == r["stream_id"] for p in self.pending.values())
                >= self.stream_limits.maximum_pending_per_stream
            ):
                raise ValueError("Per-stream pending budget exhausted")
        unknown = sorted(
            set().union(*(r["values"] for r in items))
            - set(self.graph.fields)
            - set(self.queued_fields)
        )
        if (
            len(self.graph.fields) + len(self.queued_fields) + len(unknown)
            > self.graph.max_fields
        ):
            raise ValueError("Feature ceiling exhausted")
        # Reserve serialized state bytes before evaluating; graph shape is static.
        if not self.candidate_reservation(self.graph):
            raise ValueError("Resident stream memory envelope exhausted")
        slots = [self.streams[r["stream_id"]] for r in items]
        stamps = [r["stamp"] for r in items]
        minutes = [
            5.0
            if not math.isfinite(s["last_prediction"])
            else (t - s["last_prediction"]) / 60
            for s, t in zip(slots, stamps)
        ]
        corrections = [s["correction"] for s in slots]

        def run(model, state_name):
            values = [
                {k: v for k, v in r["values"].items() if k in model.fields}
                for r in items
            ]
            # Candidate schema is shared; private corrections map by field identity.
            cs = [
                self._correction(c, self.graph.fields, model.fields)
                for c in corrections
            ]
            return self.backend.run(
                model, values, [s[state_name] for s in slots], stamps, minutes, cs
            )

        active, states = run(self.graph, "state")
        shadow, shadow_states = (
            (None, None) if self.shadow is None else run(self.shadow, "shadow_state")
        )
        fallback, fallback_states = (
            (None, None)
            if self.fallback is None
            else run(self.fallback, "fallback_state")
        )
        self.queued_fields.extend(unknown)
        output = []
        for i, (item, slot) in enumerate(zip(items, slots)):
            proposals = {}
            for _, (board, decision) in active.trace.items():
                core, first = self.graph.cores[board], decision.rounds[0]
                for index, identity in enumerate(core.ids):
                    proposals[board + "/" + identity] = dict(
                        hidden=core.hidden(first.cells)[i, index].detach().clone(),
                        probability=float(first.margins[i, index, -1].sigmoid()),
                        acted=bool(first.acted[i, index]),
                        confidence=float(first.confidence[i, index]),
                        speaks=bool(first.speaks[i, index]),
                    )
            token = self.serial
            self.serial += 1
            self.order += 1
            row = dict(
                stamp=self.order,
                event_time=item["stamp"],
                minutes=minutes[i],
                values=item["values"],
                stream_id=item["stream_id"],
                room=item["room"],
                evidence_id=item["evidence_id"],
                proposals=proposals,
                probability=float(active.probability[i]),
                answer=int(active.answer[i]),
                quorum=bool(active.committed[i]),
                trial=self.trial_number if shadow is not None else None,
                version=self.version,
                context_before=pack_state(slot["state"]),
            )
            for name, result in (("shadow", shadow), ("fallback", fallback)):
                row[name] = None if result is None else float(result.probability[i])
                row[name + "_answer"] = (
                    None if result is None else int(result.answer[i])
                )
                row[name + "_quorum"] = (
                    None if result is None else bool(result.committed[i])
                )
            self.pending[token] = row
            slot["state"] = states[i]
            if shadow_states is not None:
                slot["shadow_state"] = shadow_states[i]
            if fallback_states is not None:
                slot["fallback_state"] = fallback_states[i]
            slot["last_prediction"] = item["stamp"]
            output.append(
                dict(
                    token=token,
                    stream_id=item["stream_id"],
                    room=item["room"],
                    probability=row["probability"],
                    answer=row["answer"],
                    quorum=row["quorum"],
                    version=self.version,
                )
            )
        self.last_prediction = self.order
        self._expire_trial()
        return output

    @staticmethod
    def _correction(correction, old, new):
        mapped = correction.new_zeros(2 * len(new) + 1)
        for i, name in enumerate(old):
            if name in new:
                j = new.index(name)
                mapped[j], mapped[len(new) + j] = (
                    correction[i],
                    correction[len(old) + i],
                )
        mapped[-1] = correction[-1]
        return mapped

    def reward(self, token, value, revealed_at):
        if (
            token not in self.pending
            or not math.isfinite(value)
            or not -1 <= value <= 1
        ):
            raise ValueError("A pending prediction and bounded finite reward required")
        row = self.pending[token]
        slot = self.streams[row["stream_id"]]
        if (
            "reward" in row
            or not math.isfinite(revealed_at)
            or revealed_at
            <= max(row["event_time"], row.get("arrival_time", row["event_time"]))
            or revealed_at < slot["last_reveal"]
        ):
            raise ValueError(
                "Reward must be unique, delayed and ordered within its stream"
            )
        row.update(reward=float(value), reward_at=revealed_at)
        slot["last_reveal"] = revealed_at

    def observe(self, token, truth, revealed_at):
        if (
            token not in self.pending
            or truth not in (0, 1)
            or not math.isfinite(revealed_at)
        ):
            raise ValueError(
                "Pending token, binary truth and finite reveal time required"
            )
        row = self.pending[token]
        sid, room = row["stream_id"], row["room"]
        slot = self.streams[sid]
        first = min(t for t, r in self.pending.items() if r["stream_id"] == sid)
        if (
            token != first
            or revealed_at
            <= max(row["event_time"], row.get("arrival_time", row["event_time"]))
            or revealed_at < max(slot["last_prediction"], slot["last_reveal"])
        ):
            raise ValueError(
                "Feedback must follow prediction and be ordered within its stream"
            )
        key = json.dumps([room, row["evidence_id"]], separators=(",", ":"))
        if (
            row["event_time"] <= self.evidence_floor.get(room, -math.inf)
            and key not in self.evidence
        ):
            raise ValueError(
                "Evidence is older than the deduplication retention; replay from a retained snapshot"
            )
        signature = hashlib.sha256(
            json.dumps(
                dict(time=row["event_time"], values=row["values"], truth=truth),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if key in self.evidence and self.evidence[key][2] != signature:
            raise ValueError(
                "Conflicting replicated evidence; correct provenance before learning"
            )
        slot["last_reveal"] = revealed_at
        if key in self.evidence:
            del self.pending[token]
            return dict(shared=False, reason="replicated_evidence")
        # A graph version never replays an incompatible old liquid state.
        if row["version"] == self.version and slot["learner"] is not None:
            root = self.graph.cores["b0"]
            original = root.correction.detach().clone()
            root.correction.copy_(slot["correction"])
            try:
                learner = slot["learner"]
                learner.model = ReplayBoard(self.graph)
                before = unpack_state(row["context_before"])
                with torch.no_grad():
                    inputs = self.graph.tensor(
                        {
                            k: v
                            for k, v in row["values"].items()
                            if k in self.graph.fields
                        }
                    )
                    issued = self.graph(
                        inputs,
                        inputs.new_tensor([row["minutes"]]),
                        before,
                        stamp=row["event_time"],
                    )
                learner.observe(
                    row["event_time"],
                    revealed_at,
                    inputs,
                    before,
                    issued.decision,
                    torch.tensor([truth]),
                    stream_id=sid,
                    prediction_order=row["stamp"],
                    revealed_order=self.order + 1,
                )
                learner.receipts = learner.receipts[-self.settings.maximum_receipts :]
                moved = root.correction.detach().clone()
                self.rooms[room]["correction"] = moved
                for stream in self.streams.values():
                    if stream["room"] == room:
                        stream["correction"] = moved
            finally:
                root.correction.copy_(original)
        elif row["version"] != self.version:
            self.late_feedback += 1
        # Global sequence orders fit/check causality; physical clocks remain private.
        self.order += 1
        row["revealed_at"] = self.order
        self.evidence[key] = (room, row["event_time"], signature)
        while len(self.evidence) > self.stream_limits.maximum_deduplication:
            retired = next(iter(self.evidence))
            retired_room, retired_time, _ = self.evidence[retired]
            self.evidence_floor[retired_room] = max(
                self.evidence_floor.get(retired_room, -math.inf), retired_time
            )
            del self.evidence[retired]
        local = list(self.room_history[room]) + [dict(row, truth=int(truth))]
        while len(local) > self.stream_limits.history_per_room:
            majority = Counter(r["truth"] for r in local).most_common(1)[0][0]
            del local[next(i for i, r in enumerate(local) if r["truth"] == majority)]
        self.room_history[room] = deque(
            local, maxlen=self.stream_limits.history_per_room
        )
        others = self.pending
        self.suspended_pending = others
        self.pending = {token: row}
        self.active_token = token
        try:
            super().observe(token, truth, self.order)
        finally:
            others.pop(token, None)
            self.pending, self.active_token = others, None
            self.suspended_pending = {}
        return dict(shared=True, version=self.version)

    def propose(self):
        if self.shadow is not None or self.fallback is not None:
            return
        # Bounded class-balanced room tails prevent both rate dominance and
        # forgetting all minority examples during long occupied/vacant runs.
        rooms = sorted({s["room"] for s in self.streams.values()})
        if not rooms or any(
            min(
                Counter(r["truth"] for r in self.room_history[room]).get(c, 0)
                for c in (0, 1)
            )
            < self.settings.minority
            for room in rooms
        ):
            self.receipts.append(
                dict(
                    status="waiting_for_per_room_fitting_classes",
                    accepted=False,
                    classes={
                        room: dict(Counter(r["truth"] for r in self.room_history[room]))
                        for room in rooms
                    },
                )
            )
            self.since_trial = 0
            return
        count = min(len(self.room_history[r]) for r in rooms)
        rows = sorted(
            [row for room in rooms for row in list(self.room_history[room])[-count:]],
            key=lambda r: r["stamp"],
        )
        self.history = deque(rows, maxlen=self.settings.history)
        others = self.pending
        self.pending = {}
        representative = next(iter(self.streams.values()))
        self.state = representative["state"]
        try:
            super().propose()
        finally:
            self.pending = others
        if self.shadow is not None:
            self.protected_rooms = rooms
            self.trial_started = self.serial
            for slot in self.streams.values():
                slot["shadow_state"] = self.shadow.align(slot["state"], self.graph)

    def _reset_calibration_history(self):
        for room, shared in self.rooms.items():
            shared["learner"] = (
                create(self.graph, self.settings, stream=True)
                if self.settings.contextual
                else None
            )
            for slot in self.streams.values():
                if slot["room"] == room:
                    slot["learner"] = shared["learner"]
                    slot["correction"] = shared["correction"]

    def schema_changed(self, old_fields):
        self.version += 1
        for shared in self.rooms.values():
            shared["correction"] = self._correction(
                shared["correction"], old_fields, self.graph.fields
            )
        self._reset_calibration_history()

    def _check_ready(self, rows, rooms):
        # Bound the faster rooms while waiting for slower rooms' labels.
        kept = []
        for room in rooms:
            local = [r for r in rows if r.get("room") == room]
            kept.extend(local[: self.settings.window])
        rows[:] = sorted(kept, key=lambda r: r["stamp"])
        return all(
            sum(r.get("room") == room for r in rows) >= self.settings.window
            for room in rooms
        )

    def quality_veto(self, rows):
        rooms = self.protected_rooms if self.shadow is not None else self.monitor_rooms
        reports = {}
        for room in rooms:
            local = [r for r in rows if r.get("room") == room]
            counts = Counter(r["truth"] for r in local)
            supported = min(counts.get(c, 0) for c in (0, 1)) >= self.settings.minority
            a, b = (
                (metrics(local, "active"), metrics(local, "candidate"))
                if supported
                else (None, None)
            )
            quorum = sum(
                int(r["candidate_quorum"]) - int(r["active_quorum"]) for r in local
            ) / max(1, len(local))
            safe = (
                supported
                and b["brier"] <= a["brier"]
                and all(y >= x for x, y in zip(a["recall"], b["recall"]))
                and quorum >= 0
            )
            reports[room] = dict(
                supported=supported,
                passed=safe,
                active=a,
                candidate=b,
                quorum_gain=quorum,
            )
        return bool(reports) and all(r["passed"] for r in reports.values()), reports

    def finish(self):
        if not self._check_ready(self.check, self.protected_rooms):
            return
        old = self.graph
        super().finish()
        if self.graph is not old:
            self.version += 1
            self.monitor_rooms = list(self.protected_rooms)
            for slot in self.streams.values():
                slot["fallback_state"], slot["state"] = (
                    slot["state"],
                    slot["shadow_state"],
                )
            self._reset_calibration_history()
        for slot in self.streams.values():
            slot["shadow_state"] = GraphState()

    def finish_monitor(self):
        if not self._check_ready(self.monitor, self.monitor_rooms):
            return
        old = self.graph
        super().finish_monitor()
        if old is not self.graph:
            self.version += 1
            for slot in self.streams.values():
                slot["state"] = slot["fallback_state"]
            self._reset_calibration_history()
        for slot in self.streams.values():
            slot["fallback_state"] = GraphState()

    def _expire_trial(self):
        if (
            (self.shadow is not None or self.fallback is not None)
            and self.serial - self.trial_started
            > self.stream_limits.trial_timeout_predictions
        ):
            if self.shadow is not None:
                self.receipts.append(
                    dict(status="expired_without_all_room_evidence", accepted=False)
                )
                self.shadow, self.trial, self.check = None, None, []
                for slot in self.streams.values():
                    slot["shadow_state"] = GraphState()
            else:
                # No supported monitoring decision: restore the protected model.
                self.graph = self.fallback
                self.version += 1
                for slot in self.streams.values():
                    slot["state"] = slot["fallback_state"]
                self._reset_calibration_history()
                self.fallback, self.monitor = None, []
                self.monitor_audit.append(dict(rollback=True, reason="monitor_timeout"))

    def reserved_bytes(self, graph, stream_count=None):
        """Conservative tensor reservation; the worker separately caps total address space."""
        count = len(self.streams) if stream_count is None else stream_count
        contexts = graph.profile()["context_experts"]
        layers = max(len(c.cells) for c in graph.cores.values())
        heads = graph.profile()["heads"]
        state = (contexts * (layers * graph.max_liquid + 1) + heads * 16) * 8
        private_replay = (
            min(count, self.stream_limits.maximum_rooms)
            * (2 * self.settings.history + self.settings.window)
            if self.settings.contextual
            else 0
        )
        slots = (
            3 * count
            + self.settings.maximum_pending
            + private_replay
            + 2 * self.settings.history
            + 2 * self.settings.window * self.stream_limits.maximum_rooms
        )
        return slots * state + 12 * sum(p.numel() for p in graph.parameters())

    def candidate_reservation(self, graph):
        return self.reserved_bytes(graph) <= self.stream_limits.maximum_resident_bytes

    @torch.no_grad()
    def verify_preservation(self, candidate, rows):
        maximum, passed = 0.0, True
        for slot in self.streams.values():
            active = slot["state"].detached()
            shadow = candidate.align(active, self.graph)
            clock = (
                slot["last_prediction"]
                if math.isfinite(slot["last_prediction"])
                else 0.0
            )
            for index, row in enumerate(rows):
                stamp = clock + 300 * (index + 1)
                values = {
                    k: v for k, v in row["values"].items() if k in self.graph.fields
                }
                a, astates = self.backend.run(
                    self.graph, [values], [active], [stamp], [5.0], [slot["correction"]]
                )
                correction = self._correction(
                    slot["correction"], self.graph.fields, candidate.fields
                )
                b, bstates = self.backend.run(
                    candidate, [values], [shadow], [stamp], [5.0], [correction]
                )
                maximum = max(
                    maximum, float((a.probability - b.probability).abs().max())
                )
                passed = (
                    passed
                    and torch.equal(a.answer, b.answer)
                    and torch.equal(a.committed, b.committed)
                )
                active, shadow = astates[0], bstates[0]
        return dict(
            passed=bool(passed and maximum <= 1e-6),
            maximum_probability_difference=maximum,
            frames=len(rows),
            streams=len(self.streams),
        )

    def resident_bytes(self):
        # Tensor storage upper bound including private pending/replay states.
        def size(value):
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, GraphState):
                return size(pack_state(value))
            if isinstance(value, dict):
                return sum(size(v) for v in value.values())
            if isinstance(value, (list, tuple, deque)):
                return sum(size(v) for v in value)
            return 0

        return size(self.pending) + sum(
            size({k: v for k, v in s.items() if k != "learner"})
            + (size(pack(s["learner"])) if s["learner"] is not None else 0)
            for s in self.streams.values()
        )

    def save(self, path):
        buffer = io.BytesIO()
        super().save(buffer)
        buffer.seek(0)
        engine = torch.load(buffer, weights_only=True)
        engine["pending"].update(
            {k: v for k, v in self.suspended_pending.items() if k != self.active_token}
        )
        payload = dict(
            schema="liquid-multistream-v1",
            engine=engine,
            limits=asdict(self.stream_limits),
            device=str(self.backend.device),
            version=self.version,
            order=self.order,
            protected_rooms=self.protected_rooms,
            monitor_rooms=self.monitor_rooms,
            trial_started=self.trial_started,
            late_feedback=self.late_feedback,
            queue=list(self.queue),
            room_history={k: list(v) for k, v in self.room_history.items()},
            evidence=self.evidence,
            evidence_floor=self.evidence_floor,
            rooms={
                room: dict(
                    correction=shared["correction"],
                    learner=None
                    if shared["learner"] is None
                    else pack(shared["learner"]),
                )
                for room, shared in self.rooms.items()
            },
            streams={},
        )
        for sid, slot in self.streams.items():
            payload["streams"][sid] = {
                k: (
                    pack_state(v)
                    if isinstance(v, GraphState)
                    else pack(v)
                    if k == "learner" and v is not None
                    else v
                )
                for k, v in slot.items()
                if k not in ("correction", "learner")
            }
        if hasattr(path, "write"):
            torch.save(payload, path)
        else:
            path = Path(path)
            temporary = path.with_name(path.name + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)

    @classmethod
    def load(cls, path, *, stage_directory=None, device=None):
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if payload.pop("schema") != "liquid-multistream-v1":
            raise ValueError("Unknown multistream checkpoint schema")
        buffer = io.BytesIO()
        torch.save(payload.pop("engine"), buffer)
        buffer.seek(0)
        engine = Runtime.load(buffer, stage_directory=stage_directory)
        result = cls.__new__(cls)
        result.__dict__.update(engine.__dict__)
        result.stream_limits = StreamLimits(**payload.pop("limits"))
        saved_device = payload.pop("device")
        result.backend = Backend(device or saved_device)
        result.rooms = {
            room: dict(
                correction=shared["correction"],
                learner=None
                if shared["learner"] is None
                else restore(result.graph, result.settings, shared["learner"]),
            )
            for room, shared in payload.pop("rooms").items()
        }
        result.streams = {}
        for sid, slot in payload.pop("streams").items():
            result.streams[sid] = {
                k: (
                    unpack_state(v)
                    if k in ("state", "shadow_state", "fallback_state")
                    else restore(result.graph, result.settings, v)
                    if k == "learner" and v is not None
                    else v
                )
                for k, v in slot.items()
            }
        for slot in result.streams.values():
            slot.update(result.rooms[slot["room"]])
        result.queue = deque(payload.pop("queue"))
        result.room_history = {
            k: deque(v, maxlen=result.stream_limits.history_per_room)
            for k, v in payload.pop("room_history").items()
        }
        result.active_token = None
        result.suspended_pending = {}
        result.__dict__.update(payload)
        return result
