"""Transactional public entry point for inference, labels, rewards and recovery."""

import io
import hashlib
import platform
from pathlib import Path

import torch

from provenance import Features, observation
from runtime import Runtime


def pack_runtime(runtime):
    buffer = io.BytesIO()
    runtime.save(buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=True, map_location="cpu")


def unpack_runtime(payload, stage_directory):
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    return Runtime.load(buffer, stage_directory=stage_directory)


class Session:
    """No prediction is returned before its state and event commit together.

    One process owns a Session. Independent sessions may share the SQLite file;
    optimistic revision checks prevent either from overwriting the other's state.
    """

    def __init__(self, ledger, runtime=None, *, maximum_features=8, origin=None):
        self.ledger = ledger
        self.execution = dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            code={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
        )
        self.stages = ledger.path.parent / (ledger.path.stem + "-stages")
        self.stages.mkdir(exist_ok=True)
        self.features, self.blocked = Features(maximum_features), None
        if ledger.revision:
            self._install(ledger.recover())
        elif runtime is not None:
            self.runtime = runtime
            self.runtime.stage_directory = self.stages
            ledger.commit("initial", "initial", origin or {}, self._state(runtime), 0)
        else:
            raise ValueError("An empty ledger needs an initial liquid runtime")
        self.revision = ledger.revision

    def _state(self, runtime):
        return dict(
            schema="hypergraph-session-v1",
            runtime=pack_runtime(runtime),
            features=self.features.definitions,
            maximum_features=self.features.maximum,
            blocked=self.blocked,
            execution=self.execution,
        )

    def _install(self, state):
        if state["schema"] != "hypergraph-session-v1":
            raise ValueError("Unknown session schema")
        if state["execution"] != self.execution:
            raise ValueError(
                "Runtime source or environment changed; migrate explicitly before resuming"
            )
        self.runtime = unpack_runtime(state["runtime"], self.stages)
        self.features = Features(state["maximum_features"])
        self.features.definitions = state["features"]
        self.blocked = state["blocked"]

    def refresh(self):
        self._install(self.ledger.recover())
        self.revision = self.ledger.revision

    def _apply(self, event_id, kind, request, operation, supports=(), invalidates=()):
        request = dict(kind=kind, **request)
        old = self.ledger.result(event_id, request)
        if old is not None:
            self.refresh()
            return old["result"]
        if self.ledger.revision != self.revision:
            raise ValueError("Stale session; refresh before retry")
        previous_rng = torch.get_rng_state()
        candidate = unpack_runtime(pack_runtime(self.runtime), self.stages)
        previous_features, previous_blocked = self.features, self.blocked
        self.features = Features(self.features.maximum)
        self.features.definitions = __import__("copy").deepcopy(
            previous_features.definitions
        )
        try:
            result = operation(candidate)
            support = set(supports)
            # Every normal transition also depends on the prior committed model state.
            if not invalidates and kind != "restore":
                support.add(
                    self.ledger.db.execute(
                        "SELECT id FROM events WHERE revision=?", (self.revision,)
                    ).fetchone()[0]
                )
            self.ledger.commit(
                event_id,
                kind,
                request,
                self._state(candidate),
                self.revision,
                result=result,
                supports=support,
                invalidates=invalidates,
            )
            self.runtime = candidate
            self.revision = self.ledger.revision
            return result
        except BaseException:
            self.features, self.blocked = previous_features, previous_blocked
            torch.set_rng_state(previous_rng)
            raise

    def predict(
        self,
        event_id,
        event_time,
        arrival_time,
        values,
        sensors,
        *,
        quality=None,
        context="default",
    ):
        request = observation(
            event_id, event_time, arrival_time, values, sensors, quality or {}, context
        )

        def run(runtime):
            if self.blocked:
                raise ValueError("Model quarantined: " + self.blocked)
            result = runtime.predict(
                arrival_time, self.features.apply(request["values"])
            )
            runtime.pending[result["token"]].update(
                observation_id=event_id, context=context, model_revision=self.revision
            )
            return result

        return self._apply(event_id, "observation", request, run)

    def label(
        self,
        event_id,
        prediction_id,
        truth,
        revealed_at,
        *,
        authority,
        schema="occupancy-binary-v1",
    ):
        if not authority or schema != "occupancy-binary-v1":
            raise ValueError("An external binary occupancy label authority is required")
        row = self.ledger.db.execute(
            "SELECT kind,payload FROM events WHERE id=?", (prediction_id,)
        ).fetchone()
        if row is None or row[0] != "observation":
            raise ValueError("Unknown observation")
        result = __import__("json").loads(row[1])["result"]
        token = result["token"]

        def run(runtime):
            if self.blocked:
                raise ValueError("Model quarantined")
            from ledger import canonical

            old_receipts = {canonical(r) for r in runtime.receipts}
            old_monitor = {canonical(r) for r in runtime.monitor_audit}
            runtime.pending[token]["label_id"] = event_id
            runtime.observe(token, truth, revealed_at)
            return dict(
                token=token,
                truth=truth,
                receipts=len(runtime.receipts),
                decisions=[
                    r for r in runtime.receipts if canonical(r) not in old_receipts
                ],
                monitoring=[
                    r for r in runtime.monitor_audit if canonical(r) not in old_monitor
                ],
                proposal=runtime.trial,
            )

        return self._apply(
            event_id,
            "label",
            dict(
                prediction=prediction_id,
                truth=truth,
                revealed_at=revealed_at,
                authority=authority,
                schema=schema,
            ),
            run,
            supports=[prediction_id],
        )

    def reward(self, event_id, prediction_id, value, revealed_at):
        import json

        row = self.ledger.db.execute(
            "SELECT kind,payload FROM events WHERE id=?", (prediction_id,)
        ).fetchone()
        if row is None or row[0] != "observation":
            raise ValueError("Unknown observation")
        token = json.loads(row[1])["result"]["token"]

        def run(runtime):
            if self.blocked:
                raise ValueError("Model quarantined")
            runtime.reward(token, value, revealed_at)
            return dict(token=token, reward=value)

        return self._apply(
            event_id,
            "reward",
            dict(prediction=prediction_id, value=value, revealed_at=revealed_at),
            run,
            [prediction_id],
        )

    def correct(self, event_id, target, reason):
        if not reason:
            raise ValueError("Correction reason required")

        def run(runtime):
            self.blocked = (
                "invalidated evidence; restore an earlier clean revision or retrain"
            )
            return dict(target=target, quarantined=True)

        return self._apply(
            event_id,
            "correction",
            dict(target=target, reason=reason),
            run,
            invalidates=[target],
        )

    def restore(self, event_id, revision):
        row = self.ledger.db.execute(
            "SELECT id,valid FROM events WHERE revision=?", (revision,)
        ).fetchone()
        if row is None or not row[1]:
            raise ValueError("Restore needs a clean retained revision")
        state = self.ledger.recover(revision)
        if state["blocked"]:
            raise ValueError("Cannot restore a quarantined snapshot")

        def run(runtime):
            restored = unpack_runtime(state["runtime"], self.stages)
            runtime.__dict__.clear()
            runtime.__dict__.update(restored.__dict__)
            self.features = Features(state["maximum_features"])
            self.features.definitions = state["features"]
            self.blocked = None
            return dict(restored_revision=revision)

        return self._apply(
            event_id, "restore", dict(revision=revision), run, supports=[row[0]]
        )

    def derive(self, event_id, name, sources, evidence_id, operation="mean"):
        def run(runtime):
            if self.blocked:
                raise ValueError("Model quarantined")
            if name in runtime.graph.fields or name in runtime.queued_fields:
                raise ValueError("Feature would overwrite an existing input")
            if (
                len(runtime.graph.fields)
                + len(runtime.queued_fields)
                + len(self.features.definitions)
                + 1
                > runtime.graph.max_fields
            ):
                raise ValueError("Input ceiling exhausted")
            self.features.add(name, sources, evidence_id, operation)
            return dict(feature=name, status="queued_for_labeled_schema_stage")

        return self._apply(
            event_id,
            "derived_feature",
            dict(name=name, sources=sources, evidence=evidence_id, operation=operation),
            run,
            supports=[evidence_id],
        )
