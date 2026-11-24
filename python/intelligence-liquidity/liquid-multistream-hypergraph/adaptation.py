"""Carry the original contextual learner through the growing graph."""

from collections import deque
from dataclasses import dataclass


from graph import GraphState
from contextual import ContextFeedback
from learn import Example


class ReplayBoard:
    """The native feedback protocol, with full graph replay and private states."""

    def __init__(self, graph):
        self.graph = graph

    @property
    def outputs(self):
        return self.graph.cores["b0"].outputs

    @property
    def correction(self):
        return self.graph.cores["b0"].correction

    def context(self, rows):
        return self.graph.cores["b0"].context(rows)

    def start(self, inputs):
        return GraphState()

    def __deepcopy__(self, memo):
        return ReplayBoard(self.graph.fork())

    def __call__(self, inputs, minutes, before, *, checks=True):
        if not checks:
            raise ValueError("Contextual replay retains the native checkers")
        stamp = (
            max(float(t.max()) for t in before.stamps.values())
            if before.stamps
            else 0.0
        ) + float(minutes.max()) * 60
        return self.graph(inputs, minutes, before, stamp=stamp)


def create(graph, settings, *, stream=False):
    learner = (StreamFeedback if stream else ContextFeedback)(ReplayBoard(graph))
    learner.minimum = settings.window
    learner.minority = settings.minority
    learner.history = tuple(deque(maxlen=settings.history) for _ in range(2))
    return learner


def pack(learner):
    from runtime import pack_state

    def example(item):
        return dict(
            timestamp=item.timestamp,
            inputs=item.inputs,
            before=pack_state(item.before),
            truth=item.truth,
            reward=item.reward,
            **(
                dict(
                    event_time=item.event_time,
                    stream_id=item.stream_id,
                    prediction_order=item.prediction_order,
                    revealed_order=item.revealed_order,
                )
                if isinstance(item, StreamExample)
                else {}
            ),
        )

    return dict(
        stream=isinstance(learner, StreamFeedback),
        changed_at=getattr(learner, "changed_at", 0),
        history=[[example(x) for x in queue] for queue in learner.history],
        check=[example(x) for x in learner.check],
        latest=learner.latest,
        receipts=learner.receipts,
        origin=learner.origin,
    )


def restore(graph, settings, payload):
    from runtime import unpack_state

    learner = create(graph, settings, stream=payload.get("stream", False))

    def example(item):
        cls = StreamExample if "event_time" in item else Example
        extras = (
            (
                item["event_time"],
                item["stream_id"],
                item.get("prediction_order", 0),
                item.get("revealed_order", 0),
            )
            if cls is StreamExample
            else ()
        )
        return cls(
            item["timestamp"],
            item["inputs"],
            unpack_state(item["before"]),
            item["truth"],
            item["reward"],
            *extras,
        )

    learner.history = tuple(
        deque((example(x) for x in queue), maxlen=settings.history)
        for queue in payload["history"]
    )
    learner.check = [example(x) for x in payload["check"]]
    learner.latest, learner.receipts = payload["latest"], payload["receipts"]
    learner.origin = payload["origin"]
    learner.changed_at = payload.get("changed_at", 0)
    return learner


@dataclass(frozen=True)
class StreamExample(Example):
    event_time: float
    stream_id: str
    prediction_order: int = 0
    revealed_order: int = 0


def stream_predictions(model, examples):
    """Feedback order grades causality; each replayed liquid path has its own clock."""
    import torch

    states, previous = {}, {}
    answers, probabilities, committed = [], [], []
    for item in examples:
        key, stamp = item.stream_id, item.event_time
        state = states.get(key, item.before)
        if key in previous and stamp - previous[key] != 300:
            state = item.before
        root = state.stamps.get("b0")
        minutes = 5.0 if root is None else (stamp - float(root[0])) / 60
        result = model.graph(
            item.inputs, item.inputs.new_tensor([minutes]), state, stamp=stamp
        )
        answers.append(result.answer)
        probabilities.append(result.probability)
        committed.append(result.committed)
        states[key], previous[key] = result.state.detached(), stamp
    return tuple(torch.cat(v) for v in (answers, probabilities, committed))


class StreamFeedback(ContextFeedback):
    """Native residual fitting with actual clocks and a strict local no-harm gate."""

    def observe(
        self,
        timestamp,
        now,
        row,
        before,
        issued,
        actual,
        *,
        stream_id="default",
        prediction_order=None,
        revealed_order=None,
    ):
        import torch
        from reference import measured_reward

        if timestamp >= now or actual.numel() != 1 or int(actual) not in (0, 1):
            raise ValueError("Delayed binary feedback required")
        # Different streams in the same room may reveal older physical events.
        # Fitting/check separation follows evidence availability, not wall clocks.
        self.latest = 1 if self.latest is None else self.latest + 1
        reward = measured_reward(issued.rounds, actual).detach()
        self.check.append(
            StreamExample(
                self.latest,
                row.detach(),
                before.detached(),
                actual.detach(),
                reward,
                timestamp,
                stream_id,
                timestamp if prediction_order is None else prediction_order,
                now if revealed_order is None else revealed_order,
            )
        )
        if len(self.check) < self.minimum:
            return reward
        fit = sorted((*self.history[0], *self.history[1]), key=lambda x: x.timestamp)
        counts = torch.cat([x.truth for x in self.check]).bincount(minlength=2)
        if min(map(len, self.history)) < self.minority or counts.min() < self.minority:
            receipt = dict(status="frozen_insufficient_classes", accepted=[])
        else:
            receipt = self.try_update(fit, self.check)
            if receipt.get("accepted"):
                self.changed_at = now if revealed_order is None else revealed_order
        self.receipts.append(
            dict(
                revealed_at=now,
                check_first=self.check[0].timestamp,
                check_last=self.check[-1].timestamp,
                check_classes=counts.tolist(),
                **receipt,
            )
        )
        for item in self.check:
            self.history[int(item.truth)].append(item)
        self.check.clear()
        return reward

    def try_update(self, fit, check):
        from copy import deepcopy
        import torch
        from torch import nn
        from torch.nn import functional as F
        from reference import by_class, paired_margin

        if fit[-1].timestamp >= check[0].timestamp:
            raise ValueError("Fit and check intervals overlap")
        first_prediction = min(item.prediction_order for item in check)
        if first_prediction <= getattr(self, "changed_at", 0):
            return dict(status="frozen_check_predates_calibration", accepted=[])
        fit = [item for item in fit if item.revealed_order < first_prediction]
        if not fit:
            return dict(status="frozen_no_available_fit_history", accepted=[])
        causal = dict(
            fit_revealed_through=max(item.revealed_order for item in fit),
            check_prediction_first=first_prediction,
        )
        expected = torch.cat([item.truth for item in fit])
        if int(expected.bincount(minlength=2).min()) < self.minority:
            return dict(status="frozen_recent_history_one_class", accepted=[])
        with torch.no_grad():
            old_answer, old_probability, _ = stream_predictions(self.model, fit)
            inputs = self.model.context(torch.cat([item.inputs for item in fit]))
            prior = torch.logit(old_probability.clamp(0.01, 0.99))
        residual = nn.Parameter(torch.zeros_like(self.model.correction))
        optimizer = torch.optim.Adam([residual], lr=0.04)
        for _ in range(60):
            optimizer.zero_grad()
            error = F.binary_cross_entropy_with_logits(
                prior + inputs @ residual, expected.float(), reduction="none"
            )
            loss = by_class(error, expected) + 0.08 * residual.square().sum()
            loss.backward()
            optimizer.step()
        candidate = deepcopy(self.model)
        best = None
        with torch.no_grad():
            for step in (0.25, 0.5, 1.0):
                moved = (self.model.correction + step * residual.detach()).clamp(
                    -0.8, 0.8
                )
                candidate.correction.copy_(moved)
                answers, probability, _ = stream_predictions(candidate, fit)
                gain = float(
                    by_class(
                        answers.eq(expected).float() - old_answer.eq(expected).float(),
                        expected,
                    )
                )
                brier = float(
                    by_class(
                        (old_probability - expected).square()
                        - (probability - expected).square(),
                        expected,
                    )
                )
                if gain > 0 and (best is None or (gain, brier) > best[:2]):
                    best = gain, brier, step, moved.clone()
            details = dict(
                fit_first=fit[0].timestamp,
                fit_last=fit[-1].timestamp,
                fit_rows=len(fit),
                check_rows=len(check),
                accepted=[],
                **causal,
            )
            if best is None:
                return dict(details, status="frozen_no_fit_gain")
            candidate.correction.copy_(best[3])
            a, p, q = stream_predictions(self.model, check)
            b, r, s = stream_predictions(candidate, check)
            truth = torch.cat([item.truth for item in check])
            delta = b.eq(truth).float() - a.eq(truth).float()
            gain = float(by_class(delta, truth))
            brier = float(by_class((p - truth).square() - (r - truth).square(), truth))
            classes = [float(delta[truth == c].mean()) for c in (0, 1)]
            quorum = float((s.float() - q.float()).mean())
            lower = paired_margin(delta, truth)
            accepted = (
                gain >= 0.01
                and brier >= 0.002
                and lower is not None
                and lower > 0
                and min(classes) >= 0
                and quorum >= 0
            )
            if accepted:
                self.model.correction.copy_(best[3])
                details["accepted"] = ["stream contextual readout"]
            return dict(
                details,
                status="accepted" if accepted else "frozen_no_board_gain",
                board_gain=dict(
                    accuracy_gain=gain,
                    brier_gain=brier,
                    accuracy_lower=lower,
                    class_gain=classes,
                    quorum_gain=quorum,
                ),
            )
