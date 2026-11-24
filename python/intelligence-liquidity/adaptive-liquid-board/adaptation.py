"""Carry the original contextual learner through the growing graph."""

from collections import deque

import torch

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


def create(graph, settings):
    learner = ContextFeedback(ReplayBoard(graph))
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
        )

    return dict(
        history=[[example(x) for x in queue] for queue in learner.history],
        check=[example(x) for x in learner.check],
        latest=learner.latest,
        receipts=learner.receipts,
        origin=learner.origin,
    )


def restore(graph, settings, payload):
    from runtime import unpack_state

    learner = create(graph, settings)

    def example(item):
        return Example(
            item["timestamp"],
            item["inputs"],
            unpack_state(item["before"]),
            item["truth"],
            item["reward"],
        )

    learner.history = tuple(
        deque((example(x) for x in queue), maxlen=settings.history)
        for queue in payload["history"]
    )
    learner.check = [example(x) for x in payload["check"]]
    learner.latest, learner.receipts = payload["latest"], payload["receipts"]
    learner.origin = payload["origin"]
    return learner
