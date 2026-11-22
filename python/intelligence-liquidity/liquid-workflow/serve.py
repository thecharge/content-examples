"""Read one observation per line, carry state, and save atomically after each answer."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import torch

from model import Ledger, Workflow, learn_from_observation
from run import save, sha


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered)*fraction)-1)] if ordered else None


class Stream:
    def __init__(self, path, cutoff, *, live):
        self.model = Workflow().eval()
        self.model.load_state_dict(torch.load(path, weights_only=True,map_location="cpu"))
        self.identity = sha(path)
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=.001) if live else None
        self.cutoff = cutoff
        self.ledger = self.previous = self.stamp = self.answer = None

    def observe(self, record):
        stamp = record["timestamp"]
        vector = [record["co2_ppm"], record["temperature_c"]]
        if (type(stamp) is not int or any(type(v) not in (int,float) or not math.isfinite(v) for v in vector)
                or (self.stamp is not None and stamp <= self.stamp)):
            raise ValueError("Supply finite readings and strictly increasing integer timestamps")
        seconds = 300 if self.stamp is None else stamp-self.stamp
        if seconds>450:
            self.ledger=self.previous=None
        x = (torch.tensor([vector],dtype=torch.float32)-torch.tensor((500,22)))/torch.tensor((200,3))
        elapsed = x.new_tensor([min(seconds,450)/60])
        tick = time.perf_counter()
        with torch.no_grad():
            result = self.model.step(x,elapsed,self.ledger,trace=True)
        predicted = time.perf_counter()
        probability = float(result.logit.sigmoid()[0])
        self.answer = dict(timestamp=stamp,occupied=(probability>=self.cutoff) if result.accepted[0] else None,
                           probability=probability, prediction_ms=(predicted-tick)*1000, trace=result.attempts)
        if self.optimizer is not None and self.previous is not None:
            start = time.perf_counter()
            self.answer["feedback_loss"] = learn_from_observation(self.model,self.optimizer,self.previous,x)
            self.answer["feedback_ms"] = (time.perf_counter()-start)*1000
        self.previous = x, elapsed, self.ledger
        self.ledger, self.stamp = result.memory.detach(), stamp
        return self.answer

    def snapshot(self,path):
        previous = None
        if self.previous is not None:
            x, dt, ledger = self.previous
            previous = x, dt, asdict(ledger) if ledger else None
        save(path,dict(base_sha256=self.identity,model=self.model.state_dict(),cutoff=self.cutoff,
                       live=self.optimizer is not None, stamp=self.stamp, answer=self.answer,
                       ledger=asdict(self.ledger) if self.ledger else None, previous=previous))

    def restore(self,path):
        state = torch.load(path,weights_only=True,map_location="cpu")
        if (state["base_sha256"]!=self.identity or state["cutoff"]!=self.cutoff
                or state["live"]!=(self.optimizer is not None)):
            raise ValueError("Snapshot belongs to another checkpoint, cutoff or learning mode")
        self.model.load_state_dict(state["model"])
        self.stamp, self.answer = state["stamp"], state["answer"]
        self.ledger = Ledger(**state["ledger"]) if state["ledger"] else None
        if state["previous"] is not None:
            x, dt, ledger = state["previous"]
            self.previous = x, dt, Ledger(**ledger) if ledger else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("--cutoff",type=float,required=True)
    p.add_argument("--snapshot",type=Path)
    p.add_argument("--live",action="store_true")
    args=p.parse_args()
    if not 0<=args.cutoff<=1:
        p.error("cutoff must be between zero and one")
    torch.set_num_threads(1)
    stream=Stream(args.checkpoint,args.cutoff,live=args.live)
    if args.snapshot and args.snapshot.exists():
        stream.restore(args.snapshot)
    for line in sys.stdin:
        response=stream.observe(json.loads(line))
        if args.snapshot:
            stream.snapshot(args.snapshot)
        print(json.dumps(response,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
