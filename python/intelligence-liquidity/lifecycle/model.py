"""Deterministic CPU occupancy lifecycle. See README.md for the protocol."""
import argparse
import csv
import hashlib
import io
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import torch

reference_path = Path(__file__).resolve().parents[1] / "custom-ltc" / "model.py"
reference_spec = importlib.util.spec_from_file_location("prior_model", reference_path)
prior = importlib.util.module_from_spec(reference_spec)
reference_spec.loader.exec_module(prior)

SCHEMA = "occupancy-lifecycle-v1"
CONFIG = dict(seed=41, batch=256, cap=12, patience=2, min_gain=0.005,
              learning_rate=0.01, betas=[0.9, 0.999], epsilon=1e-8,
              threshold=0.5, promotion_gain=0.005, split=6500)
HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "custom-ltc" / "checkpoint.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return digest(prior.json_text(value).encode())


def atomic(path, value):
    payload = prior.json_text(value).encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return digest(payload)


def prepare(archive):
    rows = prior.read_rows(archive, prior.TRAIN)
    fit, validation = rows[:CONFIG["split"]], rows[CONFIG["split"]:]
    raw = torch.tensor([row[1] for row in fit])
    mean, scale = raw.mean(0), raw.std(0, correction=0)

    def pack(block):
        ids = list(range(len(block)))
        return prior.windows(block, ids, set(ids), mean, scale)

    data = dict(archive=prior.SHA, features=prior.FEATURES, mean=mean.tolist(),
                scale=scale.tolist(), window=prior.WINDOW, units=prior.UNITS,
                substeps=prior.SUBSTEPS, torch=str(torch.__version__),
                implementation=digest(Path(__file__).read_bytes()),
                reference=digest(Path(prior.__file__).read_bytes()))
    return pack(fit), pack(validation), data


def decision(value, best, waiting):
    gain = None if best is None else best - value
    improved = gain is None or gain >= CONFIG["min_gain"]
    return gain, improved, 0 if improved else waiting + 1


def trace(model, fit, ids, before, after, first_weight):
    source = fit[3]
    return dict(size=len(ids), first_target_csv_line=source[int(ids[0])] + 2,
                last_target_csv_line=source[int(ids[-1])] + 2,
                target_lines_sha256=fingerprint([source[int(ix)] + 2 for ix in ids]),
                loss_before=before, loss_after=after,
                bias_before=first_weight, bias_after=model.readout.bias.item(),
                first_label=int(fit[2][ids[0]]))


def train(name, fit, validation, data, output, resume=None):
    identity = dict(schema=SCHEMA, model=name, config=CONFIG, data=data)
    torch.manual_seed(CONFIG["seed"])
    shuffle = torch.Generator().manual_seed(CONFIG["seed"])
    model = prior.MODELS[name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"],
                                 betas=tuple(CONFIG["betas"]), eps=CONFIG["epsilon"])
    history, selected, batch = [], None, None
    if resume:
        snapshot = json.loads(resume.read_bytes())
        if snapshot["identity"] != identity or snapshot["stop"] is not None:
            raise ValueError("Snapshot identity differs or the run has already stopped")
        model = prior.load_model(name, snapshot["current"])
        optimizer = torch.optim.Adam(model.parameters())
        state = snapshot["adam"]
        state["state"] = {int(key): {field: torch.tensor(value) for field, value in values.items()}
                          for key, values in state["state"].items()}
        optimizer.load_state_dict(state)
        shuffle.set_state(torch.tensor(snapshot["shuffle"], dtype=torch.uint8))
        history, selected, batch = snapshot["history"], snapshot["selected"], snapshot["batch"]

    x, gaps, labels, _ = fit
    vx, vg, vy, _ = validation
    for epoch in range(len(history) + 1, CONFIG["cap"] + 1):
        model.train()
        loss_sum = 0.0
        for ids in torch.randperm(len(labels), generator=shuffle).split(CONFIG["batch"]):
            optimizer.zero_grad(set_to_none=True)
            loss = prior.LOSS(model(x[ids], gaps[ids]), labels[ids])
            loss.backward()
            capture = batch is None and name == "ltc"
            if capture:
                bias_before = model.readout.bias.item()
            optimizer.step()
            loss_sum += loss.item() * len(ids)
            if capture:
                with torch.no_grad():
                    after = prior.LOSS(model(x[ids], gaps[ids]), labels[ids]).item()
                batch = trace(model, fit, ids, loss.item(), after, bias_before)
        model.eval()
        value = prior.LOSS(prior.predict(model, vx, vg), vy).item()
        current = dict(weights=prior.tensor_state(model))
        gain, improved, waiting = decision(value, selected and selected["loss"],
                                           history[-1]["waiting"] if history else 0)
        if improved:
            selected = dict(**current, epoch=epoch, loss=value)
        stop = ("patience" if waiting >= CONFIG["patience"] else
                "epoch cap" if epoch == CONFIG["cap"] else None)
        feedback = dict(epoch=epoch, train_loss=loss_sum / len(labels), validation_loss=value,
                        improvement=gain, improved=improved, best_epoch=selected["epoch"],
                        waiting=waiting, stop=stop)
        history.append(feedback)
        state = optimizer.state_dict()
        state["state"] = {str(key): {field: tensor.tolist() for field, tensor in values.items()}
                          for key, values in state["state"].items()}
        snapshot = dict(identity=identity, history=history, selected=selected, current=current,
                        adam=state, shuffle=shuffle.get_state().tolist(), stop=stop, batch=batch)
        atomic(output / f"{name}-{epoch:02}.json", snapshot)
        print(prior.json_text(dict(model=name, resumed=bool(resume), **feedback)), flush=True)
        if stop:
            return snapshot
    raise ValueError("Snapshot has no remaining epochs")


def source_rows(archive, name):
    return list(csv.reader(io.StringIO(archive.read(name).decode())))[1:]


def evaluate(archive, data, entries):
    mean, scale = map(torch.tensor, (data["mean"], data["scale"]))
    scores = {}
    for file in (prior.TEST_A, prior.TEST_B):
        rows = prior.read_rows(archive, file)
        ids = list(range(len(rows)))
        kept = [ix for ix in ids if ix % 10 not in (2, 3, 4)]
        targets = set(kept[prior.WINDOW - 1:])
        modes = {}
        for mode, indices in (("recorded", ids), ("dropout_replay", kept)):
            x, gaps, labels, _ = prior.windows(rows, indices, targets, mean, scale)
            modes[mode] = {label: prior.metrics(prior.predict(prior.load_model(kind, entry), x, gaps),
                                               labels)
                           for label, (kind, entry) in entries.items()}
        scores[file] = modes
    return scores


def verify(archive_path, output):
    output.mkdir(parents=True, exist_ok=False)
    with prior.open_data(archive_path) as archive:
        fit, validation, data = prepare(archive)
        trained = {name: train(name, fit, validation, data, output) for name in prior.MODELS}
        full = trained["ltc"]
        if full["stop"] != "patience":
            raise AssertionError("Expected an observed LTC patience stop")
        resume_dir = output / "resume"
        resume_dir.mkdir()
        resume_point = output / "ltc-02.json"
        resumed = train("ltc", fit, validation, data, resume_dir, resume_point)
        final_name = f"ltc-{len(full['history']):02}.json"
        final_bytes = (output / final_name).read_bytes()
        if final_bytes != (resume_dir / final_name).read_bytes():
            raise AssertionError("Resumed snapshot differs")

        previous_bytes = PREVIOUS.read_bytes()
        previous = json.loads(previous_bytes)
        if (previous["archive_sha256"], previous["features"], previous["mean"], previous["scale"]) != (
                data["archive"], data["features"], data["mean"], data["scale"]):
            raise ValueError("Previous model uses different data or scaling")
        selected = json.loads(final_bytes)["selected"]
        frozen = {key: previous[key] for key in ("archive_sha256", "features", "mean", "scale",
                                                 "window", "schema", "units", "substeps")}
        frozen["runs"] = {"41": {"ltc": selected}}
        candidate_path = output / "candidate.json"
        candidate_hash = atomic(candidate_path, frozen)
        frozen = json.loads(candidate_path.read_bytes())
        vx, vg, vy, _ = validation
        previous_entry = previous["runs"]["41"]["ltc"]
        previous_loss = prior.LOSS(prior.predict(prior.load_model("ltc", previous_entry), vx, vg), vy).item()
        promote = previous_loss - selected["loss"] >= CONFIG["promotion_gain"]
        old_hash = digest(previous_bytes)
        active = output / "active.json"
        atomic(active, dict(sha256=old_hash))
        if promote:
            atomic(active, dict(sha256=candidate_hash))
        policy_hash = json.loads(active.read_bytes())["sha256"]
        atomic(active, dict(sha256=candidate_hash, replay=True))
        trial_hash = json.loads(active.read_bytes())["sha256"]

        raw = source_rows(archive, prior.TEST_A)[3:11]
        sample_path = HERE / "sample.csv"
        sample = prior.csv_rows(sample_path)
        if [row[:2] for row in sample] != [row[:2] for row in prior.read_rows(archive, prior.TEST_A)[3:11]]:
            raise AssertionError("Sample differs from the archive")
        now = "2015-02-02 14:30:00"
        def score_active(reference):
            selected_hash = json.loads(active.read_bytes())["sha256"]
            payload = {candidate_hash: candidate_path, old_hash: PREVIOUS}[selected_hash].read_bytes()
            if digest(payload) != selected_hash:
                raise ValueError("Active checkpoint does not match its hash")
            return prior.infer(json.loads(payload), sample, reference)

        trial = score_active(now)
        x, gaps, _, _ = prior.windows(sample, list(range(8)), {7},
                                      torch.tensor(data["mean"]), torch.tensor(data["scale"]))
        network = prior.load_model("ltc", frozen["runs"]["41"]["ltc"])
        with torch.no_grad():
            state = x.new_zeros((1, prior.UNITS))
            for reading, elapsed in zip(x.unbind(1)[:-1], gaps.unbind(1)[:-1]):
                state = network.advance(reading, state, elapsed)
            initial_state = state[0, 0].item()
            state = network.advance(x[:, -1], state, gaps[:, -1])
            last_update = dict(gap_minutes=gaps[0, -1].item(),
                               first_state_before=initial_state, first_state_after=state[0, 0].item())
        try:
            score_active("2015-02-02 14:35:00")
        except ValueError as error:
            stale = str(error)
        else:
            raise AssertionError("Expected a stale reading refusal")
        atomic(active, dict(sha256=old_hash))
        restored = json.loads(active.read_bytes())["sha256"]
        restored_bytes = PREVIOUS.read_bytes()
        if digest(restored_bytes) != restored:
            raise AssertionError("Restored checkpoint does not match active hash")
        rolled_back = score_active(now)

        entries = {name: (name, result["selected"]) for name, result in trained.items()}
        entries["previous_ltc"] = ("ltc", previous_entry)
        receipt = dict(
            identity=full["identity"], batch=full["batch"], histories={
                name: result["history"] for name, result in trained.items()},
            resume=dict(epoch=2, source_sha256=digest(resume_point.read_bytes()),
                        final_sha256=digest(final_bytes), exact_snapshot_bytes=True,
                        selected_weights_sha256=fingerprint(selected["weights"])),
            sources={name: dict(rows=len(rows), first_record=rows[0][0], last_record=rows[-1][0],
                                sha256=digest(archive.read(name)))
                     for name in (prior.TRAIN, prior.TEST_A, prior.TEST_B)
                     for rows in [source_rows(archive, name)]},
            split=dict(fit_csv_lines=[2, 6501], validation_csv_lines=[6502, 8144],
                       fit_targets=len(fit[2]), validation_targets=len(validation[2])),
            sample=dict(file=prior.TEST_A, records=[row[0] for row in raw],
                        csv_lines=[5, 12], final_raw_row=raw[-1],
                        sha256=digest(sample_path.read_bytes()), candidate=trial, stale=stale,
                        last_update=last_update,
                        rollback=rolled_back),
            promotion=dict(decision="promote" if promote else "refuse",
                           previous_validation_loss=previous_loss, candidate_validation_loss=selected["loss"],
                           before=old_hash, candidate=candidate_hash, after_policy=policy_hash,
                           trial_replay=trial_hash, restored=restored),
            held_out=evaluate(archive, data, entries))
        atomic(output / "receipt.json", receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    verification = commands.add_parser("verify")
    verification.add_argument("archive", type=Path)
    verification.add_argument("output", type=Path, help="A fresh output directory under /tmp")
    verification.add_argument("--expect", type=Path, help="Compare against the recorded receipt")
    resume = commands.add_parser("resume")
    resume.add_argument("archive", type=Path)
    resume.add_argument("snapshot", type=Path)
    resume.add_argument("output", type=Path)
    inference = commands.add_parser("infer")
    inference.add_argument("checkpoint", type=Path)
    inference.add_argument("csv", type=Path)
    inference.add_argument("--sha256", required=True)
    inference.add_argument("--now", required=True)
    args = parser.parse_args()
    try:
        if args.command == "verify":
            result = verify(args.archive, args.output)
            if args.expect and prior.json_text(result).encode() != args.expect.read_bytes():
                raise ValueError("Recorded receipt differs")
            print("Verified early stop, byte-identical resume, held-out scores, inference and rollback.")
        elif args.command == "resume":
            with prior.open_data(args.archive) as archive:
                fit, validation, data = prepare(archive)
                args.output.mkdir(parents=True, exist_ok=False)
                train("ltc", fit, validation, data, args.output, args.snapshot)
        else:
            payload = args.checkpoint.read_bytes()
            actual = digest(payload)
            if actual != args.sha256:
                raise ValueError("Checkpoint hash differs")
            result = prior.infer(json.loads(payload), prior.csv_rows(args.csv), args.now)
            print(prior.json_text(dict(checkpoint_sha256=actual, **result)))
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
