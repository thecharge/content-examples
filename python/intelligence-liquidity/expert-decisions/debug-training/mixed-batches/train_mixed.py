"""Check shuffled, reading-weighted batches without changing the original debug run."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train_debug import board, expert_loss, first_batch, fit_cutoff, fit_scale, measure
from experiment import Experts, auc, infer, occupancy, sha, windows


def pooled_windows(streams):
    groups = windows(streams, 300)
    length = max(x.shape[1] - 1 for x, _ in groups)
    samples, labels = [], []
    for x, y in groups:
        # Drop each sequence's original last row before padding. Missing rows
        # contribute no expert loss and come only after the real sequence.
        tail = length - x.shape[1] + 1
        samples.append(torch.nn.functional.pad(x[:, :-1], (0, 0, 0, tail), value=float("nan")))
        labels.append(torch.nn.functional.pad(y[:, :-1], (0, tail), value=0))
    return torch.cat(samples), torch.cat(labels)


def fit(fitting, development, lr, epochs, patience):
    torch.manual_seed(51)
    model = Experts()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    class_counts = torch.cat([s[2] for s in fitting]).bincount(minlength=2).float()
    weights = class_counts.mean() / class_counts
    x, y = pooled_windows(fitting)
    batch_count = math.ceil(len(x) / 128)
    best, selected, checkpoint, curve = -float("inf"), 0, None, []
    for epoch in range(1, epochs + 1):
        losses, total_readings, batch_stats = 0.0, 0, []
        for ids in torch.randperm(len(x)).tensor_split(batch_count):
            batch_x, batch_y = x[ids], y[ids]
            optimizer.zero_grad()
            state, outputs = None, []
            for i in range(batch_x.shape[1]):
                logits, state = model(batch_x[:, i], state)
                if i >= 3:
                    outputs.append(logits)
            scored_x, scored_y = batch_x[:, 3:], batch_y[:, 3:]
            valid = (scored_x.isfinite().unsqueeze(-2) & model.views).any(-1)
            valid_count = int(valid.sum())
            loss = expert_loss(model, torch.stack(outputs, dim=1), scored_x, scored_y, weights)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses += float(loss.detach()) * valid_count
            total_readings += valid_count
            batch_stats.append(dict(windows=len(ids), expert_readings=valid_count,
                                    occupied_fraction=float(scored_y.unsqueeze(-1).expand_as(valid)[valid].float().mean()),
                                    preclip_gradient=float(gradient)))
        model.eval()
        raw = infer(model, development)
        model.train()
        mean_expert_auc = sum(auc(development[2], values) for values in raw.T) / raw.shape[1]
        report = dict(epoch=epoch, training_loss=losses / total_readings,
                      validation_loss=float(expert_loss(model, raw, development[1], development[2], weights)),
                      validation_mean_expert_auc=mean_expert_auc,
                      validation_board_auc=auc(development[2], raw.sigmoid().mean(-1)), batches=batch_stats)
        curve.append(report)
        print(json.dumps({key: value for key, value in report.items() if key != "batches"}), flush=True)
        if mean_expert_auc > best:
            best, selected = mean_expert_auc, epoch
            checkpoint = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch - selected >= patience:
            break
    model.load_state_dict(checkpoint)
    model.eval()
    return model, selected, curve


def run(data, out, lr, epochs, patience):
    out.mkdir(parents=True, exist_ok=True)
    streams, archives = [], {}
    for h in range(1, 7):
        stream, hashes = occupancy(data, h, 14 if h <= 3 else 7)
        streams.append(stream)
        archives.update(hashes)
    dev = streams[3]
    middle = len(dev[0]) // 2
    development = tuple(value[:middle] for value in dev)
    calibration = tuple(value[middle:] for value in dev)
    model, selected, curve = fit(streams[:3], development, lr, epochs, patience)
    scales = fit_scale(infer(model, calibration), calibration[2])
    _, score, called, _ = board(model, development, scales)
    cutoff = fit_cutoff(score, called, development[2])
    results = {}
    for name, stream in zip(("H4 selection", "H4 calibration", "H5", "H6"),
                            (development, calibration, *streams[4:]), strict=True):
        raw, score, called, gate = board(model, stream, scales)
        results[name] = dict(half=measure(score, called, stream[2], .5),
                             selected_cutoff=measure(score, called, stream[2], cutoff),
                             raw_batch=first_batch(raw[:5], stream[2][:5]), **gate)
        print(name, results[name]["selected_cutoff"], flush=True)
    model_path = out / "mixed-experts.pt"
    torch.save(dict(weights=model.state_dict(), scales=scales, decision_cutoff=cutoff), model_path)
    record = dict(lr=lr, max_epochs=epochs, patience=patience, best_epoch=selected, curve=curve, results=results,
                  decision_cutoff=cutoff, scales=scales.tolist(), archives=archives,
                  code_sha256=sha(Path(__file__)), checkpoint_sha256=sha(model_path))
    (out / "mixed-results.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    if args.lr <= 0 or min(args.epochs, args.patience) < 1:
        parser.error("Learning rate, epochs and patience must be positive.")
    torch.set_num_threads(1)
    run(args.data, args.output, args.lr, args.epochs, args.patience)
