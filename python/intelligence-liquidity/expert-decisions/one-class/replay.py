"""Learn the next sensor reading in one occupancy class, then replay both classes."""

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from experiment import Experts, LiquidODE, NAMES, WIDTH, auc, balanced, occupancy, sha, windows


class Forecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("views", Experts().views)
        self.liquid = LiquidODE(20, len(NAMES))
        self.readout = nn.Parameter(torch.zeros(len(NAMES), WIDTH + 20, 10))
        self.bias = nn.Parameter(torch.zeros(len(NAMES), 10))

    def forward(self, x, state=None):
        present = x.isfinite()[:, None] & self.views
        values = torch.where(present, x[:, None], 0.)
        inputs = torch.cat((values, present.float()), dim=-1)
        ids = torch.arange(len(NAMES), device=x.device).repeat(len(x))
        if state is None:
            state = x.new_zeros(len(ids), WIDTH)
        nxt = self.liquid(inputs.flatten(0, 1), state, x.new_full((len(ids),), 5.), ids)
        state = torch.where(present.any(-1).flatten()[:, None], nxt, state * .9)
        features = torch.cat((inputs, state.reshape(len(x), len(NAMES), WIDTH)), -1)
        adjustment = torch.einsum("bew,ewf->bef", features, self.readout) + self.bias
        return values + adjustment, present, state


def retain(rows, label):
    keep = rows[2] == label
    times, inputs, states = (value[keep] for value in rows)
    gap = torch.ones(len(times), dtype=torch.bool)
    gap[1:] = times.diff() != 300
    # A change at the start of a retained segment could use a reading from
    # the excluded class. Drop those changes before training or calibration.
    inputs[gap, 5:] = float("nan")
    return times, inputs, states


def scale_features(streams):
    x = torch.cat([stream[1] for stream in streams])
    center = x.nanmedian(dim=0).values
    return torch.nanquantile((x - center).abs(), .90, dim=0).nan_to_num(nan=1.).clamp_min(.05)


def errors(prediction, present, future, scale):
    valid = present & future[:, None].isfinite()
    target = torch.where(valid, future[:, None], 0.)
    residual = torch.where(valid, (prediction - target) / scale, 0.)
    return F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none"), valid


def packed_sequences(streams):
    groups = windows(streams, 300)
    longest = max(x.shape[1] for x, _ in groups)
    return torch.cat([F.pad(x, (0, 0, 0, longest - x.shape[1]), value=float("nan")) for x, _ in groups])


@torch.no_grad()
def score_stream(model, stream, scale):
    timestamps, inputs, _ = stream
    result = inputs.new_full((len(inputs), len(NAMES)), float("nan"))
    previous = None
    state = None
    for i, row in enumerate(inputs):
        if i == 0 or timestamps[i] - timestamps[i-1] != 300:
            previous = None
            state = None
        if previous is not None:
            error, valid = errors(*previous, row[None], scale)
            readings = valid.sum(-1)
            result[i] = torch.where(readings > 0, error.sum(-1) / readings.clamp_min(1), float("nan"))[0]
        prediction, present, state = model(row[None], state)
        previous = (prediction, present)
    votes = result.isfinite().sum(-1)
    novelty = result.nanmedian(dim=-1).values
    return novelty, votes >= 3


def fit(streams, selection, scale, rate, epochs, patience):
    torch.manual_seed(51)
    model = Forecast()
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate)
    samples = packed_sequences(streams)
    best, checkpoint, selected, curve = float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        training_cost, trained_fields = 0., 0
        for ids in torch.randperm(len(samples)).split(96):
            batch = samples[ids]
            optimizer.zero_grad()
            state, total, count = None, batch.new_zeros(()), 0
            for i in range(batch.shape[1] - 1):
                prediction, present, state = model(batch[:, i], state)
                if i >= 3:
                    loss, valid = errors(prediction, present, batch[:, i+1], scale)
                    total = total + loss.sum()
                    count += int(valid.sum())
            (total / count).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            training_cost += float(total.detach())
            trained_fields += count
        novelty, scored = score_stream(model, selection, scale)
        error = float(novelty[scored].mean())
        report = dict(epoch=epoch, training_loss=training_cost/trained_fields, validation_novelty=error)
        curve.append(report)
        print(json.dumps(report), flush=True)
        if error < best:
            best, selected = error, epoch
            checkpoint = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch - selected >= patience:
            break
    model.load_state_dict(checkpoint)
    model.eval()
    return model, selected, curve, len(samples)


def measure(novelty, scored, y, known_class, threshold):
    anomaly = novelty > threshold
    occupied = anomaly if known_class == 0 else ~anomaly
    right = occupied.eq(y.bool()) & scored
    signed = novelty[scored] if known_class == 0 else -novelty[scored]
    return dict(rows=len(y), called=int(scored.sum()), withheld=int((~scored).sum()),
                right=int(right.sum()), balanced_accuracy=balanced(y,right), auc_on_answered=auc(y[scored],signed),
                **{name: dict(rows=int((y == c).sum()), right=int((right & (y == c)).sum()),
                              median_novelty=float(novelty[(y == c) & scored].median()))
                   for c,name in enumerate(("vacant","occupied"))})


def run(input_dir, output_dir, known_class, epochs, patience):
    output_dir.mkdir(parents=True, exist_ok=True)
    streams, archives = [], {}
    for home in range(1, 7):
        stream, hashes = occupancy(input_dir, home, 14 if home <= 3 else 7)
        streams.append(stream)
        archives.update(hashes)
    fitting = [retain(h, known_class) for h in streams[:3]]
    dev = streams[3]
    mid = len(dev[0]) // 2
    selection = retain(tuple(v[:mid] for v in dev), known_class)
    calibration = retain(tuple(v[mid:] for v in dev), known_class)
    scale = scale_features(fitting)
    lr = .003
    model, selected, curve, sequences = fit(fitting, selection, scale, lr, epochs, patience)
    threshold_scores, valid = score_stream(model, calibration, scale)
    threshold = float(torch.quantile(threshold_scores[valid], .90, interpolation="higher"))
    results = {}
    for name, stream in zip(("H5", "H6"), streams[4:], strict=True):
        novelty, scored = score_stream(model, stream, scale)
        results[name] = measure(novelty, scored, stream[2], known_class, threshold)
        print(name, results[name], flush=True)
    model_path = output_dir / "forecast.pt"
    torch.save(dict(weights=model.state_dict(), scale=scale, threshold=threshold, known_class=known_class), model_path)
    record = dict(known_class=known_class, fit_rows=[len(h[0]) for h in fitting], fit_sequences=sequences,
                  learning_rate=lr, max_epochs=epochs, patience=patience,
                  selection_rows=len(selection[0]), calibration_rows=len(calibration[0]),
                  calibration_scored=int(valid.sum()), calibration_above=int((threshold_scores[valid] > threshold).sum()),
                  calibration_quantile=.90, threshold=threshold, best_epoch=selected, curve=curve, results=results,
                  code_sha256=sha(Path(__file__)), checkpoint_sha256=sha(model_path), archives=archives)
    (output_dir / "results.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("known_class", type=int, choices=(0, 1), help="0 vacant, 1 occupied")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()
    if min(args.epochs, args.patience) < 1:
        parser.error("Epochs and patience must be positive.")
    torch.set_num_threads(1)
    run(args.archives, args.output, args.known_class, args.epochs, args.patience)
