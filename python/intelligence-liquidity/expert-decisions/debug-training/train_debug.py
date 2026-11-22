"""Train a separate checkpoint and diagnose optimization on occupancy readings."""

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from experiment import Experts, auc, balanced, choose, combine, infer, occupancy, sha, windows


def expert_loss(model, logits, x, y, weights=None):
    observed = (x.isfinite().unsqueeze(-2) & model.views).any(-1)
    truth = y.unsqueeze(-1).expand_as(logits)
    loss = F.binary_cross_entropy_with_logits(logits, truth.float(), reduction="none")
    if weights is not None:
        loss = loss * weights[truth]
    return (loss * observed).sum() / observed.sum().clamp_min(1)


def first_batch(logits, y):
    return dict(labels=y.tolist(), logits=logits.tolist(), probabilities=logits.sigmoid().tolist())


def tiny_batch(model_factory, fitting, lr):
    times, readings, labels = fitting[0]
    for start in range(len(labels) - 4):
        end = start + 5
        x, y = readings[start:end], labels[start:end]
        if (times[start:end].diff() == 300).all() and x.isfinite().all() and int(y.sum()) in (2, 3):
            break
    else:
        raise ValueError("Need five complete consecutive readings with both labels")
    torch.manual_seed(51)
    model = model_factory()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    loss_curve = []
    gradient = []
    before = None
    for step in range(100):
        optimizer.zero_grad()
        state = None
        outputs = []
        for row in x:
            score, state = model(row[None], state)
            outputs.append(score[0])
        logits = torch.stack(outputs)
        if step == 0:
            before = first_batch(logits.detach(), y)
        loss = expert_loss(model, logits, x, y)
        loss.backward()
        if step < 2:
            gradient.append(dict(head=float(model.read.grad.norm()), liquid=float(model.liquid.target.grad.norm())))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        loss_curve.append(float(loss.detach()))
    with torch.no_grad():
        logits = infer(model, (times[start:end], x, y))
        truth = y.bool()
        after = first_batch(logits, y)
        return dict(lr=lr, first_row=start, samples=len(y), steps=100, before=before, after=after,
                    losses=loss_curve, final_loss=float(expert_loss(model, logits, x, y)),
                    expert_correct=logits.ge(0).eq(truth[:, None]).sum(0).tolist(),
                    board_correct=int(logits.sigmoid().mean(-1).ge(.5).eq(truth).sum()),
                    gradient_first_two_steps=gradient,
                    parameter_change_l2={name: float((p-initial[name]).norm()) for name,p in model.named_parameters()})


def fit(fitting, selection, lr, epochs, patience):
    torch.manual_seed(51)
    model = Experts()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    counts = torch.cat([stream[2] for stream in fitting]).bincount(minlength=2).float()
    weights = counts.mean() / counts
    groups = windows(fitting, 300)
    best, selected, checkpoint, curve = -float("inf"), 0, None, []
    for epoch in range(1, epochs + 1):
        batch_losses = []
        for x, y in groups:
            for ids in torch.randperm(len(x)).split(128):
                optimizer.zero_grad()
                state, outputs = None, []
                for i in range(x.shape[1] - 1):
                    logits, state = model(x[ids, i], state)
                    if i >= 3:
                        outputs.append(logits)
                loss = expert_loss(model, torch.stack(outputs, dim=1), x[ids, 3:-1], y[ids, 3:-1], weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                batch_losses.append(float(loss.detach()))
        raw = infer(model, selection)
        expert_auc = [auc(selection[2], col) for col in raw.T]
        score = sum(expert_auc) / len(expert_auc)
        stats = dict(epoch=epoch, training_loss=sum(batch_losses) / len(batch_losses),
                     validation_loss=float(expert_loss(model, raw, selection[1], selection[2], weights)),
                     validation_mean_expert_auc=score, validation_board_auc=auc(selection[2], raw.sigmoid().mean(-1)))
        curve.append(stats)
        print(json.dumps(stats), flush=True)
        if score > best:
            best, selected = score, epoch
            checkpoint = {key: p.detach().clone() for key, p in model.state_dict().items()}
        if epoch - selected >= patience:
            break
    model.load_state_dict(checkpoint)
    model.eval()
    return model, selected, curve


def fit_scale(raw, y):
    """Fit positive slopes with every intercept fixed to zero."""
    log_scale = torch.zeros(raw.shape[1], requires_grad=True)
    optimizer = torch.optim.LBFGS([log_scale], max_iter=60, line_search_fn="strong_wolfe")
    target = y[:, None].expand_as(raw).float()

    def loss():
        optimizer.zero_grad()
        slope = log_scale.exp()
        value = F.binary_cross_entropy_with_logits(raw * slope, target) + .005 * (slope-1).square().mean()
        value.backward()
        return value

    optimizer.step(loss)
    return log_scale.detach().exp()


def board(model, stream, scales):
    stamps, x, _ = stream
    raw = infer(model, stream)
    p = (raw * scales).sigmoid()
    coverage = (x.isfinite()[:, None] & model.views).sum(-1) / model.views.sum(-1)
    _, _, acted, _, curious = choose(p, coverage, stamps)
    score, called = combine(p, acted)
    return raw, score, called, dict(expert_proposals=int(acted.sum()), bonus_claims=int(curious.sum()))


def fit_cutoff(score, called, y):
    cutoffs = torch.unique(torch.cat((score[called], score.new_tensor([.5, 1.]))))
    correct = (score >= cutoffs[:, None]).eq(y.bool()) & called
    scores = (correct[:, y == 0].float().mean(-1) + correct[:, y == 1].float().mean(-1)) / 2
    return float(cutoffs[scores.argmax()])


def measure(score, called, labels, cutoff):
    right = (score >= cutoff).eq(labels.bool()) & called
    groups = {name: dict(rows=int((labels == c).sum()), correct=int((right & (labels == c)).sum()),
                         mean_score=float(score[labels == c].mean()))
              for c, name in enumerate(("vacant", "occupied"))}
    return dict(cutoff=cutoff, balanced_accuracy=balanced(labels, right), accuracy=float(right.float().mean()),
                withheld=int((~called).sum()), auc=auc(labels, score), **groups)


def run(data, out, lr, epochs, patience):
    out.mkdir(parents=True, exist_ok=True)
    streams, archives = [], {}
    for home in range(1, 7):
        stream, hashes = occupancy(data, home, 14 if home <= 3 else 7)
        streams.append(stream)
        archives.update(hashes)
    fitting = streams[:3]
    dev = streams[3]
    middle = len(dev[0]) // 2
    selection = tuple(p[:middle] for p in dev)
    calibration = tuple(p[middle:] for p in dev)
    tiny = [tiny_batch(Experts, fitting, step) for step in (lr, .02)]
    for trial in tiny:
        print("tiny batch", trial["lr"], "loss", trial["losses"][0], "to", trial["final_loss"],
              "correct", trial["board_correct"], "/", trial["samples"], "per expert", trial["expert_correct"],
              "before", trial["before"]["probabilities"], "after", trial["after"]["probabilities"], flush=True)
    model, epoch, curve = fit(fitting, selection, lr, epochs, patience)
    scale = fit_scale(infer(model, calibration), calibration[2])
    _, score, called, _ = board(model, selection, scale)
    cutoff = fit_cutoff(score, called, selection[2])
    reports = {}
    for name, stream in zip(("H4 selection", "H4 calibration", "H5", "H6"),
                            (selection, calibration, *streams[4:]), strict=True):
        raw, score, called, gate = board(model, stream, scale)
        reports[name] = dict(half=measure(score, called, stream[2], .5),
                             selected_cutoff=measure(score, called, stream[2], cutoff),
                             raw_batch=first_batch(raw[:5], stream[2][:5]), **gate)
        print(name, reports[name]["selected_cutoff"], flush=True)
    model_file = out / "debug-experts.pt"
    torch.save(dict(weights=model.state_dict(), scales=scale, decision_cutoff=cutoff), model_file)
    receipt = dict(lr=lr, max_epochs=epochs, patience=patience, best_epoch=epoch, curve=curve,
                   decision_cutoff=cutoff, scales=scale.tolist(), tiny=tiny, archives=archives,
                   code_sha256=sha(Path(__file__)), checkpoint_sha256=sha(model_file), results=reports)
    (out / "debug-results.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


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
