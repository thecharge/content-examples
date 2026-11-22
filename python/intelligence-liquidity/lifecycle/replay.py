# Replay the run starting with the saved best at epoch 7.
# Change PATIENCE from 2 to 1 and run again.
PATIENCE = 2
MIN_GAIN = 0.005

best_epoch = 7
best = 0.0919579491019249
waiting = 0
checks = {8: 0.10190725326538086, 9: 0.10804963111877441}

for epoch, loss in checks.items():
    gain = best - loss
    if gain >= MIN_GAIN:
        best, best_epoch, waiting = loss, epoch, 0
    else:
        waiting += 1
    print(f"epoch={epoch} gain={gain:+.4f} waiting={waiting}")
    if waiting >= PATIENCE:
        print(f"stop at {epoch}, keep {best_epoch}")
        break
