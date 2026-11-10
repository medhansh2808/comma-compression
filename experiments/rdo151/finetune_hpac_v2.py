"""Fine-tune existing #151 HPAC; reuse train_entropy.py's format/inference code.

Selection uses a fixed-corrector integer-frequency proxy, NOT archive bytes.
All 600 target frames may be used: this is sequence-specific compression.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import train_entropy as base


def emit(out, filename, row):
    with (out / filename).open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(row, allow_nan=False), flush=True)


def reload_model(ref, canonical, record):
    from container import hpac2
    loaded = ref["renderer"].load_hpac(canonical, ref["device"])
    model = ref["model_class"](**ref["kwargs"]).to(ref["device"]).eval()
    missing = model.load_state_dict(loaded.state_dict(), strict=False)
    if set(missing.missing_keys) != {"m2", "b2", "e2"} or missing.unexpected_keys:
        raise ValueError("Unexpected deployed model layout")
    hpac2.unpack_record_share(record, model)
    deployed = dict(ref, model=model)
    if base.serialize(deployed, model) != (canonical, record):
        raise ValueError("Serialized model does not survive load/save")
    return deployed


def evaluate(ref, out, epoch):
    torch = ref["torch"]
    canonical, record = base.serialize(ref, ref["model"])
    deployed = reload_model(ref, canonical, record)
    # Check the training forward really represents serialized parameters.
    with torch.no_grad():
        for frame in (0, 199, 399, 599):
            a, _ = base.logits_for(ref, frame)
            b, _ = base.logits_for(deployed, frame)
            if not torch.equal(a, b):
                raise ValueError(f"Training/deployed logits differ at frame {frame}")
    total = 0.0
    started = time.monotonic()
    for frame in range(600):
        # probe([frame]) returns a 600-frame extrapolation; divide by 600.
        total += base.probe(deployed, [frame]) / 600
        if (frame + 1) % 100 == 0:
            elapsed = time.monotonic() - started
            print(json.dumps(dict(phase="measure", epoch=epoch, frames=frame+1,
                                  of=600, elapsed_seconds=elapsed,
                                  remaining_seconds=elapsed/(frame+1)*(599-frame))), flush=True)
    _, model_bytes = base.storage(ref, canonical, record)
    row = dict(epoch=epoch, rate_proxy_bytes=total,
               stored_entropy_model_bytes=model_bytes,
               combined_proxy_bytes=total+model_bytes,
               changed_hpac_bytes=sum(a != b for a,b in zip(canonical, ref["canonical"])),
               hpac_sha256=base.sha(canonical),
               temporal_state_identical=record == ref["record"],
               full_sequence=True, serialized_logits_verified=True,
               actual_archive_evaluated=False)
    emit(out, "measurements.jsonl", row)
    return row, canonical, record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--accum-frames", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output", type=Path,
                    default=base.ROOT / "experiments/rdo151/work/hpac_finetune_v2")
    args = ap.parse_args()
    if args.epochs < 1 or args.accum_frames < 1 or args.lr <= 0:
        ap.error("epochs, accum-frames and lr must be positive")
    out = args.output.expanduser().resolve()
    # setup() changes cwd, so resolve output first.
    out.mkdir(parents=True, exist_ok=False)
    ref = base.setup(args.device, base.TOKENS)
    torch = ref["torch"]
    from torch.nn import functional as F
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    base.check(ref)
    model = ref["model"]
    selected = []
    for name, p in model.named_parameters():
        # Isolate HPAC weights/biases. Keep frame and shared temporal state fixed.
        trainable = (name not in {"frame_embed.weight", "m2", "b2", "e2"}
                     and not name.endswith("exponent"))
        p.requires_grad_(trainable)
        if trainable:
            selected.append((name, p))
    if not selected:
        raise ValueError("No HPAC parameters selected")
    params = [p for _, p in selected]
    config = dict(device=args.device, epochs=args.epochs,
                  accum_frames=args.accum_frames, lr=args.lr, seed=args.seed,
                  archive_sha256=base.ARCHIVE_SHA, tokens_sha256=base.TOKEN_SHA,
                  trainable_parameters={name: p.numel() for name,p in selected},
                  objective="uniform token cross entropy; deployed full-sequence proxy selection",
                  proxy_limitations="fixed correction table; no native coding or termination measurement")
    (out / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    best, canonical, record = evaluate(ref, out, 0)
    best.update(is_reference=True)
    base.save_best(ref, out, canonical, record, best)
    baseline = best["combined_proxy_bytes"]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    updates = 0
    for epoch in range(1, args.epochs+1):
        order = rng.permutation(600)
        epoch_bits = 0.0
        for offset in range(0, 600, args.accum_frames):
            batch = order[offset:offset+args.accum_frames]
            optimizer.zero_grad(set_to_none=True)
            for frame in batch:
                logits, target = base.logits_for(ref, int(frame))
                # Ordinary NLL gives the correct rate objective. No extra
                # weighting of rare/expensive pixels changes that objective.
                loss = F.cross_entropy(logits, target) / math.log(2)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                epoch_bits += float(loss.detach())
                (loss / len(batch)).backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            if updates == 0 and float(norm) == 0:
                raise ValueError("Zero gradient: inspect quantized model backward path")
            optimizer.step()
            base.project(ref)
            updates += 1
            if updates == 1 or updates % 10 == 0:
                emit(out, "training.jsonl", dict(epoch=epoch, updates=updates,
                     frames_this_epoch=min(offset+len(batch),600),
                     mean_training_bits_per_token=epoch_bits/(offset+len(batch)),
                     gradient_norm=float(norm), elapsed_seconds=time.monotonic()-started))
        metrics, canonical, record = evaluate(ref, out, epoch)
        metrics["combined_proxy_change_bytes"] = metrics["combined_proxy_bytes"]-baseline
        metrics["is_reference"] = canonical == ref["canonical"] and record == ref["record"]
        # Keep every epoch, including candidates that lose the proxy comparison.
        folder = out / f"epoch_{epoch:03d}"
        folder.mkdir()
        base.save_best(ref, folder, canonical, record, metrics)
        if metrics["combined_proxy_bytes"] < best["combined_proxy_bytes"]:
            best = metrics
            base.save_best(ref, out, canonical, record, best)
        torch.save(dict(epoch=epoch, model=model.state_dict(),
                        optimizer=optimizer.state_dict(), config=config), out/"last.pt")
        emit(out, "epochs.jsonl", dict(**metrics, best_epoch=best["epoch"],
                                      elapsed_seconds=time.monotonic()-started))
    print("Training finished. best.json is a proxy selection; no new archive has been produced.", flush=True)


if __name__ == "__main__":
    main()
