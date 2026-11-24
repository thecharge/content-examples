# Onboard AMD GPU

## What this session actually exposes

The observed CPU is an AMD Ryzen 7 PRO 8840HS with Radeon 780M graphics.
PCI inspection reports AMD Phoenix3, device `1002:1900`.
The Linux kernel is `6.17.0-1033-oem`.

The Python environment used for verification reports:

```text
torch: 2.7.1+cpu
torch.version.hip: None
torch.cuda.is_available(): False
/dev/kfd: absent
/dev/dri: absent
rocminfo: not found
```

These are observations of the execution environment. They do not establish that
the host driver is broken or that the physical GPU is unusable elsewhere.
[device.py](device.py) reproduces the probe without installing anything.

## Why CPU is currently required

There are three independent issues:

1. The running PyTorch wheel contains CPU code, not ROCm kernels.
2. This session exposes neither the compute device nor DRM render devices.
3. The inherited model runtime allocates tensors on CPU and rejects a GPU model.

The third issue needs a device-aware implementation and verification even after
the first two are resolved. Removing the guard alone would leave incompatible
tensor allocations, masks, generators and checkpoint paths.

Nothing about a liquid hypergraph inherently requires a CPU.
The SQLite journal and structural controller can remain on CPU while neural
tensor operations run on a GPU. Whether that helps this small model needs measurement.
Kernel launches, small batches, host synchronization and durable event writes can
dominate the workload. This experiment has no measured AMD speedup.

## What AMD's documentation establishes

The [current unified ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
lists Radeon 780M and the `gfx1103` architecture among APU entries.
That is a reason to investigate this hardware, not to dismiss it categorically.
The exact processor, operating-system, kernel and framework combination still needs
matching to the selected ROCm release. The matrix retrieved for this work lists
newer Ubuntu kernel combinations than this session's 6.17 OEM kernel.

The [older Ryzen ROCm 7.2.1 matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)
lists different APUs and architectures. Its absence there cannot be treated as
proof that every later ROCm release excludes Radeon 780M.

[PyTorch's HIP notes](https://docs.pytorch.org/docs/2.15/notes/hip.html)
explain that ROCm builds reuse the `torch.cuda` API surface.
Seeing that namespace in a probe does not mean an NVIDIA GPU is required.

## What a verified device port still needs

- Expose usable compute/render devices to the execution environment.
- Select a compatible host driver, ROCm release and HIP-enabled PyTorch build.
- Move model inputs, masks, recurrent/delta state and fitting allocations together.
- Keep structural stages synchronized with device work and rebuild optimizer state.
- Keep durable snapshots portable through CPU tensor serialization.
- Check forward, gradient, fitting, growth, sharing, recovery and monitoring behavior.
- Measure the same-data accuracy and end-to-end latency, including synchronization.

This work made no driver installation, kernel change or device passthrough change.
`python device.py --require rocm` and `--device rocm` refuse execution explicitly.
GPU model execution remains unverified.
