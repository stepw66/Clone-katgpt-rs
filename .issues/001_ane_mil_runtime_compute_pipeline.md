# Issue 001: ANE MIL Runtime Compute Pipeline

**Source:** Research 223 — maderix/ANE Distillation
**Status:** Deferred
**Blocked On:** Private API stability testing across macOS versions
**Priority:** Low

## What

Generate MIL text at runtime from `TransformerWeights` structs, compile via `_ANEInMemoryModelDescriptor.modelWithMILText:weights:optionsPlist:`. Eliminates .mlmodelc file dependency for ANE path.

## Why

- Truly modelless — weights live in-memory, compute pipeline generated on-the-fly
- maderix/ANE proves this works at 109M param scale
- Simpler than protobuf spec approach (Research 157)

## Blocker

- Uses private API (`_ANEInMemoryModelDescriptor`) — could break across macOS versions
- Risky for MIT-licensed katgpt-rs default path
- Needs stability testing across macOS 14/15/16

## When to Unblock

- After `ane_direct` feature (rane) is tested and stable
- After macOS API stability survey (check if private API has changed across 3 versions)

## Tasks

- [ ] Survey `_ANEInMemoryModelDescriptor` API stability across macOS 14/15/16
- [ ] Implement MIL string generator for microGPT forward pass
- [ ] Compile and verify ANE residency
- [ ] Benchmark: MIL runtime vs .mlmodelc file loading
- [ ] If stable: promote to `ane_direct` feature
