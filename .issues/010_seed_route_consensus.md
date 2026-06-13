# Seed-Route Consensus for GPart

**Source**: Plan 257 (GPart Isometric Partition Adapter) — Deferred Idea 4
**Priority**: Low
**Blocked**: Yes — blocked on multi-node consensus layer
**Depends**: Plan 257, multi-node consensus layer

## Summary
Use GPart's seed-derived partition assignments as a deterministic routing key for consensus. Two nodes with the same seed must produce identical partition matrices — this property enables seed-route consensus for distributed inference validation.

## Acceptance Criteria
- [ ] Multi-node consensus layer must exist first
- [ ] Define seed-route protocol (seed → P → commitment → quorum verify)
- [ ] Implement seed exchange in ChainConsensus
- [ ] Test determinism across simulated nodes

## Notes
- GPart's `commitment()` already produces BLAKE3 hash of partition matrix
- Natural fit for quorum commit since seed → P is deterministic
