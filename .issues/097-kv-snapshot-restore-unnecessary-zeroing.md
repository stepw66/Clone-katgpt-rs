# Issue 097: KVSnapshot::restore() Zeroes Cache Beyond Snapshot Unnecessarily

## Severity: Medium
## Files: `katgpt-rs/src/transformer.rs` (L201-211)

## Description
After restoring snapshot data, `restore()` zeroes `layer.key[end..]` and `layer.value[end..]` for every layer. For large `block_size` and small snapshots, this zeroing is O(block_size × kv_dim) per layer — potentially much larger than the actual snapshot copy.

The comment on `KVCache::reset()` (L146-148) explicitly notes that zeroing is unnecessary because "each position is written before being read." The same invariant applies after restore — future forward passes will overwrite positions sequentially.

## Fix
Remove the zeroing from `restore()`, matching the rationale from `KVCache::reset()`:
```rust
pub fn restore(&mut self, snapshot: &KVSnapshot, config: &Config) {
    let kd = types::kv_dim(config);
    for (layer, snap_layer) in self.layers.iter_mut().zip(snapshot.layers.iter()) {
        let end = snapshot.pos * kd;
        layer.key[..end].copy_from_slice(&snap_layer.key);
        layer.value[..end].copy_from_slice(&snap_layer.value);
    }
}
```

## Impact
Medium — speculative decoding snapshots/restores per draft token. For models with block_size=1024 and kv_dim=256, this eliminates zeroing of ~256K elements per restore.
