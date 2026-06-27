//! KV cache types for autoregressive generation, paged branching, and Raven routing.
//!
//! All types here are pure data + allocator helpers — no forward logic.

use katgpt_core::types::{self, Config};

/// KV cache for a single layer (autoregressive generation).
pub struct KVCache {
    pub key: Vec<f32>,   // [block_size, kv_dim] where kv_dim = n_kv_head * head_dim
    pub value: Vec<f32>, // [block_size, kv_dim]
}

impl KVCache {
    pub fn new(config: &Config) -> Self {
        let kvd = types::kv_dim(config);
        Self {
            key: vec![0.0; config.block_size * kvd],
            value: vec![0.0; config.block_size * kvd],
        }
    }

    pub fn reset(&mut self) {
        // No-op: each position is written before being read, so stale data
        // from previous sequences is never observed. Avoids O(block_size × kv_dim) zeroing.
    }
}

/// Multi-layer KV cache: one KVCache per transformer layer.
pub struct MultiLayerKVCache {
    pub layers: Vec<KVCache>,
    /// Highest position written + 1 across all layers, for efficient snapshot.
    fill_pos: usize,
}

impl MultiLayerKVCache {
    pub fn new(config: &Config) -> Self {
        let mut layers = Vec::with_capacity(config.n_layer);
        layers.extend((0..config.n_layer).map(|_| KVCache::new(config)));
        Self {
            layers,
            fill_pos: 0,
        }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.fill_pos = 0;
    }

    /// Update fill_pos tracker. Call after writing to the cache at a position.
    pub fn advance_pos(&mut self, pos: usize) {
        self.fill_pos = self.fill_pos.max(pos + 1);
    }

    /// Get the tracked fill position (highest position written + 1).
    pub fn fill_pos(&self) -> usize {
        self.fill_pos
    }

    /// Snapshot KV cache state up to position `pos`.
    /// Copies only filled slots [0..pos) per layer — cheap at our model scale.
    pub fn snapshot(&self, pos: usize, config: &Config) -> KVSnapshot {
        let kd = types::kv_dim(config);
        let end = pos * kd;
        // Pre-allocate outer Vec to avoid collect() reallocation jitter.
        // Called per-speculation-step (step.rs L149/L354/L487/L649/L1083).
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(KVLayerSnapshot {
                key: layer.key[..end].to_vec(),
                value: layer.value[..end].to_vec(),
            });
        }
        KVSnapshot { pos, layers }
    }

    /// Restore KV cache from a snapshot.
    /// Writes snapshot data back. No zeroing needed — each position is written before being read.
    pub fn restore(&mut self, snapshot: &KVSnapshot, config: &Config) {
        let kd = types::kv_dim(config);
        // Hoist loop-invariant `end` out of the per-layer loop.
        let end = snapshot.pos * kd;
        for (layer, snap_layer) in self.layers.iter_mut().zip(snapshot.layers.iter()) {
            layer.key[..end].copy_from_slice(&snap_layer.key);
            layer.value[..end].copy_from_slice(&snap_layer.value);
        }
    }
}

/// Cheap snapshot of KV cache state up to position `pos`.
/// Only copies filled slots [0..pos) per layer, not the entire block_size buffer.
pub struct KVSnapshot {
    pub pos: usize,
    pub layers: Vec<KVLayerSnapshot>,
}

/// Per-layer snapshot of KV cache data.
pub struct KVLayerSnapshot {
    pub key: Vec<f32>,   // [pos * kv_dim]
    pub value: Vec<f32>, // [pos * kv_dim]
}

/// Preload drafter's KV cache with target's pre-computed key/value pairs.
///
/// Copies target's KV for positions [0..pos) into drafter's cache.
/// This enables cross-attention: the drafter attends to the target's past KV
/// instead of computing its own from scratch.
///
/// Only active when `target_kv_dim == draft_kv_dim` (dimensions must match).
/// When dimensions don't match, silently returns (drafter computes its own KV).
///
/// Hybrid behavior after preload:
/// - Past positions [0..pos): read from preloaded target KV
/// - New positions [pos..]: computed by drafter during forward pass
pub fn preload_kv_cache(
    draft_cache: &mut MultiLayerKVCache,
    target_cache: &MultiLayerKVCache,
    pos: usize,
    target_config: &Config,
    draft_config: &Config,
) {
    let target_kv_dim = types::kv_dim(target_config);
    let draft_kv_dim = types::kv_dim(draft_config);

    // Dimension guard: can only share when kv_dim matches
    if target_kv_dim != draft_kv_dim {
        return;
    }

    // Layer guard: can only share layers that exist in both caches
    let min_layers = draft_cache.layers.len().min(target_cache.layers.len());

    // Copy KV for positions [0..pos) for each shared layer
    let copy_len = pos * target_kv_dim;
    if copy_len > 0 {
        for layer_idx in 0..min_layers {
            let draft_layer = &mut draft_cache.layers[layer_idx];
            let target_layer = &target_cache.layers[layer_idx];
            draft_layer.key[..copy_len].copy_from_slice(&target_layer.key[..copy_len]);
            draft_layer.value[..copy_len].copy_from_slice(&target_layer.value[..copy_len]);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Paged KV cache — DDTree branch exploration (copy-on-write fork)
// ──────────────────────────────────────────────────────────────────────────

/// Page size in tokens (tuneable, must be power of 2).
pub const PAGE_SIZE: usize = 16;

/// Paged KV cache for DDTree branch exploration.
/// Allocates memory in fixed-size pages with copy-on-write fork.
///
/// Page layout per page: `[K_data | V_data]` where each segment is `PAGE_SIZE * kv_dim` floats.
/// This enables sharing prefix pages between branches without cloning data.
pub struct PagedKVCache {
    /// Pool of pages. Each page: `[PAGE_SIZE * kv_dim * 2]` floats (K then V).
    pub pages: Vec<Vec<f32>>,
    /// Per-layer page tables. `layer_page_tables[layer][seq_idx]` = vec of page indices.
    pub layer_page_tables: Vec<Vec<Vec<usize>>>,
    /// Free list of page indices for reuse.
    pub free_pages: Vec<usize>,
    /// Dimension of each KV entry (`n_kv_head * head_dim`).
    pub kv_dim: usize,
    /// Total pages ever allocated (monotonically increasing).
    pub total_pages: usize,
    /// Reusable scratch: per-layer page deficits (cleared + refilled each call).
    pub deficits: Vec<usize>,
    /// Reusable scratch: per-layer new page indices (cleared + refilled each call).
    pub new_pages: Vec<Vec<usize>>,
    /// Reusable scratch: flat buffer for all newly allocated pages in `ensure_pages()`.
    pub all_new_buf: Vec<usize>,
    /// Per-page reference counts for O(1) rollback (replaces HashSet scan).
    pub page_ref_counts: Vec<u32>,
    /// Reusable scratch: drained page indices awaiting recycle in `rollback()`.
    pub rollback_removed: Vec<usize>,
}

impl PagedKVCache {
    /// Create a new paged KV cache.
    /// `max_sequences`: initial number of sequence slots (can grow via fork).
    pub fn new(config: &Config, max_sequences: usize) -> Self {
        let kvd = types::kv_dim(config);
        let initial_pages_per_layer = config.block_size / PAGE_SIZE + 1;

        Self {
            pages: (0..initial_pages_per_layer * config.n_layer)
                .map(|_| vec![0.0; PAGE_SIZE * kvd * 2])
                .collect(),
            layer_page_tables: (0..config.n_layer)
                .map(|_| {
                    (0..max_sequences)
                        .map(|_| Vec::with_capacity(initial_pages_per_layer))
                        .collect()
                })
                .collect(),
            free_pages: Vec::with_capacity(initial_pages_per_layer * config.n_layer),
            kv_dim: kvd,
            total_pages: initial_pages_per_layer * config.n_layer,
            deficits: Vec::with_capacity(config.n_layer),
            new_pages: vec![Vec::new(); config.n_layer],
            all_new_buf: Vec::with_capacity(initial_pages_per_layer * config.n_layer),
            page_ref_counts: vec![config.n_layer as u32; initial_pages_per_layer * config.n_layer],
            rollback_removed: Vec::with_capacity(initial_pages_per_layer),
        }
    }

    /// Allocate a new page. Reuse from free list or grow the pool.
    pub fn alloc_page(&mut self) -> usize {
        match self.free_pages.pop() {
            Some(idx) => {
                self.pages[idx].fill(0.0);
                self.page_ref_counts[idx] = 1;
                idx
            }
            None => {
                self.pages.push(vec![0.0; PAGE_SIZE * self.kv_dim * 2]);
                self.page_ref_counts.push(1);
                let idx = self.total_pages;
                self.total_pages += 1;
                idx
            }
        }
    }

    /// Ensure sequence `seq_idx` has enough pages to cover position `pos` for all layers.
    pub fn ensure_pages(&mut self, seq_idx: usize, pos: usize) {
        let pages_needed = pos / PAGE_SIZE + 1;

        // Grow sequence slots if needed (no page allocation, just empty vecs)
        for layer_tables in &mut self.layer_page_tables {
            while seq_idx >= layer_tables.len() {
                layer_tables.push(Vec::new());
            }
        }

        // Collect how many new pages each layer needs (reuse scratch buffer)
        self.deficits.clear();
        for lt in &self.layer_page_tables {
            self.deficits
                .push(pages_needed.saturating_sub(lt[seq_idx].len()));
        }

        // Allocate all pages upfront (reuse scratch buffer via take+put-back
        // to avoid borrow conflict with alloc_page)
        let total_deficit: usize = self.deficits.iter().sum();
        let mut buf = std::mem::take(&mut self.all_new_buf);
        buf.clear();
        buf.reserve(total_deficit);
        for _ in 0..total_deficit {
            buf.push(self.alloc_page());
        }

        // Partition into per-layer lists and assign
        self.new_pages.resize_with(self.deficits.len(), Vec::new);
        let mut offset = 0;
        for (i, &deficit) in self.deficits.iter().enumerate() {
            self.new_pages[i].clear();
            self.new_pages[i].extend_from_slice(&buf[offset..offset + deficit]);
            offset += deficit;
        }
        self.all_new_buf = buf;

        // Assign new pages to each layer's page table
        for (layer_tables, pages) in self.layer_page_tables.iter_mut().zip(self.new_pages.iter()) {
            layer_tables[seq_idx].extend(pages.iter().copied());
        }
    }

    /// Write K and V for a token position in a specific layer.
    /// Layout per page: `[K_data | V_data]` where each is `PAGE_SIZE * kv_dim` floats.
    #[inline]
    pub fn write_kv(&mut self, layer_idx: usize, seq_idx: usize, pos: usize, k: &[f32], v: &[f32]) {
        let page_local = pos % PAGE_SIZE;
        let page_list_idx = pos / PAGE_SIZE;
        let pidx = self.layer_page_tables[layer_idx][seq_idx][page_list_idx];
        let page = &mut self.pages[pidx];
        let kv_page_size = PAGE_SIZE * self.kv_dim;
        let k_off = page_local * self.kv_dim;
        let v_off = kv_page_size + page_local * self.kv_dim;
        page[k_off..k_off + self.kv_dim].copy_from_slice(k);
        page[v_off..v_off + self.kv_dim].copy_from_slice(v);
    }

    /// Read K and V for a token position in a specific layer.
    #[inline]
    pub fn read_kv(
        &self,
        layer_idx: usize,
        seq_idx: usize,
        pos: usize,
        k: &mut [f32],
        v: &mut [f32],
    ) {
        let page_local = pos % PAGE_SIZE;
        let page_list_idx = pos / PAGE_SIZE;
        let pidx = self.layer_page_tables[layer_idx][seq_idx][page_list_idx];
        let page = &self.pages[pidx];
        let kv_page_size = PAGE_SIZE * self.kv_dim;
        let k_off = page_local * self.kv_dim;
        let v_off = kv_page_size + page_local * self.kv_dim;
        k.copy_from_slice(&page[k_off..k_off + self.kv_dim]);
        v.copy_from_slice(&page[v_off..v_off + self.kv_dim]);
    }

    /// Fork a sequence with copy-on-write semantics.
    /// Shares prefix pages up to `fork_at_pos`, allocates new pages on demand after fork.
    /// Returns the new sequence index.
    pub fn fork(&mut self, seq_idx: usize, fork_at_pos: usize) -> usize {
        let fork_page = fork_at_pos / PAGE_SIZE;
        let new_seq = self.layer_page_tables[0].len();

        for layer_tables in &mut self.layer_page_tables {
            let source = &layer_tables[seq_idx];
            let shared_pages = source[..fork_page.min(source.len())].to_vec();
            // Increment ref counts for shared pages
            for &pidx in &shared_pages {
                self.page_ref_counts[pidx] += 1;
            }
            layer_tables.push(shared_pages);
        }

        new_seq
    }

    /// Rollback a sequence to a given position, freeing exclusive pages.
    ///
    /// Truncates page tables to keep only pages covering positions `[0..rollback_to_pos)`.
    /// Pages that are exclusively owned by this sequence (not referenced by any other
    /// sequence in any layer) are returned to the free list for reuse.
    ///
    /// This is the "page table CoW rollback" — no data is copied, only page table
    /// entries are manipulated and exclusive pages are recycled.
    pub fn rollback(&mut self, seq_idx: usize, rollback_to_pos: usize) {
        let keep_count = rollback_to_pos / PAGE_SIZE;

        // Truncate page tables and decrement ref counts for dropped pages.
        // Pages with ref count == 0 go to the free list.
        for layer_tables in &mut self.layer_page_tables {
            if seq_idx >= layer_tables.len() {
                continue;
            }
            let table = &mut layer_tables[seq_idx];
            self.rollback_removed.clear();
            self.rollback_removed.extend(table.drain(keep_count..));
            for pidx in self.rollback_removed.drain(..) {
                self.page_ref_counts[pidx] -= 1;
                if self.page_ref_counts[pidx] == 0 {
                    self.free_pages.push(pidx);
                }
            }
        }
    }

    /// Reset all sequences and free all pages.
    pub fn reset(&mut self) {
        for layer_tables in &mut self.layer_page_tables {
            for table in layer_tables.iter_mut() {
                self.free_pages.append(table);
            }
        }
        // Reset all ref counts to 0
        self.page_ref_counts.fill(0);
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Raven RSM — Routing Slot Memory (O(1) KV replacement for draft model)
// ──────────────────────────────────────────────────────────────────────────

/// Raven Routing Slot Memory — O(1) KV replacement for the draft model.
///
/// Fixed-size `[num_slots × kv_dim]` memory updated via sparse Top-K routing.
/// Unselected slots are completely frozen — perfect for preserving struct
/// definitions and imports while churning through syntax tokens.
pub struct RavenKVCache {
    // ── Vec fields first (ptr+len+cap = 24 bytes, 8-byte aligned) ──
    /// Key memory: [num_slots × kv_dim]
    pub keys: Vec<f32>,
    /// Value memory: [num_slots × kv_dim]
    pub values: Vec<f32>,
    // Pre-allocated buffers for zero-alloc router computation
    pub router_scored: Vec<(usize, f32)>, // [num_slots]
    pub router_r_t: Vec<f32>,             // [num_slots]
    /// Pre-allocated score buffer for raven_readout_into [num_slots]
    pub readout_scores: Vec<f32>,
    /// Pre-allocated output buffer for raven_readout_into [kv_dim]
    pub readout_output: Vec<f32>,
    // ── usize fields (8-byte aligned, no padding after Vecs) ──
    /// Number of memory slots
    pub num_slots: usize,
    /// Dimension of each KV entry (= kv_dim = n_kv_head × head_dim)
    pub kv_dim: usize,
    /// Top-K slots to update per token
    pub top_k: usize,
    // ── f32 field last (4-byte aligned, no trailing padding on 64-bit) ──
    /// Forget rate for gated update (negative = slower decay)
    pub forget_rate: f32,
}

impl RavenKVCache {
    pub fn new(config: &Config, num_slots: usize, top_k: usize) -> Self {
        let kvd = types::kv_dim(config);
        Self {
            num_slots,
            kv_dim: kvd,
            top_k,
            keys: vec![0.0; num_slots * kvd],
            values: vec![0.0; num_slots * kvd],
            router_scored: vec![(0usize, 0.0f32); num_slots],
            router_r_t: vec![0.0f32; num_slots],
            readout_scores: vec![0.0; num_slots],
            readout_output: vec![0.0; kvd],
            forget_rate: -1.0,
        }
    }

    pub fn reset(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        // Use fill() instead of clear() to preserve pre-allocated capacity.
        // clear() drops len to 0, forcing reallocation on next use via resize.
        self.router_scored.fill((0, 0.0));
        self.router_r_t.fill(0.0);
        self.readout_scores.fill(0.0);
        self.readout_output.fill(0.0);
    }

    /// Export the current routing vector `r_t` (post-router, pre-update).
    /// Returns the normalized Top-K routing weights for all slots.
    /// Used by Phase 3 routed speculation to feed slot selection into anyrag.
    #[inline]
    pub fn r_t(&self) -> &[f32] {
        &self.router_r_t
    }
}
