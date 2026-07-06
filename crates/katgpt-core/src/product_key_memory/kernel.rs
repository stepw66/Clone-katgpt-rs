//! Retrieval kernel for [`ProductKeyMemory`] — **Phase 2 stub**.
//!
//! Plan 408 Phase 1 lands ONLY the type definitions in [`crate::types`]. The
//! actual `query_into` retrieval kernel (Plan 408 Phase 2, tasks T2.1–T2.7)
//! ships in this file in a follow-up commit. The stub here exists so the
//! module structure is in place and Phase 1 type checks + unit tests can run
//! without waiting on the kernel.
//!
//! # Phase 2 contract (what lands next)
//!
//! The kernel implements the O(√N) factored retrieval:
//!
//! ```text
//! 1. Split q into q1 = q[..D_K/2], q2 = q[D_K/2..].
//! 2. Heapselect top-k from codebook 1: s1[i] = score(q1, keys_1[i]).  O(√N)
//! 3. Heapselect top-k from codebook 2: s2[j] = score(q2, keys_2[j]).  O(√N)
//! 4. Cartesian product: for (i,j) in I1 × I2, score_{i,j} = s1[i] + s2[j].  O(K²)
//! 5. Top-k of the K² candidates, map (i,j) -> flat_index = i*SQRT_N + j.  O(K² log K)
//! 6. Normalize weights (sigmoid-gate OR softmax-over-k² per Plan 408 T2.1 step 6).
//! 7. Write (flat_index, weight) into out[..k].
//! ```
//!
//! All scratch buffers are caller-allocated (`&mut [f32]` for the two √N score
//! arrays) — zero allocation inside the hot path (Plan 408 G4 gate).
//!
//! # Why this stub exists separately
//!
//! Phase 1's [`crate::types`] is independently valuable (the const-generic
//! table layout, the `ScoreFn` enum, the fixed-size `PkQuery<K>` result type)
//! and lets callers / sibling repos start typing against the API before the
//! kernel lands. Phase 2 then adds `query_into` against this stable surface.
