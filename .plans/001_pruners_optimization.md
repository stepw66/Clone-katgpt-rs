# 001 Pruners Optimization Plan

Implementation of `.issues/001_pruners_optimization.md` findings.

## Tasks

### CRITICAL
- [x] **C-1**: BomberState `cells: Vec<Vec<Cell>>` → flat array `[Cell; 169]`
- [ ] **C-2**: `available_actions()` returns `Vec<BomberAction>` → use `available_actions_into()` with pre-allocated buffer in MCTS
- [ ] **C-3**: GoHeuristic `influence()` — per-cell BFS → multi-source BFS

### HIGH (Correctness)
- [ ] **H-7**: `is_circuit_neuron()` broken binary search → `HashSet<(usize, usize)>`

### HIGH (Performance)
- [ ] **H-1**: `Arc<BFCP>` instead of deep clone in cache pipeline
- [ ] **H-2**: `soft_route_relevance()` allocates 3 Vecs → pre-allocated scratch buffers
- [ ] **H-3**: `CurvatureInfluence arm_bandit_score()` allocates Vec per arm → cache concentration
- [ ] **H-5**: `SensitivityCache` uses `Arc<RwLock<HashMap>>` → papaya
- [ ] **H-8**: `softmax()` + `kl_divergence()` allocate 3 Vecs → pre-allocate in struct
- [ ] **H-9**: `review_metrics` cascading atomic loads → snapshot once
- [ ] **H-11**: `selected_arms.remove(0)` O(n) → VecDeque
- [ ] **H-12**: `bfcp_preimage` sigmoid waste → simple comparison
- [ ] **H-13**: `roaring_membership` len() iterates → cache cardinality
- [ ] **H-14**: `phrase_trie` O(n²) dedup → HashSet
- [x] **H-16**: `lsh_cache` column-major → row-major iteration
- [x] **H-17**: MCTSNode children/unexpanded Vecs → SmallVec (pre-allocated with `Vec::with_capacity(8)`)
- [ ] **H-19**: Bomber `softmax()` in players.rs → sigmoid
- [x] **H-24**: Bomber blast zone → pre-compute grid
- [x] **H-25**: Bomber `escape_distance()` HashSet → bitset

### MEDIUM
- [ ] Remaining MEDIUM items from issue

### LOW
- [ ] Remaining LOW items from issue

## GOAT Gates
| Gate | Change | Metric |
|------|--------|--------|
| `goat_flat_cells` | BomberState flat cells | MCTS nodes/sec |
| `goat_multisource_bfs` | Go multi-source BFS | evaluate() μs |
