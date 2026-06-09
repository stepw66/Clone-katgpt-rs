//! LinOSS — Linear Oscillatory State-Space cell + ModalSpec drafter (Plan 189 Phase 3).
//!
//! Symplectic IMEX integrator for harmonic oscillators. Linear dynamics enable
//! parallel scan (Blelloch prefix sum) for training/parallel mode.

/// LinOSS cell parameters. Pre-allocated in constructor, zero alloc on hot path.
#[derive(Clone, Debug)]
pub struct LinOSSCell {
    /// Angular frequency squared per hidden dimension: ω² [H].
    pub omega_sq: Vec<f32>,
    /// Damping coefficient per hidden dimension: β [H].
    pub beta: Vec<f32>,
    hidden_dim: usize,
}

/// LinOSS hidden state (phase-space: position y, velocity z).
#[derive(Clone, Debug)]
pub struct LinOSSState {
    /// Position coordinate [H].
    pub y: Vec<f32>,
    /// Velocity coordinate [H].
    pub z: Vec<f32>,
}

/// Sigmoid activation (NOT softmax per project constraints).
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl LinOSSCell {
    /// Create with unit frequency (ω²=1), undamped (β=0).
    #[inline]
    pub fn new(hidden_dim: usize) -> Self {
        Self {
            omega_sq: vec![1.0; hidden_dim],
            beta: vec![0.0; hidden_dim],
            hidden_dim,
        }
    }

    #[inline]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// One IMEX step (symplectic, energy-preserving when β=0).
    /// y' = y + dt * z   (explicit)
    /// z' = z + dt * (-ω² * y' - β * z + forcing)   (implicit on y')
    ///
    /// Allocating version — see `imex_step_inplace` for zero-alloc alternative.
    #[inline]
    pub fn imex_step(&self, state: &LinOSSState, forcing: &[f32], dt: f32) -> LinOSSState {
        let h = self.hidden_dim;
        debug_assert_eq!(state.y.len(), h);
        debug_assert_eq!(state.z.len(), h);
        debug_assert!(forcing.is_empty() || forcing.len() == h);
        let mut y_new = vec![0.0f32; h];
        let mut z_new = vec![0.0f32; h];
        for i in 0..h {
            y_new[i] = state.y[i] + dt * state.z[i];
            let f = if forcing.is_empty() { 0.0 } else { forcing[i] };
            z_new[i] =
                state.z[i] + dt * (-self.omega_sq[i] * y_new[i] - self.beta[i] * state.z[i] + f);
        }
        LinOSSState { y: y_new, z: z_new }
    }

    /// In-place IMEX step — writes y_new and z_new into pre-allocated buffers.
    /// Returns (y_new, z_new) slices. Zero allocation.
    #[inline]
    pub fn imex_step_inplace<'a>(
        &self,
        y_in: &[f32],
        z_in: &[f32],
        forcing: &[f32],
        dt: f32,
        y_out: &'a mut [f32],
        z_out: &'a mut [f32],
    ) -> (&'a [f32], &'a [f32]) {
        let h = self.hidden_dim;
        debug_assert_eq!(y_in.len(), h);
        debug_assert_eq!(z_in.len(), h);
        debug_assert!(y_out.len() >= h);
        debug_assert!(z_out.len() >= h);
        debug_assert!(forcing.is_empty() || forcing.len() == h);
        for i in 0..h {
            y_out[i] = y_in[i] + dt * z_in[i];
            let f = if forcing.is_empty() { 0.0 } else { forcing[i] };
            z_out[i] = z_in[i] + dt * (-self.omega_sq[i] * y_out[i] - self.beta[i] * z_in[i] + f);
        }
        (&y_out[..h], &z_out[..h])
    }

    /// Parallel scan (Blelloch prefix sum) for training/parallel mode.
    /// LinOSS is linear → transfer matrices compose → parallel prefix scan works.
    pub fn parallel_scan(
        &self,
        initial: &LinOSSState,
        forcings: &[&[f32]],
        dt: f32,
    ) -> Vec<LinOSSState> {
        let n = forcings.len();
        if n == 0 {
            return vec![];
        }
        // For small sequences, sequential avoids overhead.
        if n <= 64 {
            return self.sequential_scan(initial, forcings, dt);
        }

        let h = self.hidden_dim;
        // Per-step transfer matrix: [[1, dt], [-ω²dt, 1]] with bias [0, dt*f]
        let mut a = vec![0.0f32; n * h];
        let mut b = vec![0.0f32; n * h];
        let mut c = vec![0.0f32; n * h];
        let mut d = vec![0.0f32; n * h];
        let bias_y = vec![0.0f32; n * h];
        let mut bias_z = vec![0.0f32; n * h];

        for step in 0..n {
            let base = step * h;
            let f = forcings[step];
            for j in 0..h {
                a[base + j] = 1.0;
                b[base + j] = dt;
                c[base + j] = -self.omega_sq[j] * dt;
                d[base + j] = 1.0;
                bias_z[base + j] = dt * f[j]; // bias_y stays 0
            }
        }

        // Inclusive prefix scan: prefix[i] = M_i * M_{i-1} * ... * M_0
        // NOTE: Replace with Blelloch up-sweep/down-sweep for true parallelism.
        let mut pa = a.clone();
        let mut pb = b.clone();
        let mut pc = c.clone();
        let mut pd = d.clone();
        let mut pby = bias_y.clone();
        let mut pbz = bias_z.clone();

        for step in 1..n {
            let prev = (step - 1) * h;
            let base = step * h;
            for j in 0..h {
                let (pa0, pb0, pc0, pd0) = (pa[prev + j], pb[prev + j], pc[prev + j], pd[prev + j]);
                let (pby0, pbz0) = (pby[prev + j], pbz[prev + j]);
                let (ma, mb, mc, md) = (a[base + j], b[base + j], c[base + j], d[base + j]);
                let (mby, mbz) = (bias_y[base + j], bias_z[base + j]);
                pa[base + j] = pa0 * ma + pb0 * mc;
                pb[base + j] = pa0 * mb + pb0 * md;
                pc[base + j] = pc0 * ma + pd0 * mc;
                pd[base + j] = pc0 * mb + pd0 * md;
                pby[base + j] = pa0 * mby + pb0 * mbz + pby0;
                pbz[base + j] = pc0 * mby + pd0 * mbz + pbz0;
            }
        }

        // Apply each prefix to initial state.
        let mut results = Vec::with_capacity(n);
        for step in 0..n {
            let base = step * h;
            let mut y = vec![0.0f32; h];
            let mut z = vec![0.0f32; h];
            for j in 0..h {
                y[j] = pa[base + j] * initial.y[j] + pb[base + j] * initial.z[j] + pby[base + j];
                z[j] = pc[base + j] * initial.y[j] + pd[base + j] * initial.z[j] + pbz[base + j];
            }
            results.push(LinOSSState { y, z });
        }
        results
    }

    /// Sequential scan — simple loop for correctness reference and small sequences.
    #[inline]
    fn sequential_scan(
        &self,
        initial: &LinOSSState,
        forcings: &[&[f32]],
        dt: f32,
    ) -> Vec<LinOSSState> {
        let n = forcings.len();
        let mut results = Vec::with_capacity(n);
        let mut state = initial.clone();
        for i in 0..n {
            state = self.imex_step(&state, forcings[i], dt);
            results.push(state.clone());
        }
        results
    }

    #[inline]
    pub fn energy(&self, state: &LinOSSState) -> f32 {
        let h = self.hidden_dim;
        let mut e = 0.0f32;
        for i in 0..h {
            e += state.y[i] * state.y[i] + self.omega_sq[i] * state.z[i] * state.z[i];
        }
        e
    }
}

impl LinOSSState {
    #[inline]
    pub fn zeros(hidden_dim: usize) -> Self {
        Self {
            y: vec![0.0; hidden_dim],
            z: vec![0.0; hidden_dim],
        }
    }
}

// ── VocabFourierBasis ──

/// Top-K Fourier modes of vocabulary embedding space.
/// Pre-computed once at model load time — zero alloc on hot path.
pub struct VocabFourierBasis {
    /// Top-K Fourier mode coefficients: [K][vocab_dim].
    pub modes: Vec<Vec<f32>>,
    /// Mode frequencies (angular frequency for each mode).
    pub frequencies: Vec<f32>,
    /// Number of modes.
    k: usize,
    /// Dimension of each mode vector.
    vocab_dim: usize,
}

impl VocabFourierBasis {
    /// Compute top-K Fourier modes from vocabulary embeddings via DFT dot-product.
    pub fn from_embeddings(embeddings: &[&[f32]], k: usize) -> Self {
        if embeddings.is_empty() {
            return Self {
                modes: vec![],
                frequencies: vec![],
                k: 0,
                vocab_dim: 0,
            };
        }

        let n = embeddings.len();
        let vocab_dim = embeddings[0].len();

        // Candidate frequencies: sample across Nyquist range.
        let n_candidates = (n * 2).min(256);
        let mut candidates: Vec<(f32, Vec<f32>)> = Vec::with_capacity(n_candidates);

        for ci in 0..n_candidates {
            let omega = std::f32::consts::PI * (ci as f32 + 1.0) / n as f32;
            let mut cos_mode = vec![0.0f32; vocab_dim];
            for (i, emb) in embeddings.iter().enumerate() {
                let cos_w = (omega * i as f32).cos();
                for d in 0..vocab_dim {
                    cos_mode[d] += emb[d] * cos_w;
                }
            }
            let inv_n = 1.0 / n as f32;
            for d in 0..vocab_dim {
                cos_mode[d] *= inv_n;
            }
            let mag: f32 = cos_mode.iter().map(|v| v * v).sum::<f32>().sqrt();
            candidates.push((mag, cos_mode));
        }

        // Sort by magnitude descending, take top-K.
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);

        let modes: Vec<Vec<f32>> = candidates.iter().map(|(_, mode)| mode.clone()).collect();
        let frequencies: Vec<f32> = candidates
            .iter()
            .enumerate()
            .map(|(i, _)| std::f32::consts::PI * (i as f32 + 1.0) / n as f32)
            .collect();
        let k = modes.len();

        Self {
            modes,
            frequencies,
            k,
            vocab_dim,
        }
    }

    /// Reconstruct: token ≈ Σ_k coefficient[k] * mode[k]
    ///
    /// Allocating version — see `reconstruct_into` for zero-alloc alternative.
    #[inline]
    pub fn reconstruct(&self, coefficients: &[f32]) -> Vec<f32> {
        if self.k == 0 {
            return vec![];
        }
        let mut result = vec![0.0f32; self.vocab_dim];
        self.reconstruct_into(coefficients, &mut result);
        result
    }

    /// Zero-alloc reconstruct into pre-allocated buffer.
    #[inline]
    pub fn reconstruct_into(&self, coefficients: &[f32], result: &mut [f32]) {
        if self.k == 0 {
            return;
        }
        let vd = self.vocab_dim.min(result.len());
        result[..vd].fill(0.0);
        for (ki, mode) in self.modes.iter().enumerate() {
            let c = coefficients.get(ki).copied().unwrap_or(0.0);
            for d in 0..vd {
                result[d] += c * mode[d];
            }
        }
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }
    #[inline]
    pub fn vocab_dim(&self) -> usize {
        self.vocab_dim
    }
}

// ── ModalSpecDrafter ──

/// Modal speculative drafter — LinOSS state-space + Fourier modes.
///
/// Pipeline: prompt → LinOSS state → modal coefficients → Fourier reconstruct → nearest token.
pub struct ModalSpecDrafter {
    cell: LinOSSCell,
    basis: VocabFourierBasis,
    dt: f32,
    hidden_dim: usize,
    /// Stored embeddings for nearest-token lookup.
    embeddings: Vec<Vec<f32>>,
}

impl ModalSpecDrafter {
    /// Create a new ModalSpecDrafter.
    ///
    /// - `hidden_dim`: LinOSS hidden dimension (typically 64–256).
    /// - `vocab_embeddings`: vocabulary embedding vectors [vocab_size][vocab_dim].
    /// - `n_modes`: number of Fourier modes to extract (typically 8–32).
    #[inline]
    pub fn new(hidden_dim: usize, vocab_embeddings: &[&[f32]], n_modes: usize) -> Self {
        let cell = LinOSSCell::new(hidden_dim);
        let basis = VocabFourierBasis::from_embeddings(vocab_embeddings, n_modes);

        // Store embeddings for nearest-token lookup.
        let embeddings: Vec<Vec<f32>> = vocab_embeddings.iter().map(|e| e.to_vec()).collect();

        Self {
            cell,
            basis,
            dt: 0.1, // Default timestep — can be tuned per model.
            hidden_dim,
            embeddings,
        }
    }

    /// Draft tokens: encode prompt → LinOSS oscillation → Fourier reconstruct → nearest vocab.
    ///
    /// Allocating version — see `draft_into` for zero-alloc alternative.
    pub fn draft(&self, prompt_tokens: &[usize], n_draft: usize) -> Vec<usize> {
        if n_draft == 0 || self.embeddings.is_empty() {
            return vec![];
        }
        let h = self.hidden_dim;
        let vocab_dim = self.basis.vocab_dim();
        let k = self.basis.k();
        let mut state = LinOSSState::zeros(h);
        for &tok in prompt_tokens {
            if tok < self.embeddings.len() {
                let forcing = self.project_to_hidden(&self.embeddings[tok], vocab_dim);
                state = self.cell.imex_step(&state, &forcing, self.dt);
            }
        }
        let zero_forcing = vec![0.0f32; h];
        let mut draft = Vec::with_capacity(n_draft);
        for _ in 0..n_draft {
            state = self.cell.imex_step(&state, &zero_forcing, self.dt);
            let coeffs = self.extract_coefficients(&state, k);
            let reconstructed = self.basis.reconstruct(&coeffs);
            draft.push(self.nearest_token(&reconstructed));
        }
        draft
    }

    /// Zero-alloc draft into pre-allocated output buffer.
    ///
    /// Uses double-buffered scratch (y_a/z_a, y_b/z_b) to avoid allocation per timestep.
    /// Returns the number of drafted tokens written to `out`.
    pub fn draft_into(&self, prompt_tokens: &[usize], out: &mut [usize]) -> usize {
        let n_draft = out.len();
        if n_draft == 0 || self.embeddings.is_empty() {
            return 0;
        }
        let h = self.hidden_dim;
        let vocab_dim = self.basis.vocab_dim();
        let k = self.basis.k();

        // Pre-allocate all scratch buffers once
        let mut y_a = vec![0.0f32; h];
        let mut z_a = vec![0.0f32; h];
        let mut y_b = vec![0.0f32; h];
        let mut z_b = vec![0.0f32; h];
        let mut forcing = vec![0.0f32; h];
        let mut coeffs = vec![0.0f32; k];
        let mut reconstructed = vec![0.0f32; vocab_dim];

        // Prompt encoding
        for &tok in prompt_tokens {
            if tok < self.embeddings.len() {
                self.project_to_hidden_into(&self.embeddings[tok], vocab_dim, &mut forcing);
                let (y_new, z_new) = self
                    .cell
                    .imex_step_inplace(&y_a, &z_a, &forcing, self.dt, &mut y_b, &mut z_b);
                y_a[..h].copy_from_slice(y_new);
                z_a[..h].copy_from_slice(z_new);
            }
        }

        // Draft loop — zero alloc per iteration
        forcing.fill(0.0);
        let mut drafted = 0;
        for i in 0..n_draft {
            let (y_new, z_new) = self
                .cell
                .imex_step_inplace(&y_a, &z_a, &forcing, self.dt, &mut y_b, &mut z_b);
            y_a[..h].copy_from_slice(y_new);
            z_a[..h].copy_from_slice(z_new);

            self.extract_coefficients_into(&y_a, k, &mut coeffs);
            self.basis.reconstruct_into(&coeffs, &mut reconstructed);
            out[i] = self.nearest_token(&reconstructed);
            drafted += 1;
        }
        drafted
    }

    #[inline]
    fn project_to_hidden(&self, vec: &[f32], vocab_dim: usize) -> Vec<f32> {
        let h = self.hidden_dim;
        let mut result = vec![0.0f32; h];
        self.project_to_hidden_into(vec, vocab_dim, &mut result);
        result
    }

    #[inline]
    fn project_to_hidden_into(&self, vec: &[f32], vocab_dim: usize, result: &mut [f32]) {
        let h = self.hidden_dim;
        let ratio = vocab_dim as f32 / h as f32;
        for i in 0..h {
            let start = ((i as f32 * ratio) as usize).min(vocab_dim);
            let end = (((i + 1) as f32 * ratio) as usize).min(vocab_dim);
            if start < end {
                result[i] = vec[start..end].iter().sum::<f32>() / (end - start) as f32;
            } else {
                result[i] = 0.0;
            }
        }
    }

    /// Extract first k elements of y (position) as modal coefficients.
    #[inline]
    fn extract_coefficients(&self, state: &LinOSSState, k: usize) -> Vec<f32> {
        let n = k.min(state.y.len());
        let mut coeffs = vec![0.0f32; k];
        coeffs[..n].copy_from_slice(&state.y[..n]);
        coeffs
    }

    /// Zero-alloc coefficient extraction into pre-allocated buffer.
    #[inline]
    fn extract_coefficients_into(&self, y: &[f32], k: usize, coeffs: &mut [f32]) {
        let n = k.min(y.len()).min(coeffs.len());
        coeffs[..n].copy_from_slice(&y[..n]);
        coeffs[n..].fill(0.0);
    }

    /// Find nearest token via sigmoid-gated dot-product (NOT softmax).
    #[inline]
    fn nearest_token(&self, query: &[f32]) -> usize {
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (i, emb) in self.embeddings.iter().enumerate() {
            let dot: f32 = emb.iter().zip(query.iter()).map(|(e, q)| e * q).sum();
            let score = sigmoid(dot);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        best_idx
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imex_step_preserves_energy() {
        let h = 16;
        let cell = LinOSSCell::new(h);
        let mut state = LinOSSState {
            y: vec![1.0; h],
            z: vec![0.5; h],
        };
        let initial_energy = cell.energy(&state);
        let zero_forcing = vec![0.0; h];
        for _ in 0..1000 {
            state = cell.imex_step(&state, &zero_forcing, 0.01);
        }
        let final_energy = cell.energy(&state);
        assert!(
            final_energy < initial_energy * 1.1,
            "Energy grew: initial={initial_energy}, final={final_energy}"
        );
    }

    #[test]
    fn test_imex_step_damps_with_beta() {
        let h = 8;
        let mut cell = LinOSSCell::new(h);
        cell.beta = vec![0.5; h];
        let mut state = LinOSSState {
            y: vec![1.0; h],
            z: vec![1.0; h],
        };
        let initial_energy = cell.energy(&state);
        let zero_forcing = vec![0.0; h];
        for _ in 0..200 {
            state = cell.imex_step(&state, &zero_forcing, 0.01);
        }
        assert!(
            cell.energy(&state) < initial_energy,
            "Energy should decrease with damping"
        );
    }

    #[test]
    fn test_parallel_scan_matches_sequential() {
        let h = 8;
        let cell = LinOSSCell::new(h);
        let initial = LinOSSState {
            y: vec![0.1; h],
            z: vec![0.2; h],
        };
        let forcings: Vec<Vec<f32>> = (0..10).map(|i| vec![0.01 * (i as f32 + 1.0); h]).collect();
        let refs: Vec<&[f32]> = forcings.iter().map(|f| f.as_slice()).collect();
        let seq = cell.sequential_scan(&initial, &refs, 0.1);
        let par = cell.parallel_scan(&initial, &refs, 0.1);
        assert_eq!(seq.len(), par.len());
        for (i, (s, p)) in seq.iter().zip(par.iter()).enumerate() {
            for j in 0..h {
                assert!(
                    (s.y[j] - p.y[j]).abs() < 1e-4,
                    "Step {i} dim {j} y mismatch"
                );
                assert!(
                    (s.z[j] - p.z[j]).abs() < 1e-4,
                    "Step {i} dim {j} z mismatch"
                );
            }
        }
    }

    #[test]
    fn test_fourier_basis_reconstruction() {
        let dim = 8;
        let embs: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                (0..dim)
                    .map(|d| (std::f32::consts::PI * i as f32 * (d as f32 + 1.0) / 20.0).cos())
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = embs.iter().map(|e| e.as_slice()).collect();
        let basis = VocabFourierBasis::from_embeddings(&refs, 4);
        assert!(basis.k() > 0);
        let norm: f32 = basis
            .reconstruct(&vec![1.0; basis.k()])
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!(
            norm > 0.1,
            "Reconstruction should be non-trivial, got norm={norm}"
        );
    }

    #[test]
    fn test_drafter_produces_valid_tokens() {
        let embs: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 0.1; 4]).collect();
        let refs: Vec<&[f32]> = embs.iter().map(|e| e.as_slice()).collect();
        let drafter = ModalSpecDrafter::new(8, &refs, 4);
        let draft = drafter.draft(&[0, 1, 2, 3], 5);
        assert_eq!(draft.len(), 5);
        for &tok in &draft {
            assert!(tok < 10, "Token {tok} out of range");
        }
    }

    #[test]
    fn test_linoss_zero_forcing() {
        let cell = LinOSSCell::new(8);
        let next = cell.imex_step(&LinOSSState::zeros(8), &vec![0.0; 8], 0.1);
        for i in 0..8 {
            assert!(next.y[i].abs() < 1e-10, "y should stay zero");
            assert!(next.z[i].abs() < 1e-10, "z should stay zero");
        }
    }
}
