//! `extract_top_k_peaks` — extracts the top-K values from a logit slice
//! in descending order.

/// Returns the top `k` values from `logits`, sorted descending.
/// If `k` exceeds the slice length, returns all values sorted descending.
pub fn extract_top_k_peaks(logits: &[f32], k: usize) -> Vec<f32> {
    let k = k.min(logits.len());
    if k == 0 {
        return Vec::new();
    }
    let mut values: Vec<f32> = logits.to_vec();
    values.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    values.truncate(k);
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_top_k() {
        let logits = vec![0.1, 0.9, 0.3, 0.7, 0.5];
        let peaks = extract_top_k_peaks(&logits, 3);
        assert_eq!(peaks, vec![0.9, 0.7, 0.5]);
    }

    #[test]
    fn k_exceeds_length() {
        let logits = vec![0.1, 0.3];
        let peaks = extract_top_k_peaks(&logits, 5);
        assert_eq!(peaks, vec![0.3, 0.1]);
    }

    #[test]
    fn empty_input() {
        let peaks = extract_top_k_peaks(&[], 3);
        assert!(peaks.is_empty());
    }
}
