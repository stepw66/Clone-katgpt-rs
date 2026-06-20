    use super::*;
    use crate::types::TernaryDir;

    #[test]
    fn test_ternary_dot_matches_scalar() {
        let state = [1.0f32, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let dir = TernaryDir {
            pos_bits: 0b0101, // indices 0, 2 are positive
            neg_bits: 0b1000, // index 3 is negative
            row_scale: 0.5,
        };
        let result = simd_ternary_dot_f32(&state, &dir);
        // +1*1.0 + 0*2.0 + 1*3.0 + (-1)*4.0 = 0.0, scaled by 0.5 = 0.0
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_ternary_dot_all_positive() {
        let state = [1.0f32, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let dir = TernaryDir {
            pos_bits: 0b11,
            neg_bits: 0,
            row_scale: 1.0,
        };
        let result = simd_ternary_dot_f32(&state, &dir);
        assert!((result - 3.0).abs() < 1e-6);
    }
