//! ToaST split tree types for tree-based tokenization.
//!
//! **Source:** Schmidt et al. (2026). Tokenization with Split Trees. arXiv:2605.22705

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A node in a split tree.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SplitNode {
    /// Byte span [start, end) within the original pretoken.
    pub start: u16,
    pub end: u16,
    /// Index of left child in nodes vec, or None for leaf (single byte).
    pub left: Option<u32>,
    /// Index of right child in nodes vec, or None for leaf (single byte).
    pub right: Option<u32>,
}

/// A full binary split tree for a single pretoken.
/// Nodes stored in array form; root is index 0.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SplitTree {
    /// The original pretoken bytes.
    pub pretoken: Vec<u8>,
    /// All nodes in the tree (preorder). Root = index 0.
    pub nodes: Vec<SplitNode>,
}

/// ToaST tokenizer: vocabulary + pre-built split trees for pretokens.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToastTokenizer {
    /// Token bytes → ID mapping.
    #[serde(with = "super::types::map_serde_bytes")]
    pub vocab_to_id: HashMap<Vec<u8>, usize>,
    /// ID → token bytes mapping.
    pub id_to_vocab: Vec<Vec<u8>>,
    /// Pretoken bytes → SplitTree (pre-built from n-gram counts).
    #[serde(with = "map_serde_trees")]
    pub trees: HashMap<Vec<u8>, SplitTree>,
    /// BOS token ID.
    pub bos_id: usize,
    /// EOS token ID.
    pub eos_id: usize,
    /// PAD token ID.
    pub pad_id: usize,
    /// UNK token ID.
    pub unk_id: usize,
}

impl ToastTokenizer {
    /// Number of tokens in vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.id_to_vocab.len()
    }
}

/// Serde module for `HashMap<Vec<u8>, SplitTree>` — keys as hex strings.
mod map_serde_trees {
    use super::SplitTree;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;

    pub fn serialize<S: Serializer>(
        map: &HashMap<Vec<u8>, SplitTree>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let vec: Vec<(String, &SplitTree)> =
            map.iter().map(|(k, v)| (bytes_to_hex(k), v)).collect();
        vec.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<HashMap<Vec<u8>, SplitTree>, D::Error> {
        let vec: Vec<(String, SplitTree)> = Vec::deserialize(d)?;
        Ok(vec
            .into_iter()
            .filter_map(|(hex, tree)| hex_to_bytes(&hex).map(|bytes| (bytes, tree)))
            .collect())
    }

    fn bytes_to_hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn hex_to_bytes(hex: &str) -> Option<Vec<u8>> {
        if !hex.len().is_multiple_of(2) {
            return None;
        }
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).ok())
            .collect()
    }
}
