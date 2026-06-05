//! AND-OR tree module for hierarchical goal decomposition.
//!
//! Provides [`AndOrNode`], a generic AND-OR tree structure inspired by
//! LEAP's AND-OR DAG proof search (arXiv 2606.03303).
//!
//! # Usage
//!
//! ```ignore
//! use katgpt_core::and_or::AndOrNode;
//!
//! // OR node: try alternatives
//! let mut root = AndOrNode::or("prove_theorem");
//! root.push_child(AndOrNode::unsolved_leaf("tactic_1"));
//! root.push_child(AndOrNode::solved_leaf("tactic_2", vec![0, 1, 2]));
//!
//! assert!(root.is_solved()); // OR: any child solved
//! ```

mod types;

pub use types::AndOrNode;
