mod bpe;
mod types;

pub use bpe::{BpeTokenizerImpl, BpeTrainer};
pub use types::{BpeTokenizer, MergeRule};

#[cfg(feature = "toast_tokenizer")]
mod toast_builder;
#[cfg(feature = "toast_tokenizer")]
mod toast_inference;
#[cfg(feature = "toast_tokenizer")]
mod toast_types;

#[cfg(feature = "toast_tokenizer")]
pub use toast_builder::SplitTreeBuilder;
#[cfg(feature = "toast_tokenizer")]
pub use toast_inference::ToastTokenizerImpl;
#[cfg(feature = "toast_tokenizer")]
pub use toast_types::{SplitNode, SplitTree, ToastTokenizer};
