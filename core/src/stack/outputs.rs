use alloc::vec::Vec;

use miden_crypto::{Word, ZERO};

use super::{
    get_stack_values_num, ByteWriter, Felt, OutputError, Serializable, ToElements, MIN_STACK_DEPTH,
};
use crate::utils::{range, ByteReader, Deserializable, DeserializationError};

// STACK OUTPUTS
// ================================================================================================

/// Output container for Miden VM programs.
///
/// Miden program outputs contain the full state of the stack at the end of execution.
///
/// `stack` is expected to be ordered as if the elements were popped off the stack one by one.
/// Thus, the value at the top of the stack is expected to be in the first position, and the order
/// of the rest of the output elements will also match the order on the stack.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StackOutputs {
    elements: [Felt; MIN_STACK_DEPTH],
}

impl StackOutputs {
    // CONSTRUCTORS
    // --------------------------------------------------------------------------------------------

    /// Constructs a new [StackOutputs] struct from the provided stack elements.
    ///
    /// # Errors
    ///  Returns an error if the number of stack elements is greater than `MIN_STACK_DEPTH` (16).
    pub fn new(mut stack: Vec<Felt>) -> Result<Self, OutputError> {
        // validate stack length
        if stack.len() > MIN_STACK_DEPTH {
            return Err(OutputError::OutputSizeTooBig(stack.len()));
        }
        stack.resize(MIN_STACK_DEPTH, ZERO);

        Ok(Self { elements: stack.try_into().unwrap() })
    }

    /// Attempts to create [StackOutputs] struct from the provided stack elements represented as
    /// vector of `u64` values.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Any of the provided stack elements are invalid field elements.
    pub fn try_from_ints(stack: Vec<u64>) -> Result<Self, OutputError> {
        // Validate stack elements
        let stack = stack
            .iter()
            .map(|v| Felt::try_from(*v))
            .collect::<Result<Vec<Felt>, _>>()
            .map_err(OutputError::InvalidStackElement)?;

        Self::new(stack)
    }

    // PUBLIC ACCESSORS
    // --------------------------------------------------------------------------------------------

    /// Returns the element located at the specified position on the stack or `None` if out of
    /// bounds.
    pub fn get_stack_item(&self, idx: usize) -> Option<Felt> {
        self.elements.get(idx).cloned()
    }

    /// Returns the word located starting at the specified Felt position on the stack or `None` if
    /// out of bounds. For example, passing in `0` returns the word at the top of the stack, and
    /// passing in `4` returns the word starting at element index `4`.
    pub fn get_stack_word(&self, idx: usize) -> Option<Word> {
        let word_elements: Word = {
            let word_elements: Vec<Felt> = range(idx, 4)
                .map(|idx| self.get_stack_item(idx))
                // Elements need to be reversed, since a word `[a, b, c, d]` will be stored on the
                // stack as `[d, c, b, a]`
                .rev()
                .collect::<Option<_>>()?;

            word_elements.try_into().expect("a Word contains 4 elements")
        };

        Some(word_elements)
    }

    /// Returns the stack outputs, which is state of the stack at the end of execution.
    pub fn elements(&self) -> &[Felt] {
        &self.elements
    }

    /// Returns the number of requested stack outputs or returns the full stack if fewer than the
    /// requested number of stack values exist.
    pub fn stack_truncated(&self, num_outputs: usize) -> &[Felt] {
        let len = self.elements.len().min(num_outputs);
        &self.elements[..len]
    }

    // PUBLIC MUTATORS
    // --------------------------------------------------------------------------------------------

    /// Returns mutable access to the stack outputs, to be used for testing or running examples.
    /// TODO: this should be marked with #[cfg(test)] attribute, but that currently won't work with
    /// the integration test handler util.
    pub fn stack_mut(&mut self) -> &mut [Felt] {
        &mut self.elements
    }
}

// HELPER FUNCTIONS
// ================================================================================================

impl ToElements<Felt> for StackOutputs {
    fn to_elements(&self) -> Vec<Felt> {
        self.elements.to_vec()
    }
}

// SERIALIZATION
// ================================================================================================

impl Serializable for StackOutputs {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        target.write_u8(get_stack_values_num(self.elements()));
        target.write_many(self.elements);
    }
}

impl Deserializable for StackOutputs {
    fn read_from<R: ByteReader>(source: &mut R) -> Result<Self, DeserializationError> {
        let elements_num = source.read_u8()?;
        let mut elements = source.read_many::<Felt>(elements_num.into())?;

        elements.resize(MIN_STACK_DEPTH, ZERO);

        Ok(Self { elements: elements.try_into().unwrap() })
    }
}
