use core::{fmt, ops::Index};

use alloc::vec::Vec;
use miden_crypto::{hash::rpo::RpoDigest, Felt, WORD_SIZE};
use winter_utils::{ByteReader, ByteWriter, Deserializable, DeserializationError, Serializable};

use crate::{
    errors::ProgramError,
    mast::{MastForest, MastNode, MastNodeId},
    utils::ToElements,
};

use super::Kernel;

// PROGRAM
// ===============================================================================================

#[derive(Clone, Debug)]
pub struct Program {
    mast_forest: MastForest,
}

/// Constructors
impl Program {
    pub fn new(mast_forest: MastForest) -> Result<Self, ProgramError> {
        if mast_forest.entrypoint().is_some() {
            Ok(Self { mast_forest })
        } else {
            Err(ProgramError::NoEntrypoint)
        }
    }
}

/// Public accessors
impl Program {
    /// Returns the underlying [`MastForest`].
    pub fn mast_forest(&self) -> &MastForest {
        &self.mast_forest
    }

    /// Returns the kernel associated with this program.
    pub fn kernel(&self) -> &Kernel {
        self.mast_forest.kernel()
    }

    /// Returns the entrypoint associated with this program.
    pub fn entrypoint(&self) -> MastNodeId {
        self.mast_forest.entrypoint().unwrap()
    }

    /// A convenience method that provides the hash of the entrypoint.
    pub fn hash(&self) -> RpoDigest {
        self.mast_forest.entrypoint_digest().unwrap()
    }

    /// Returns the [`MastNode`] associated with the provided [`MastNodeId`] if valid, or else
    /// `None`.
    ///
    /// This is the faillible version of indexing (e.g. `program[node_id]`).
    #[inline(always)]
    pub fn get_node_by_id(&self, node_id: MastNodeId) -> Option<&MastNode> {
        self.mast_forest.get_node_by_id(node_id)
    }

    /// Returns the [`MastNodeId`] associated with a given digest, if any.
    ///
    /// That is, every [`MastNode`] hashes to some digest. If there exists a [`MastNode`] in the
    /// forest that hashes to this digest, then its id is returned.
    #[inline(always)]
    pub fn get_node_id_by_digest(&self, digest: RpoDigest) -> Option<MastNodeId> {
        self.mast_forest.get_node_id_by_digest(digest)
    }
}

impl Index<MastNodeId> for Program {
    type Output = MastNode;

    fn index(&self, node_id: MastNodeId) -> &Self::Output {
        &self.mast_forest[node_id]
    }
}

impl crate::prettier::PrettyPrint for Program {
    fn render(&self) -> crate::prettier::Document {
        use crate::prettier::*;
        let entrypoint = self[self.entrypoint()].to_pretty_print(&self.mast_forest);

        indent(4, const_text("begin") + nl() + entrypoint.render()) + nl() + const_text("end")
    }
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::prettier::PrettyPrint;
        self.pretty_print(f)
    }
}

impl TryFrom<MastForest> for Program {
    type Error = ProgramError;

    fn try_from(mast_forest: MastForest) -> Result<Self, Self::Error> {
        Self::new(mast_forest)
    }
}

impl From<Program> for MastForest {
    fn from(program: Program) -> Self {
        program.mast_forest
    }
}

// PROGRAM INFO
// ===============================================================================================

/// A program information set consisting of its MAST root and set of kernel procedure roots used
/// for its compilation.
///
/// This will be used as public inputs of the proof so we bind its verification to the kernel and
/// root used to execute the program. This way, we extend the correctness of the proof to the
/// security guarantees provided by the kernel. We also allow the user to easily prove the
/// membership of a given kernel procedure for a given proof, without compromising its
/// zero-knowledge properties.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProgramInfo {
    program_hash: RpoDigest,
    kernel: Kernel,
}

impl ProgramInfo {
    // CONSTRUCTORS
    // --------------------------------------------------------------------------------------------

    /// Creates a new instance of a program info.
    pub const fn new(program_hash: RpoDigest, kernel: Kernel) -> Self {
        Self {
            program_hash,
            kernel,
        }
    }

    // PUBLIC ACCESSORS
    // --------------------------------------------------------------------------------------------

    /// Returns the program hash computed from its code block root.
    pub const fn program_hash(&self) -> &RpoDigest {
        &self.program_hash
    }

    /// Returns the program kernel used during the compilation.
    pub const fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    /// Returns the list of procedures of the kernel used during the compilation.
    pub fn kernel_procedures(&self) -> &[RpoDigest] {
        self.kernel.proc_hashes()
    }
}

impl From<Program> for ProgramInfo {
    fn from(program: Program) -> Self {
        let program_hash = program.hash();
        let kernel = program.kernel().clone();

        Self {
            program_hash,
            kernel,
        }
    }
}

// SERIALIZATION
// ------------------------------------------------------------------------------------------------

impl Serializable for ProgramInfo {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        self.program_hash.write_into(target);
        self.kernel.write_into(target);
    }
}

impl Deserializable for ProgramInfo {
    fn read_from<R: ByteReader>(source: &mut R) -> Result<Self, DeserializationError> {
        let program_hash = source.read()?;
        let kernel = source.read()?;
        Ok(Self {
            program_hash,
            kernel,
        })
    }
}

// TO ELEMENTS
// ------------------------------------------------------------------------------------------------

impl ToElements for ProgramInfo {
    fn to_elements(&self) -> Vec<Felt> {
        let num_kernel_proc_elements = self.kernel.proc_hashes().len() * WORD_SIZE;
        let mut result = Vec::with_capacity(WORD_SIZE + num_kernel_proc_elements);

        // append program hash elements
        result.extend_from_slice(self.program_hash.as_elements());

        // append kernel procedure hash elements
        for proc_hash in self.kernel.proc_hashes() {
            result.extend_from_slice(proc_hash.as_elements());
        }
        result
    }
}