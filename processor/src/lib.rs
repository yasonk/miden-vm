#![no_std]

#[macro_use]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

use alloc::vec::Vec;
use core::cell::RefCell;

use miden_air::trace::{
    CHIPLETS_WIDTH, DECODER_TRACE_WIDTH, MIN_TRACE_LEN, RANGE_CHECK_TRACE_WIDTH, STACK_TRACE_WIDTH,
    SYS_TRACE_WIDTH,
};
pub use miden_air::{ExecutionOptions, ExecutionOptionsError, RowIndex};
pub use vm_core::{
    chiplets::hasher::Digest,
    crypto::merkle::SMT_DEPTH,
    errors::InputError,
    mast::{MastForest, MastNode, MastNodeId},
    utils::DeserializationError,
    AdviceInjector, AssemblyOp, Felt, Kernel, Operation, Program, ProgramInfo, QuadExtension,
    StackInputs, StackOutputs, Word, EMPTY_WORD, ONE, ZERO,
};
use vm_core::{
    mast::{BasicBlockNode, CallNode, JoinNode, LoopNode, OpBatch, SplitNode, OP_GROUP_SIZE},
    Decorator, DecoratorIterator, FieldElement, StackTopState,
};
pub use winter_prover::matrix::ColMatrix;

mod operations;

mod system;
use system::System;
pub use system::{ContextId, FMP_MIN, SYSCALL_FMP_MIN};

mod decoder;
use decoder::Decoder;

mod stack;
use stack::Stack;

mod range;
use range::RangeChecker;

mod host;
pub use host::{
    advice::{
        AdviceExtractor, AdviceInputs, AdviceMap, AdviceProvider, AdviceSource, MemAdviceProvider,
        RecAdviceProvider,
    },
    DefaultHost, Host, HostResponse, MastForestStore, MemMastForestStore,
};

mod chiplets;
use chiplets::Chiplets;

mod trace;
use trace::TraceFragment;
pub use trace::{ChipletsLengths, ExecutionTrace, TraceLenSummary, NUM_RAND_ROWS};

mod errors;
pub use errors::{ExecutionError, Ext2InttError};

pub mod utils;

mod debug;
pub use debug::{AsmOpInfo, VmState, VmStateIterator};

// RE-EXPORTS
// ================================================================================================

pub mod math {
    pub use vm_core::{Felt, FieldElement, StarkField};
    pub use winter_prover::math::fft;
}

pub mod crypto {
    pub use vm_core::crypto::{
        hash::{
            Blake3_192, Blake3_256, ElementHasher, Hasher, Rpo256, RpoDigest, Rpx256, RpxDigest,
        },
        merkle::{
            MerkleError, MerklePath, MerkleStore, MerkleTree, NodeIndex, PartialMerkleTree,
            SimpleSmt,
        },
        random::{RandomCoin, RpoRandomCoin, RpxRandomCoin, WinterRandomCoin},
    };
}

// TYPE ALIASES
// ================================================================================================

type QuadFelt = QuadExtension<Felt>;

type SysTrace = [Vec<Felt>; SYS_TRACE_WIDTH];

pub struct DecoderTrace {
    trace: [Vec<Felt>; DECODER_TRACE_WIDTH],
    aux_builder: decoder::AuxTraceBuilder,
}

pub struct StackTrace {
    trace: [Vec<Felt>; STACK_TRACE_WIDTH],
}

pub struct RangeCheckTrace {
    trace: [Vec<Felt>; RANGE_CHECK_TRACE_WIDTH],
    aux_builder: range::AuxTraceBuilder,
}

pub struct ChipletsTrace {
    trace: [Vec<Felt>; CHIPLETS_WIDTH],
    aux_builder: chiplets::AuxTraceBuilder,
}

// EXECUTORS
// ================================================================================================

/// Returns an execution trace resulting from executing the provided program against the provided
/// inputs.
#[tracing::instrument("execute_program", skip_all)]
pub fn execute<H>(
    program: &Program,
    stack_inputs: StackInputs,
    host: H,
    options: ExecutionOptions,
) -> Result<ExecutionTrace, ExecutionError>
where
    H: Host,
{
    let mut process = Process::new(program.kernel().clone(), stack_inputs, host, options);
    let stack_outputs = process.execute(program)?;
    let trace = ExecutionTrace::new(process, stack_outputs);
    assert_eq!(&program.hash(), trace.program_hash(), "inconsistent program hash");
    Ok(trace)
}

/// Returns an iterator which allows callers to step through the execution and inspect VM state at
/// each execution step.
pub fn execute_iter<H>(program: &Program, stack_inputs: StackInputs, host: H) -> VmStateIterator
where
    H: Host,
{
    let mut process = Process::new_debug(program.kernel().clone(), stack_inputs, host);
    let result = process.execute(program);
    if result.is_ok() {
        assert_eq!(
            program.hash(),
            process.decoder.program_hash().into(),
            "inconsistent program hash"
        );
    }
    VmStateIterator::new(process, result)
}

// PROCESS
// ================================================================================================

/// A [Process] is the underlying execution engine for a Miden [Program].
///
/// Typically, you do not need to worry about, or use [Process] directly, instead you should prefer
/// to use either [execute] or [execute_iter], which also handle setting up the process state,
/// inputs, as well as compute the [ExecutionTrace] for the program.
///
/// However, for situations in which you want finer-grained control over those steps, you will need
/// to construct an instance of [Process] using [Process::new], invoke [Process::execute], and then
/// get the execution trace using [ExecutionTrace::new] using the outputs produced by execution.
#[cfg(not(any(test, feature = "testing")))]
pub struct Process<H>
where
    H: Host,
{
    system: System,
    decoder: Decoder,
    stack: Stack,
    range: RangeChecker,
    chiplets: Chiplets,
    host: RefCell<H>,
    max_cycles: u32,
    enable_tracing: bool,
}

#[cfg(any(test, feature = "testing"))]
pub struct Process<H>
where
    H: Host,
{
    pub system: System,
    pub decoder: Decoder,
    pub stack: Stack,
    pub range: RangeChecker,
    pub chiplets: Chiplets,
    pub host: RefCell<H>,
    pub max_cycles: u32,
    pub enable_tracing: bool,
}

impl<H> Process<H>
where
    H: Host,
{
    // CONSTRUCTORS
    // --------------------------------------------------------------------------------------------
    /// Creates a new process with the provided inputs.
    pub fn new(
        kernel: Kernel,
        stack_inputs: StackInputs,
        host: H,
        execution_options: ExecutionOptions,
    ) -> Self {
        Self::initialize(kernel, stack_inputs, host, execution_options)
    }

    /// Creates a new process with provided inputs and debug options enabled.
    pub fn new_debug(kernel: Kernel, stack_inputs: StackInputs, host: H) -> Self {
        Self::initialize(
            kernel,
            stack_inputs,
            host,
            ExecutionOptions::default().with_tracing().with_debugging(),
        )
    }

    fn initialize(
        kernel: Kernel,
        stack: StackInputs,
        host: H,
        execution_options: ExecutionOptions,
    ) -> Self {
        let in_debug_mode = execution_options.enable_debugging();
        Self {
            system: System::new(execution_options.expected_cycles() as usize),
            decoder: Decoder::new(in_debug_mode),
            stack: Stack::new(&stack, execution_options.expected_cycles() as usize, in_debug_mode),
            range: RangeChecker::new(),
            chiplets: Chiplets::new(kernel),
            host: RefCell::new(host),
            max_cycles: execution_options.max_cycles(),
            enable_tracing: execution_options.enable_tracing(),
        }
    }

    // PROGRAM EXECUTOR
    // --------------------------------------------------------------------------------------------

    /// Executes the provided [`Program`] in this process.
    pub fn execute(&mut self, program: &Program) -> Result<StackOutputs, ExecutionError> {
        if self.system.clk() != 0 {
            return Err(ExecutionError::ProgramAlreadyExecuted);
        }

        self.execute_mast_node(program.entrypoint(), &program.mast_forest().clone())?;

        self.stack.build_stack_outputs()
    }

    // NODE EXECUTORS
    // --------------------------------------------------------------------------------------------

    fn execute_mast_node(
        &mut self,
        node_id: MastNodeId,
        program: &MastForest,
    ) -> Result<(), ExecutionError> {
        let node = program
            .get_node_by_id(node_id)
            .ok_or(ExecutionError::MastNodeNotFoundInForest { node_id })?;

        match node {
            MastNode::Block(node) => self.execute_basic_block_node(node),
            MastNode::Join(node) => self.execute_join_node(node, program),
            MastNode::Split(node) => self.execute_split_node(node, program),
            MastNode::Loop(node) => self.execute_loop_node(node, program),
            MastNode::Call(node) => self.execute_call_node(node, program),
            MastNode::Dyn => self.execute_dyn_node(program),
            MastNode::External(external_node) => {
                let node_digest = external_node.digest();
                let mast_forest = self
                    .host
                    .borrow()
                    .get_mast_forest(&node_digest)
                    .ok_or(ExecutionError::MastForestNotFound { root_digest: node_digest })?;

                // We limit the parts of the program that can be called externally to procedure
                // roots, even though MAST doesn't have that restriction.
                let root_id = mast_forest.find_procedure_root(node_digest).ok_or(
                    ExecutionError::MalformedMastForestInHost { root_digest: node_digest },
                )?;

                // if the node that we got by looking up an external reference is also an External
                // node, we are about to enter into an infinite loop - so, return an error
                if mast_forest[root_id].is_external() {
                    return Err(ExecutionError::CircularExternalNode(node_digest));
                }

                self.execute_mast_node(root_id, &mast_forest)
            },
        }
    }

    /// Executes the specified [JoinNode].
    #[inline(always)]
    fn execute_join_node(
        &mut self,
        node: &JoinNode,
        program: &MastForest,
    ) -> Result<(), ExecutionError> {
        self.start_join_node(node, program)?;

        // execute first and then second child of the join block
        self.execute_mast_node(node.first(), program)?;
        self.execute_mast_node(node.second(), program)?;

        self.end_join_node(node)
    }

    /// Executes the specified [SplitNode].
    #[inline(always)]
    fn execute_split_node(
        &mut self,
        node: &SplitNode,
        program: &MastForest,
    ) -> Result<(), ExecutionError> {
        // start the SPLIT block; this also pops the stack and returns the popped element
        let condition = self.start_split_node(node, program)?;

        // execute either the true or the false branch of the split block based on the condition
        if condition == ONE {
            self.execute_mast_node(node.on_true(), program)?;
        } else if condition == ZERO {
            self.execute_mast_node(node.on_false(), program)?;
        } else {
            return Err(ExecutionError::NotBinaryValue(condition));
        }

        self.end_split_node(node)
    }

    /// Executes the specified [LoopNode].
    #[inline(always)]
    fn execute_loop_node(
        &mut self,
        node: &LoopNode,
        program: &MastForest,
    ) -> Result<(), ExecutionError> {
        // start the LOOP block; this also pops the stack and returns the popped element
        let condition = self.start_loop_node(node, program)?;

        // if the top of the stack is ONE, execute the loop body; otherwise skip the loop body
        if condition == ONE {
            // execute the loop body at least once
            self.execute_mast_node(node.body(), program)?;

            // keep executing the loop body until the condition on the top of the stack is no
            // longer ONE; each iteration of the loop is preceded by executing REPEAT operation
            // which drops the condition from the stack
            while self.stack.peek() == ONE {
                self.decoder.repeat();
                self.execute_op(Operation::Drop)?;
                self.execute_mast_node(node.body(), program)?;
            }

            // end the LOOP block and drop the condition from the stack
            self.end_loop_node(node, true)
        } else if condition == ZERO {
            // end the LOOP block, but don't drop the condition from the stack because it was
            // already dropped when we started the LOOP block
            self.end_loop_node(node, false)
        } else {
            Err(ExecutionError::NotBinaryValue(condition))
        }
    }

    /// Executes the specified [CallNode].
    #[inline(always)]
    fn execute_call_node(
        &mut self,
        call_node: &CallNode,
        program: &MastForest,
    ) -> Result<(), ExecutionError> {
        // if this is a syscall, make sure the call target exists in the kernel
        if call_node.is_syscall() {
            let callee = program.get_node_by_id(call_node.callee()).ok_or_else(|| {
                ExecutionError::MastNodeNotFoundInForest { node_id: call_node.callee() }
            })?;
            self.chiplets.access_kernel_proc(callee.digest())?;
        }

        self.start_call_node(call_node, program)?;
        self.execute_mast_node(call_node.callee(), program)?;
        self.end_call_node(call_node)
    }

    /// Executes the specified [vm_core::mast::DynNode].
    ///
    /// The MAST root of the callee is assumed to be at the top of the stack, and the callee is
    /// expected to be either in the current `program` or in the host.
    #[inline(always)]
    fn execute_dyn_node(&mut self, program: &MastForest) -> Result<(), ExecutionError> {
        // get target hash from the stack
        let callee_hash = self.stack.get_word(0);
        self.start_dyn_node(callee_hash)?;

        // if the callee is not in the program's MAST forest, try to find a MAST forest for it in
        // the host (corresponding to an external library loaded in the host); if none are
        // found, return an error.
        match program.find_procedure_root(callee_hash.into()) {
            Some(callee_id) => self.execute_mast_node(callee_id, program)?,
            None => {
                let mast_forest = self
                    .host
                    .borrow()
                    .get_mast_forest(&callee_hash.into())
                    .ok_or_else(|| ExecutionError::DynamicNodeNotFound(callee_hash.into()))?;

                // We limit the parts of the program that can be called externally to procedure
                // roots, even though MAST doesn't have that restriction.
                let root_id = mast_forest.find_procedure_root(callee_hash.into()).ok_or(
                    ExecutionError::MalformedMastForestInHost { root_digest: callee_hash.into() },
                )?;

                self.execute_mast_node(root_id, &mast_forest)?
            },
        }

        self.end_dyn_node()
    }

    /// Executes the specified [BasicBlockNode].
    #[inline(always)]
    fn execute_basic_block_node(
        &mut self,
        basic_block: &BasicBlockNode,
    ) -> Result<(), ExecutionError> {
        self.start_basic_block_node(basic_block)?;

        let mut op_offset = 0;
        let mut decorators = basic_block.decorator_iter();

        // execute the first operation batch
        self.execute_op_batch(&basic_block.op_batches()[0], &mut decorators, op_offset)?;
        op_offset += basic_block.op_batches()[0].ops().len();

        // if the span contains more operation batches, execute them. each additional batch is
        // preceded by a RESPAN operation; executing RESPAN operation does not change the state
        // of the stack
        for op_batch in basic_block.op_batches().iter().skip(1) {
            self.respan(op_batch);
            self.execute_op(Operation::Noop)?;
            self.execute_op_batch(op_batch, &mut decorators, op_offset)?;
            op_offset += op_batch.ops().len();
        }

        self.end_basic_block_node(basic_block)?;

        // execute any decorators which have not been executed during span ops execution; this
        // can happen for decorators appearing after all operations in a block. these decorators
        // are executed after SPAN block is closed to make sure the VM clock cycle advances beyond
        // the last clock cycle of the SPAN block ops.
        for decorator in decorators {
            self.execute_decorator(decorator)?;
        }

        Ok(())
    }

    /// Executes all operations in an [OpBatch]. This also ensures that all alignment rules are
    /// satisfied by executing NOOPs as needed. Specifically:
    /// - If an operation group ends with an operation carrying an immediate value, a NOOP is
    ///   executed after it.
    /// - If the number of groups in a batch is not a power of 2, NOOPs are executed (one per group)
    ///   to bring it up to the next power of two (e.g., 3 -> 4, 5 -> 8).
    #[inline(always)]
    fn execute_op_batch(
        &mut self,
        batch: &OpBatch,
        decorators: &mut DecoratorIterator,
        op_offset: usize,
    ) -> Result<(), ExecutionError> {
        let op_counts = batch.op_counts();
        let mut op_idx = 0;
        let mut group_idx = 0;
        let mut next_group_idx = 1;

        // round up the number of groups to be processed to the next power of two; we do this
        // because the processor requires the number of groups to be either 1, 2, 4, or 8; if
        // the actual number of groups is smaller, we'll pad the batch with NOOPs at the end
        let num_batch_groups = batch.num_groups().next_power_of_two();

        // execute operations in the batch one by one
        for (i, &op) in batch.ops().iter().enumerate() {
            while let Some(decorator) = decorators.next_filtered(i + op_offset) {
                self.execute_decorator(decorator)?;
            }

            // decode and execute the operation
            self.decoder.execute_user_op(op, op_idx);
            self.execute_op(op)?;

            // if the operation carries an immediate value, the value is stored at the next group
            // pointer; so, we advance the pointer to the following group
            let has_imm = op.imm_value().is_some();
            if has_imm {
                next_group_idx += 1;
            }

            // determine if we've executed all non-decorator operations in a group
            if op_idx == op_counts[group_idx] - 1 {
                // if we are at the end of the group, first check if the operation carries an
                // immediate value
                if has_imm {
                    // an operation with an immediate value cannot be the last operation in a group
                    // so, we need execute a NOOP after it. the assert also makes sure that there
                    // is enough room in the group to execute a NOOP (if there isn't, there is a
                    // bug somewhere in the assembler)
                    debug_assert!(op_idx < OP_GROUP_SIZE - 1, "invalid op index");
                    self.decoder.execute_user_op(Operation::Noop, op_idx + 1);
                    self.execute_op(Operation::Noop)?;
                }

                // then, move to the next group and reset operation index
                group_idx = next_group_idx;
                next_group_idx += 1;
                op_idx = 0;

                // if we haven't reached the end of the batch yet, set up the decoder for
                // decoding the next operation group
                if group_idx < num_batch_groups {
                    self.decoder.start_op_group(batch.groups()[group_idx]);
                }
            } else {
                // if we are not at the end of the group, just increment the operation index
                op_idx += 1;
            }
        }

        // make sure we execute the required number of operation groups; this would happen when
        // the actual number of operation groups was not a power of two
        for group_idx in group_idx..num_batch_groups {
            self.decoder.execute_user_op(Operation::Noop, 0);
            self.execute_op(Operation::Noop)?;

            // if we are not at the last group yet, set up the decoder for decoding the next
            // operation groups. the groups were are processing are just NOOPs - so, the op group
            // value is ZERO
            if group_idx < num_batch_groups - 1 {
                self.decoder.start_op_group(ZERO);
            }
        }

        Ok(())
    }

    /// Executes the specified decorator
    fn execute_decorator(&mut self, decorator: &Decorator) -> Result<(), ExecutionError> {
        match decorator {
            Decorator::Advice(injector) => {
                self.host.borrow_mut().set_advice(self, *injector)?;
            },
            Decorator::Debug(options) => {
                self.host.borrow_mut().on_debug(self, options)?;
            },
            Decorator::AsmOp(assembly_op) => {
                if self.decoder.in_debug_mode() {
                    self.decoder.append_asmop(self.system.clk(), assembly_op.clone());
                }
            },
            Decorator::Event(id) => {
                self.host.borrow_mut().on_event(self, *id)?;
            },
            Decorator::Trace(id) => {
                if self.enable_tracing {
                    self.host.borrow_mut().on_trace(self, *id)?;
                }
            },
        }
        Ok(())
    }

    // PUBLIC ACCESSORS
    // --------------------------------------------------------------------------------------------

    pub const fn kernel(&self) -> &Kernel {
        self.chiplets.kernel()
    }

    pub fn into_parts(self) -> (System, Decoder, Stack, RangeChecker, Chiplets, H) {
        (
            self.system,
            self.decoder,
            self.stack,
            self.range,
            self.chiplets,
            self.host.into_inner(),
        )
    }
}

// PROCESS STATE
// ================================================================================================

/// A trait that defines a set of methods which allow access to the state of the process.
pub trait ProcessState {
    /// Returns the current clock cycle of a process.
    fn clk(&self) -> RowIndex;

    /// Returns the current execution context ID.
    fn ctx(&self) -> ContextId;

    /// Returns the current value of the free memory pointer.
    fn fmp(&self) -> u64;

    /// Returns the value located at the specified position on the stack at the current clock cycle.
    fn get_stack_item(&self, pos: usize) -> Felt;

    /// Returns a word located at the specified word index on the stack.
    ///
    /// Specifically, word 0 is defined by the first 4 elements of the stack, word 1 is defined
    /// by the next 4 elements etc. Since the top of the stack contains 4 word, the highest valid
    /// word index is 3.
    ///
    /// The words are created in reverse order. For example, for word 0 the top element of the
    /// stack will be at the last position in the word.
    ///
    /// Creating a word does not change the state of the stack.
    fn get_stack_word(&self, word_idx: usize) -> Word;

    /// Returns stack state at the current clock cycle. This includes the top 16 items of the
    /// stack + overflow entries.
    fn get_stack_state(&self) -> Vec<Felt>;

    /// Returns a word located at the specified context/address, or None if the address hasn't
    /// been accessed previously.
    fn get_mem_value(&self, ctx: ContextId, addr: u32) -> Option<Word>;

    /// Returns the entire memory state for the specified execution context at the current clock
    /// cycle.
    ///
    /// The state is returned as a vector of (address, value) tuples, and includes addresses which
    /// have been accessed at least once.
    fn get_mem_state(&self, ctx: ContextId) -> Vec<(u64, Word)>;
}

impl<H: Host> ProcessState for Process<H> {
    fn clk(&self) -> RowIndex {
        self.system.clk()
    }

    fn ctx(&self) -> ContextId {
        self.system.ctx()
    }

    fn fmp(&self) -> u64 {
        self.system.fmp().as_int()
    }

    fn get_stack_item(&self, pos: usize) -> Felt {
        self.stack.get(pos)
    }

    fn get_stack_word(&self, word_idx: usize) -> Word {
        self.stack.get_word(word_idx)
    }

    fn get_stack_state(&self) -> Vec<Felt> {
        self.stack.get_state_at(self.system.clk())
    }

    fn get_mem_value(&self, ctx: ContextId, addr: u32) -> Option<Word> {
        self.chiplets.get_mem_value(ctx, addr)
    }

    fn get_mem_state(&self, ctx: ContextId) -> Vec<(u64, Word)> {
        self.chiplets.get_mem_state_at(ctx, self.system.clk())
    }
}
