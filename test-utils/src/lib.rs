#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

// IMPORTS
// ================================================================================================

#[cfg(not(target_family = "wasm"))]
use alloc::format;
use alloc::{
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};

use assembly::Library;
// EXPORTS
// ================================================================================================
pub use assembly::{diagnostics::Report, LibraryPath, SourceFile, SourceManager};
pub use pretty_assertions::{assert_eq, assert_ne, assert_str_eq};
pub use processor::{
    AdviceInputs, AdviceProvider, ContextId, DefaultHost, ExecutionError, ExecutionOptions,
    ExecutionTrace, Process, ProcessState, StackInputs, VmStateIterator,
};
use processor::{MastForest, Program};
#[cfg(not(target_family = "wasm"))]
use proptest::prelude::{Arbitrary, Strategy};
pub use prover::{prove, MemAdviceProvider, ProvingOptions};
pub use test_case::test_case;
pub use verifier::{verify, AcceptableOptions, VerifierError};
use vm_core::{chiplets::hasher::apply_permutation, ProgramInfo};
pub use vm_core::{
    chiplets::hasher::{hash_elements, STATE_WIDTH},
    stack::STACK_TOP_SIZE,
    utils::{collections, group_slice_elements, IntoBytes, ToElements},
    Felt, FieldElement, StarkField, Word, EMPTY_WORD, ONE, WORD_SIZE, ZERO,
};

pub mod math {
    pub use winter_prover::math::{
        fft, fields::QuadExtension, polynom, ExtensionOf, FieldElement, StarkField, ToElements,
    };
}

pub mod serde {
    pub use vm_core::utils::{
        ByteReader, ByteWriter, Deserializable, DeserializationError, Serializable, SliceReader,
    };
}

pub mod crypto;

#[cfg(not(target_family = "wasm"))]
pub mod rand;

mod test_builders;

#[cfg(not(target_family = "wasm"))]
pub use proptest;

// TYPE ALIASES
// ================================================================================================

pub type QuadFelt = vm_core::QuadExtension<Felt>;

// CONSTANTS
// ================================================================================================

/// A value just over what a [u32] integer can hold.
pub const U32_BOUND: u64 = u32::MAX as u64 + 1;

// TEST HANDLER
// ================================================================================================

/// Asserts that running the given assembler test will result in the expected error.
#[cfg(all(feature = "std", not(target_family = "wasm")))]
#[macro_export]
macro_rules! expect_assembly_error {
    ($test:expr, $(|)? $( $pattern:pat_param )|+ $( if $guard: expr )? $(,)?) => {
        let error = $test.compile().expect_err("expected assembly to fail");
        match error.downcast::<assembly::AssemblyError>() {
            Ok(error) => {
                ::vm_core::assert_matches!(error, $( $pattern )|+ $( if $guard )?);
            }
            Err(report) => {
                panic!(r#"
assertion failed (expected assembly error, but got a different type):
    left: `{:?}`,
    right: `{}`"#, report, stringify!($($pattern)|+ $(if $guard)?));
            }
        }
    };
}

/// Asserts that running the given execution test will result in the expected error.
#[cfg(all(feature = "std", not(target_family = "wasm")))]
#[macro_export]
macro_rules! expect_exec_error_matches {
    ($test:expr, $(|)? $( $pattern:pat_param )|+ $( if $guard: expr )? $(,)?) => {
        match $test.execute() {
            Ok(_) => panic!("expected execution to fail @ {}:{}", file!(), line!()),
            Err(error) => ::vm_core::assert_matches!(error, $( $pattern )|+ $( if $guard )?),
        }
    };
}

/// Asserts that running the given execution test will result in the expected error.
#[cfg(all(feature = "std", not(target_family = "wasm")))]
#[macro_export]
macro_rules! expect_exec_error {
    ($test:expr, $expected:expr) => {
        match $test.execute() {
            Ok(_) => panic!("expected execution to fail @ {}:{}", file!(), line!()),
            Err(error) => $crate::assert_eq!(error, $expected),
        }
    };
}

/// Like [assembly::assert_diagnostic], but matches each non-empty line of the rendered output
/// to a corresponding pattern.
///
/// So if the output has 3 lines, the second of which is empty, and you provide 2 patterns, the
/// assertion passes if the first line matches the first pattern, and the third line matches the
/// second pattern - the second line is ignored because it is empty.
#[cfg(all(feature = "std", not(target_family = "wasm")))]
#[macro_export]
macro_rules! assert_diagnostic_lines {
    ($diagnostic:expr, $($expected:expr),+) => {{
        use assembly::testing::Pattern;
        let actual = format!("{}", assembly::diagnostics::reporting::PrintDiagnostic::new_without_color($diagnostic));
        let lines = actual.lines().filter(|l| !l.trim().is_empty()).zip([$(Pattern::from($expected)),*].into_iter());
        for (actual_line, expected) in lines {
            expected.assert_match_with_context(actual_line, &actual);
        }
    }};
}

#[cfg(all(feature = "std", not(target_family = "wasm")))]
#[macro_export]
macro_rules! assert_assembler_diagnostic {
    ($test:ident, $($expected:literal),+) => {{
        let error = $test
            .compile()
            .expect_err("expected diagnostic to be raised, but compilation succeeded");
        assert_diagnostic_lines!(error, $($expected),*);
    }};

    ($test:ident, $($expected:expr),+) => {{
        let error = $test
            .compile()
            .expect_err("expected diagnostic to be raised, but compilation succeeded");
        assert_diagnostic_lines!(error, $($expected),*);
    }};
}

/// This is a container for the data required to run tests, which allows for running several
/// different types of tests.
///
/// Types of valid result tests:
/// - Execution test: check that running a program compiled from the given source has the specified
///   results for the given (optional) inputs.
/// - Proptest: run an execution test inside a proptest.
///
/// Types of failure tests:
/// - Assembly error test: check that attempting to compile the given source causes an AssemblyError
///   which contains the specified substring.
/// - Execution error test: check that running a program compiled from the given source causes an
///   ExecutionError which contains the specified substring.
pub struct Test {
    pub source_manager: Arc<dyn SourceManager>,
    pub source: Arc<SourceFile>,
    pub kernel_source: Option<Arc<SourceFile>>,
    pub stack_inputs: StackInputs,
    pub advice_inputs: AdviceInputs,
    pub in_debug_mode: bool,
    pub libraries: Vec<Library>,
    pub add_modules: Vec<(LibraryPath, String)>,
}

impl Test {
    // CONSTRUCTOR
    // --------------------------------------------------------------------------------------------

    /// Creates the simplest possible new test, with only a source string and no inputs.
    pub fn new(name: &str, source: &str, in_debug_mode: bool) -> Self {
        let source_manager = Arc::new(assembly::DefaultSourceManager::default());
        let source = source_manager.load(name, source.to_string());
        Test {
            source_manager,
            source,
            kernel_source: None,
            stack_inputs: StackInputs::default(),
            advice_inputs: AdviceInputs::default(),
            in_debug_mode,
            libraries: Vec::default(),
            add_modules: Vec::default(),
        }
    }

    /// Add an extra module to link in during assembly
    pub fn add_module(&mut self, path: assembly::LibraryPath, source: impl ToString) {
        self.add_modules.push((path, source.to_string()));
    }

    // TEST METHODS
    // --------------------------------------------------------------------------------------------

    /// Builds a final stack from the provided stack-ordered array and asserts that executing the
    /// test will result in the expected final stack state.
    #[track_caller]
    pub fn expect_stack(&self, final_stack: &[u64]) {
        let result = stack_to_ints(&self.get_last_stack_state());
        let expected = stack_top_to_ints(final_stack);
        assert_eq!(expected, result, "Expected stack to be {:?}, found {:?}", expected, result);
    }

    /// Executes the test and validates that the process memory has the elements of `expected_mem`
    /// at address `mem_start_addr` and that the end of the stack execution trace matches the
    /// `final_stack`.
    #[track_caller]
    pub fn expect_stack_and_memory(
        &self,
        final_stack: &[u64],
        mut mem_start_addr: u32,
        expected_mem: &[u64],
    ) {
        // compile the program
        let (program, kernel) = self.compile().expect("Failed to compile test source.");
        let mut host = DefaultHost::new(MemAdviceProvider::from(self.advice_inputs.clone()));
        if let Some(kernel) = kernel {
            host.load_mast_forest(kernel);
        }
        for library in &self.libraries {
            host.load_mast_forest(library.mast_forest().clone());
        }

        // execute the test
        let mut process = Process::new(
            program.kernel().clone(),
            self.stack_inputs.clone(),
            host,
            ExecutionOptions::default(),
        );
        process.execute(&program).unwrap();

        // validate the memory state
        for data in expected_mem.chunks(WORD_SIZE) {
            // Main memory is zeroed by default, use zeros as a fallback when unwrap to make testing
            // easier
            let mem_state =
                process.get_mem_value(ContextId::root(), mem_start_addr).unwrap_or(EMPTY_WORD);

            let mem_state = stack_to_ints(&mem_state);
            assert_eq!(
                data, mem_state,
                "Expected memory [{}] => {:?}, found {:?}",
                mem_start_addr, data, mem_state
            );
            mem_start_addr += 1;
        }

        // validate the stack state
        self.expect_stack(final_stack);
    }

    /// Asserts that executing the test inside a proptest results in the expected final stack state.
    /// The proptest will return a test failure instead of panicking if the assertion condition
    /// fails.
    #[cfg(not(target_family = "wasm"))]
    pub fn prop_expect_stack(
        &self,
        final_stack: &[u64],
    ) -> Result<(), proptest::prelude::TestCaseError> {
        let result = self.get_last_stack_state();
        proptest::prop_assert_eq!(stack_top_to_ints(final_stack), stack_to_ints(&result));

        Ok(())
    }

    // UTILITY METHODS
    // --------------------------------------------------------------------------------------------

    /// Compiles a test's source and returns the resulting Program or Assembly error.
    pub fn compile(&self) -> Result<(Program, Option<MastForest>), Report> {
        use assembly::{ast::ModuleKind, Assembler, CompileOptions};

        let (assembler, compiled_kernel) = if let Some(kernel) = self.kernel_source.clone() {
            let kernel_lib =
                Assembler::new(self.source_manager.clone()).assemble_kernel(kernel).unwrap();
            let compiled_kernel = kernel_lib.mast_forest().clone();

            (
                Assembler::with_kernel(self.source_manager.clone(), kernel_lib),
                Some(compiled_kernel),
            )
        } else {
            (Assembler::new(self.source_manager.clone()), None)
        };
        let mut assembler = self
            .add_modules
            .iter()
            .fold(assembler, |assembler, (path, source)| {
                assembler
                    .with_module_and_options(
                        source,
                        CompileOptions::new(ModuleKind::Library, path.clone()).unwrap(),
                    )
                    .expect("invalid masm source code")
            })
            .with_debug_mode(self.in_debug_mode);
        for library in &self.libraries {
            assembler.add_library(library).unwrap();
        }

        Ok((assembler.assemble_program(self.source.clone())?, compiled_kernel))
    }

    /// Compiles the test's source to a Program and executes it with the tests inputs. Returns a
    /// resulting execution trace or error.
    #[track_caller]
    pub fn execute(&self) -> Result<ExecutionTrace, ExecutionError> {
        let (program, kernel) = self.compile().expect("Failed to compile test source.");
        let mut host = DefaultHost::new(MemAdviceProvider::from(self.advice_inputs.clone()));
        if let Some(kernel) = kernel {
            host.load_mast_forest(kernel);
        }
        for library in &self.libraries {
            host.load_mast_forest(library.mast_forest().clone());
        }
        processor::execute(&program, self.stack_inputs.clone(), host, ExecutionOptions::default())
    }

    /// Compiles the test's source to a Program and executes it with the tests inputs. Returns the
    /// process once execution is finished.
    pub fn execute_process(
        &self,
    ) -> Result<Process<DefaultHost<MemAdviceProvider>>, ExecutionError> {
        let (program, kernel) = self.compile().expect("Failed to compile test source.");
        let mut host = DefaultHost::new(MemAdviceProvider::from(self.advice_inputs.clone()));
        if let Some(kernel) = kernel {
            host.load_mast_forest(kernel);
        }
        for library in &self.libraries {
            host.load_mast_forest(library.mast_forest().clone());
        }

        let mut process = Process::new(
            program.kernel().clone(),
            self.stack_inputs.clone(),
            host,
            ExecutionOptions::default(),
        );
        process.execute(&program)?;
        Ok(process)
    }

    /// Compiles the test's code into a program, then generates and verifies a proof of execution
    /// using the given public inputs and the specified number of stack outputs. When `test_fail`
    /// is true, this function will force a failure by modifying the first output.
    pub fn prove_and_verify(&self, pub_inputs: Vec<u64>, test_fail: bool) {
        let stack_inputs = StackInputs::try_from_ints(pub_inputs).unwrap();
        let (program, kernel) = self.compile().expect("Failed to compile test source.");
        let mut host = DefaultHost::new(MemAdviceProvider::from(self.advice_inputs.clone()));
        if let Some(kernel) = kernel {
            host.load_mast_forest(kernel);
        }
        for library in &self.libraries {
            host.load_mast_forest(library.mast_forest().clone());
        }
        let (mut stack_outputs, proof) =
            prover::prove(&program, stack_inputs.clone(), host, ProvingOptions::default()).unwrap();

        let program_info = ProgramInfo::from(program);
        if test_fail {
            stack_outputs.stack_mut()[0] += ONE;
            assert!(verifier::verify(program_info, stack_inputs, stack_outputs, proof).is_err());
        } else {
            let result = verifier::verify(program_info, stack_inputs, stack_outputs, proof);
            assert!(result.is_ok(), "error: {result:?}");
        }
    }

    /// Compiles the test's source to a Program and executes it with the tests inputs. Returns a
    /// VmStateIterator that allows us to iterate through each clock cycle and inspect the process
    /// state.
    pub fn execute_iter(&self) -> VmStateIterator {
        let (program, kernel) = self.compile().expect("Failed to compile test source.");
        let mut host = DefaultHost::new(MemAdviceProvider::from(self.advice_inputs.clone()));
        if let Some(kernel) = kernel {
            host.load_mast_forest(kernel);
        }
        for library in &self.libraries {
            host.load_mast_forest(library.mast_forest().clone());
        }
        processor::execute_iter(&program, self.stack_inputs.clone(), host)
    }

    /// Returns the last state of the stack after executing a test.
    #[track_caller]
    pub fn get_last_stack_state(&self) -> [Felt; STACK_TOP_SIZE] {
        let trace = self.execute().unwrap();

        trace.last_stack_state()
    }
}

// HELPER FUNCTIONS
// ================================================================================================

/// Converts an array of Felts into u64
pub fn stack_to_ints(values: &[Felt]) -> Vec<u64> {
    values.iter().map(|e| (*e).as_int()).collect()
}

pub fn stack_top_to_ints(values: &[u64]) -> Vec<u64> {
    let mut result: Vec<u64> = values.to_vec();
    result.resize(STACK_TOP_SIZE, 0);
    result
}

/// A proptest strategy for generating a random word with 4 values of type T.
#[cfg(not(target_family = "wasm"))]
pub fn prop_randw<T: Arbitrary>() -> impl Strategy<Value = Vec<T>> {
    use proptest::prelude::{any, prop};
    prop::collection::vec(any::<T>(), 4)
}

/// Given a hasher state, perform one permutation.
///
/// The values of `values` should be:
/// - 0..4 the capacity
/// - 4..12 the rate
///
/// Return the result of the permutation in stack order.
pub fn build_expected_perm(values: &[u64]) -> [Felt; STATE_WIDTH] {
    let mut expected = [ZERO; STATE_WIDTH];
    for (&value, result) in values.iter().zip(expected.iter_mut()) {
        *result = Felt::new(value);
    }
    apply_permutation(&mut expected);
    expected.reverse();

    expected
}

pub fn build_expected_hash(values: &[u64]) -> [Felt; 4] {
    let digest = hash_elements(&values.iter().map(|&v| Felt::new(v)).collect::<Vec<_>>());
    let mut expected: [Felt; 4] = digest.into();
    expected.reverse();

    expected
}
