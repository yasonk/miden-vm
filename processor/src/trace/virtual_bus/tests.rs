use crate::{
    prove_virtual_bus,
    trace::virtual_bus::{multilinear::EqFunction, sum_check::FinalOpeningClaim},
    verify_virtual_bus, DefaultHost, ExecutionTrace, Process,
};
use alloc::vec::Vec;
use miden_air::{
    trace::{main_trace::MainTrace, range::M_COL_IDX},
    ExecutionOptions,
};
use vm_core::{
    code_blocks::CodeBlock, CodeBlockTable, Felt, FieldElement, Kernel, Operation, StackInputs,
};
use vm_core::{
    crypto::{
        hash::Rpo256,
        random::{RandomCoin, RpoRandomCoin},
    },
    polynom,
};
use winter_prover::math::fft;

#[test]
fn test_vb_prover_verifier() {
    let s = 6;
    let o = 6;
    let stack: Vec<_> = (0..(1 << s)).into_iter().collect();
    let operations: Vec<_> = (0..(1 << o))
        .flat_map(|_| {
            vec![Operation::U32split, Operation::U32add, Operation::U32xor, Operation::MStoreW]
        })
        .collect();

    let trace = build_full_trace(&stack, operations, Kernel::default());

    // this should be generated using the transcript up to when the prover sends the commitment
    // to the main trace.
    let log_up_randomness: Vec<Felt> = vec![test_utils::rand::rand_value()];

    let seed = [Felt::ZERO; 4]; // should be initialized with the appropriate transcript
    let mut transcript = RpoRandomCoin::new(seed.into());
    let proof = prove_virtual_bus(&trace, log_up_randomness.clone(), &mut transcript).unwrap();

    let seed = [Felt::ZERO; 4]; // should be initialized with the appropriate transcript
    let mut transcript = RpoRandomCoin::new(seed.into());
    let final_opening_claim =
        verify_virtual_bus(Felt::ZERO, proof, log_up_randomness, &mut transcript);
    assert!(final_opening_claim.is_ok());

    // Final evaluation claim proof using univariate sum-check IOP

    // get evaluation point and evaluation claims
    let FinalOpeningClaim {
        eval_point,
        openings,
    } = final_opening_claim.unwrap();

    // generate the lambda to batch the opening claims
    transcript.reseed(Rpo256::hash_elements(&openings));
    let lambdas: Vec<Felt> = (0..openings.len()).map(|_| transcript.draw().unwrap()).collect();

    // compute the batched opening claim
    let batched_opening = (0..openings.len())
        .map(|i| openings[i] * lambdas[i])
        .fold(Felt::ZERO, |acc, term| acc + term);

    // compute the L and Q polynomial evaluations
    let lagrange_kernel_evals = EqFunction::new(eval_point).evaluations();
    let columns: Vec<Vec<Felt>> = trace.clone().into_columns();
    let trace_len = columns[0].len();

    let big_q_evals: Vec<Felt> = (0..columns[0].len())
        .map(|i| {
            (0..openings.len())
                .map(|j| columns[j][i] * lambdas[j])
                .fold(Felt::ZERO, |acc, term| acc + term)
        })
        .collect();


    // compute the coefficients of L(X) * Q(X)
    let mut big_q_coef = big_q_evals.clone();
    let mut lagrange_kernel_coef = lagrange_kernel_evals.clone();
    let inv_twiddles = fft::get_inv_twiddles::<Felt>(big_q_coef.len());
    fft::interpolate_poly_with_offset(&mut big_q_coef, &inv_twiddles, Felt::ONE);
    fft::interpolate_poly_with_offset(&mut lagrange_kernel_coef, &inv_twiddles, Felt::ONE);
    // TODO: should be done using FFT
    let prod_poly = polynom::mul(&big_q_coef, &lagrange_kernel_coef);

    // compute L(X) * Q(X) - claim / |H| where H is the trace domain
    let correction_term = batched_opening / Felt::from((trace_len) as u32);
    let corrected_product_poly = polynom::sub(&prod_poly, &[correction_term]);

    // compute q and r such that L(X) * Q(X) - claim / |H| = q(X) * (X^|H| - 1) + r(X) with r(X) = X * f(X) and deg(f) < n - 1 and deg(g) < n
    let mut quotient_coef = corrected_product_poly;
    let mut remainder_coef = syn_div_with_remainder_in_place(&mut quotient_coef, trace_len, Felt::ONE);
    assert_eq!(remainder_coef[0], Felt::ZERO);

    // only f(X) is needed 
    remainder_coef.rotate_left(1);
    assert!(remainder_coef[0] != Felt::ZERO);
    assert_eq!(remainder_coef[remainder_coef.len() - 1], Felt::ZERO);

    // the verifier generates a random OOD point
    let z: Felt = transcript.draw().unwrap();

    // the prover provides the OOD of the trace columns
    let mut ood_evals = vec![];
    for col in columns {
        let mut poly_tmp = col;
        fft::interpolate_poly_with_offset(&mut poly_tmp, &inv_twiddles, Felt::ONE);
        ood_evals.push(polynom::eval(&poly_tmp, z))
    }
    assert_eq!(ood_evals.len(), openings.len());

    // the prover also provides the OOD of the Lagrange as well as the f(X) and q(X) polys
    let lagrange_at_z = polynom::eval(&lagrange_kernel_coef, z);
    let quotient_at_z = polynom::eval(&quotient_coef, z);
    let remainder_at_z = polynom::eval(&remainder_coef, z);

    // the verifier checks the following identity
    let lhs = lagrange_at_z
        * lambdas
            .iter()
            .zip(ood_evals.iter())
            .fold(Felt::ZERO, |acc, (&lambda, &eval)| acc + lambda * eval)
        - correction_term;
    let rhs = z * remainder_at_z + (z.exp((trace_len as u32).into()) - Felt::ONE) * quotient_at_z;

    assert_eq!(lhs, rhs);
    // the only remaining thing is to now check that f(X) and q(X) are low-degree
}

pub fn syn_div_with_remainder_in_place<E>(p: &mut [E], a: usize, b: E) -> Vec<E>
where
    E: FieldElement,
{
    assert!(a != 0, "divisor degree cannot be zero");
    assert!(b != E::ZERO, "constant cannot be zero");
    assert!(p.len() > a, "divisor degree  cannot be greater than dividend size");

    if a == 1 {
        // if we are dividing by (x - `b`), we can use a single variable to keep track
        // of the remainder; this way, we can avoid shifting the values in the slice later
        let mut c = E::ZERO;
        for coeff in p.iter_mut().rev() {
            *coeff += b * c;
            core::mem::swap(coeff, &mut c);
        }
        vec![c]
    } else {
        // if we are dividing by a polynomial of higher power, we need to keep track of the
        // full remainder. we do that in place, but then need to shift the values at the end
        // to discard the remainder
        let degree_offset = p.len() - a;
        if b == E::ONE {
            // if `b` is 1, no need to multiply by `b` in every iteration of the loop
            for i in (0..degree_offset).rev() {
                p[i] += p[i + a];
            }
        } else {
            for i in (0..degree_offset).rev() {
                p[i] += p[i + a] * b;
            }
        }
        // get the remainder
        let mut remainder: Vec<_> = vec![];
        remainder.extend_from_slice(&p[..a]);
        p.copy_within(a.., 0);
        p[degree_offset..].fill(E::ZERO);
        remainder
    }
}

#[test]
fn test_vb_prover_verifier_failure() {
    let s = 6;
    let o = 6;
    let stack: Vec<_> = (0..(1 << s)).into_iter().collect();
    let operations: Vec<_> = (0..(1 << o))
        .flat_map(|_| {
            vec![Operation::U32split, Operation::U32add, Operation::U32xor, Operation::MStoreW]
        })
        .collect();

    // modifying the multiplicity
    let mut trace = build_full_trace(&stack, operations, Kernel::default());
    let index = trace.get_column(M_COL_IDX).iter().position(|v| *v != Felt::ZERO).unwrap();
    trace.get_column_mut(M_COL_IDX)[index] = Felt::ONE;

    // this should be generated using the transcript up to when the prover sends the commitment
    // to the main trace.
    let log_up_randomness: Vec<Felt> = vec![test_utils::rand::rand_value()];

    let seed = [Felt::ZERO; 4]; // should be initialized with the appropriate transcript
    let mut transcript = RpoRandomCoin::new(seed.into());
    let proof = prove_virtual_bus(&trace, log_up_randomness.clone(), &mut transcript).unwrap();

    let seed = [Felt::ZERO; 4]; // should be initialized with the appropriate transcript
    let mut transcript = RpoRandomCoin::new(seed.into());
    let final_opening_claim =
        verify_virtual_bus(Felt::ZERO, proof, log_up_randomness, &mut transcript);
    assert!(final_opening_claim.is_err())
}

fn build_full_trace(stack_inputs: &[u64], operations: Vec<Operation>, kernel: Kernel) -> MainTrace {
    let stack_inputs: Vec<Felt> = stack_inputs.iter().map(|a| Felt::new(*a)).collect();
    let stack_inputs = StackInputs::new(stack_inputs).unwrap();
    let host = DefaultHost::default();
    let mut process = Process::new(kernel, stack_inputs, host, ExecutionOptions::default());
    let program = CodeBlock::new_span(operations);
    process.execute_code_block(&program, &CodeBlockTable::default()).unwrap();
    let (trace, _, _): (MainTrace, _, _) = ExecutionTrace::test_finalize_trace(process);

    trace
}
