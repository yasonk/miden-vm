use processor::ExecutionError;
use test_utils::{build_expected_hash, build_expected_perm, expect_exec_error};

#[test]
fn test_invalid_end_addr() {
    // end_addr can not be smaller than start_addr
    let empty_range = "
    use.std::crypto::hashes::rpo

    begin
        push.0999 # end address
        push.1000 # start address

        exec.rpo::hash_memory_words
    end
    ";
    let test = build_test!(empty_range, &[]);
    expect_exec_error!(
        test,
        ExecutionError::FailedAssertion {
            clk: 18,
            err_code: 0,
            err_msg: None,
        }
    );

    // address range can not contain zero elements
    let empty_range = "
    use.std::crypto::hashes::rpo

    begin
        push.1000 # end address
        push.1000 # start address

        exec.rpo::hash_memory_words
    end
    ";
    let test = build_test!(empty_range, &[]);
    expect_exec_error!(
        test,
        ExecutionError::FailedAssertion {
            clk: 18,
            err_code: 0,
            err_msg: None,
        }
    );
}

#[test]
fn test_hash_empty() {
    // computes the hash for 8 consecutive zeros using mem_stream directly
    let two_zeros_mem_stream = "
    use.std::crypto::hashes::rpo

    begin
        # mem_stream state
        push.1000 padw padw padw
        mem_stream hperm

        # drop everything except the hash
        exec.rpo::squeeze_digest movup.4 drop
    end
    ";

    #[rustfmt::skip]
    let zero_hash: Vec<u64> = build_expected_hash(&[
        0, 0, 0, 0,
        0, 0, 0, 0,
    ]).into_iter().map(|e| e.as_int()).collect();
    build_test!(two_zeros_mem_stream, &[]).expect_stack(&zero_hash);

    // checks the hash compute from 8 zero elements is the same when using hash_memory_words
    let two_zeros = "
    use.std::crypto::hashes::rpo

    begin
        push.1002 # end address
        push.1000 # start address

        exec.rpo::hash_memory_words
    end
    ";

    build_test!(two_zeros, &[]).expect_stack(&zero_hash);
}

#[test]
fn test_single_iteration() {
    // computes the hash of 1 using mem_stream
    let one_memstream = "
    use.std::crypto::hashes::rpo

    begin
        # insert 1 to memory
        push.1.1000 mem_store

        # mem_stream state
        push.1000 padw padw padw
        mem_stream hperm

        # drop everything except the hash
        exec.rpo::squeeze_digest movup.4 drop
    end
    ";

    #[rustfmt::skip]
    let one_hash: Vec<u64> = build_expected_hash(&[
        1, 0, 0, 0,
        0, 0, 0, 0,
    ]).into_iter().map(|e| e.as_int()).collect();
    build_test!(one_memstream, &[]).expect_stack(&one_hash);

    // checks the hash of 1 is the same when using hash_memory_words
    // Note: This is testing the hashing of two words, so no padding is added
    // here
    let one_element = "
    use.std::crypto::hashes::rpo

    begin
        # insert 1 to memory
        push.1.1000 mem_store

        push.1002 # end address
        push.1000 # start address

        exec.rpo::hash_memory_words
    end
    ";

    build_test!(one_element, &[]).expect_stack(&one_hash);
}

#[test]
fn test_hash_one_word() {
    // computes the hash of a single 1, the procedure adds padding as required

    // This slice must not have the second word, that will be padded by the hasher with the correct
    // value
    #[rustfmt::skip]
    let one_hash: Vec<u64> = build_expected_hash(&[
        1, 0, 0, 0,
    ]).into_iter().map(|e| e.as_int()).collect();

    // checks the hash of 1 is the same when using hash_memory_words
    let one_element = "
    use.std::crypto::hashes::rpo

    begin
        push.1.1000 mem_store # push data to memory

        push.1001 # end address
        push.1000 # start address

        exec.rpo::hash_memory_words
    end
    ";

    build_test!(one_element, &[]).expect_stack(&one_hash);
}

#[test]
fn test_hash_even_words() {
    // checks the hash of two words
    let even_words = "
    use.std::crypto::hashes::rpo

    begin
        push.1.0.0.0.1000 mem_storew dropw
        push.0.1.0.0.1001 mem_storew dropw

        push.1002 # end address
        push.1000 # start address

        exec.rpo::hash_memory_words
    end
    ";

    #[rustfmt::skip]
    let even_hash: Vec<u64> = build_expected_hash(&[
        1, 0, 0, 0,
        0, 1, 0, 0,
    ]).into_iter().map(|e| e.as_int()).collect();
    build_test!(even_words, &[]).expect_stack(&even_hash);
}

#[test]
fn test_hash_odd_words() {
    // checks the hash of three words
    let odd_words = "
    use.std::crypto::hashes::rpo

    begin
        push.1.0.0.0.1000 mem_storew dropw
        push.0.1.0.0.1001 mem_storew dropw
        push.0.0.1.0.1002 mem_storew dropw

        push.1003 # end address
        push.1000 # start address

        exec.rpo::hash_memory_words
    end
    ";

    #[rustfmt::skip]
    let odd_hash: Vec<u64> = build_expected_hash(&[
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    ]).into_iter().map(|e| e.as_int()).collect();
    build_test!(odd_words, &[]).expect_stack(&odd_hash);
}

#[test]
fn test_absorb_double_words_from_memory() {
    let even_words = "
    use.std::crypto::hashes::rpo

    begin
        push.1.0.0.0.1000 mem_storew dropw
        push.0.1.0.0.1001 mem_storew dropw

        push.1002      # end address
        push.1000      # start address
        padw padw padw # hasher state
        exec.rpo::absorb_double_words_from_memory
    end
    ";

    #[rustfmt::skip]
    let mut even_hash: Vec<u64> = build_expected_perm(&[
        0, 0, 0, 0, // capacity, no padding required
        1, 0, 0, 0, // first word of the rate
        0, 1, 0, 0, // second word of the rate
    ]).into_iter().map(|e| e.as_int()).collect();

    // start and end addr
    even_hash.push(1002);
    even_hash.push(1002);

    build_test!(even_words, &[]).expect_stack(&even_hash);
}

#[test]
fn test_squeeze_digest() {
    let even_words = "
    use.std::crypto::hashes::rpo

    begin
        push.1.0.0.0.1000 mem_storew dropw
        push.0.1.0.0.1001 mem_storew dropw
        push.0.0.1.0.1002 mem_storew dropw
        push.0.0.0.1.1003 mem_storew dropw

        push.1004      # end address
        push.1000      # start address
        padw padw padw # hasher state
        exec.rpo::absorb_double_words_from_memory

        exec.rpo::squeeze_digest
    end
    ";

    #[rustfmt::skip]
    let mut even_hash: Vec<u64> = build_expected_hash(&[
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]).into_iter().map(|e| e.as_int()).collect();

    // start and end addr
    even_hash.push(1004);
    even_hash.push(1004);

    build_test!(even_words, &[]).expect_stack(&even_hash);
}

#[test]
fn test_hash_memory() {
    // hash fewer than 8 elements
    let compute_inputs_hash_5 = "
    use.std::crypto::hashes::rpo

    begin
        push.1.2.3.4.1000 mem_storew dropw
        push.5.0.0.0.1001 mem_storew dropw
        push.11

        push.5.1000

        exec.rpo::hash_memory
    end
    ";

    #[rustfmt::skip]
    let mut expected_hash: Vec<u64> = build_expected_hash(&[
        1, 2, 3, 4, 5
    ]).into_iter().map(|e| e.as_int()).collect();
    // make sure that value `11` stays unchanged
    expected_hash.push(11);
    build_test!(compute_inputs_hash_5, &[]).expect_stack(&expected_hash);

    // hash exactly 8 elements
    let compute_inputs_hash_8 = "
    use.std::crypto::hashes::rpo

    begin
        push.1.2.3.4.1000 mem_storew dropw
        push.5.6.7.8.1001 mem_storew dropw
        push.11

        push.8.1000

        exec.rpo::hash_memory
    end
    ";

    #[rustfmt::skip]
    let mut expected_hash: Vec<u64> = build_expected_hash(&[
        1, 2, 3, 4, 5, 6, 7, 8
    ]).into_iter().map(|e| e.as_int()).collect();
    // make sure that value `11` stays unchanged
    expected_hash.push(11);
    build_test!(compute_inputs_hash_8, &[]).expect_stack(&expected_hash);

    // hash more than 8 elements
    let compute_inputs_hash_15 = "
    use.std::crypto::hashes::rpo

    begin
        push.1.2.3.4.1000 mem_storew dropw
        push.5.6.7.8.1001 mem_storew dropw
        push.9.10.11.12.1002 mem_storew dropw
        push.13.14.15.0.1003 mem_storew dropw
        push.11

        push.15.1000

        exec.rpo::hash_memory
    end
    ";

    #[rustfmt::skip]
    let mut expected_hash: Vec<u64> = build_expected_hash(&[
        1, 2, 3, 4, 
        5, 6, 7, 8, 
        9, 10, 11, 12, 
        13, 14, 15
    ]).into_iter().map(|e| e.as_int()).collect();
    // make sure that value `11` stays unchanged
    expected_hash.push(11);
    build_test!(compute_inputs_hash_15, &[]).expect_stack(&expected_hash);
}

#[test]
fn test_hash_memory_fail() {
    // try to hash 0 values
    let compute_inputs_hash = "
    use.std::crypto::hashes::rpo

    begin
        push.0.1000

        exec.rpo::hash_memory
    end
    ";

    assert!(build_test!(compute_inputs_hash, &[]).execute().is_err());
}
