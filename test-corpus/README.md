# Shared Test Corpus

This folder contains `.lino` regression inputs that are shared by the
JavaScript and Rust test suites.

Each corpus file is listed in `expected.lino` with the query results both
implementations must produce. Tests in `js/tests/shared-test-corpus.test.mjs`
and `rust/tests/shared_test_corpus.rs` walk `test-corpus/*.lino`, excluding
`expected.lino`, and compare runtime output against that single contract.
