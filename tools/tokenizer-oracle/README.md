# Tokenizer oracle

This directory pins the official Hugging Face Python binding over the Rust tokenizers
implementation. It loads only committed local fixture files; generation and verification never
download tokenizers or models from the Hub.

Install with ./tools/tokenizer-oracle/install.sh. Gradle's verifyTokenizerOracle checks the
interpreter, platform, and package pin. generateTokenizerOracleFixtures is the only supported way
to rewrite expected JSON; verifyTokenizerOracleFixtures is read-only.
