# Local tokenizer-config example

Compile and run the example with Java 21 through the maintained Gradle verification tasks:

    ./gradlew :jmlx-jinja:compileTokenizerConfigExample
    ./gradlew :jmlx-jinja:verifyTokenizerConfigExample

The example extracts a string chat_template, parses it once, and renders a prompt. Real
applications should use their chosen JSON parser to read the config; the small extractor keeps this
dependency-free example runnable with only the JDK and jmlx-jinja.
