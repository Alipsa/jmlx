# Local tokenizer-config example

Compile with Java 21 against a built jmlx-jinja JAR:

    javac --release 21 --module-path ../../build/libs/jmlx-jinja-0.6.0-SNAPSHOT.jar --add-modules se.alipsa.jmlx.jinja -d . TokenizerConfigExample.java
    java --module-path ../../build/libs/jmlx-jinja-0.6.0-SNAPSHOT.jar --add-modules se.alipsa.jmlx.jinja -cp . example.TokenizerConfigExample

It reads this local tokenizer_config.json, extracts its string chat_template, parses it once, and
renders a prompt. Real applications should use their chosen JSON parser to read the config; the
small extractor keeps this dependency-free example runnable with only the JDK and jmlx-jinja.
