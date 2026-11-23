# Differential corpus coverage

Vendored non-model unit sources: 5
Vendored template fixture definitions: 159
Automatically extracted fixture definitions: 186 (16 interpreter whitespace vectors, 15 error vectors)
Schema-excluded fixture definitions: 4
Committed template-bearing corpus records: 384 (executed by both the pinned Node oracle and Java differential runner)
Policy-excluded test sources: 1 (test/e2e.test.js)

## Extracted fixtures

| Corpus id | Upstream source |
| --- | --- |
| `templates.no-template` | `upstream/vendor/test/templates.test.js:10` |
| `templates.text-nodes` | `upstream/vendor/test/templates.test.js:11` |
| `templates.boolean-literals` | `upstream/vendor/test/templates.test.js:14` |
| `templates.logical-and` | `upstream/vendor/test/templates.test.js:17` |
| `templates.logical-or` | `upstream/vendor/test/templates.test.js:18` |
| `templates.logical-not` | `upstream/vendor/test/templates.test.js:19` |
| `templates.logical-not-not` | `upstream/vendor/test/templates.test.js:20` |
| `templates.logical-and-or` | `upstream/vendor/test/templates.test.js:21` |
| `templates.logical-and-not` | `upstream/vendor/test/templates.test.js:22` |
| `templates.logical-or-not` | `upstream/vendor/test/templates.test.js:23` |
| `templates.logical-combined` | `upstream/vendor/test/templates.test.js:24` |
| `templates.if-only` | `upstream/vendor/test/templates.test.js:27` |
| `templates.if-else-only` | `upstream/vendor/test/templates.test.js:28` |
| `templates.if-elif-else` | `upstream/vendor/test/templates.test.js:29` |
| `templates.nested-statements` | `upstream/vendor/test/templates.test.js:30` |
| `templates.for-loop` | `upstream/vendor/test/templates.test.js:33` |
| `templates.for-loop-unpacking` | `upstream/vendor/test/templates.test.js:34` |
| `templates.for-loop-default` | `upstream/vendor/test/templates.test.js:35` |
| `templates.for-loop-select` | `upstream/vendor/test/templates.test.js:36` |
| `templates.for-loop-select-2` | `upstream/vendor/test/templates.test.js:37` |
| `templates.for-loop-break` | `upstream/vendor/test/templates.test.js:38` |
| `templates.for-loop-continue` | `upstream/vendor/test/templates.test.js:39` |
| `templates.for-loop-range-negative-step` | `upstream/vendor/test/templates.test.js:40` |
| `templates.for-loop-objects` | `upstream/vendor/test/templates.test.js:41` |
| `templates.variables` | `upstream/vendor/test/templates.test.js:44` |
| `templates.variables-2` | `upstream/vendor/test/templates.test.js:45` |
| `templates.variables-block` | `upstream/vendor/test/templates.test.js:46` |
| `templates.variables-unpacking` | `upstream/vendor/test/templates.test.js:47` |
| `templates.binop-expr` | `upstream/vendor/test/templates.test.js:53` |
| `templates.binop-expr-1` | `upstream/vendor/test/templates.test.js:54` |
| `templates.strings` | `upstream/vendor/test/templates.test.js:57` |
| `templates.strings-1` | `upstream/vendor/test/templates.test.js:58` |
| `templates.strings-2` | `upstream/vendor/test/templates.test.js:59` |
| `templates.strings-3` | `upstream/vendor/test/templates.test.js:60` |
| `templates.strings-4` | `upstream/vendor/test/templates.test.js:61` |
| `templates.properties` | `upstream/vendor/test/templates.test.js:67` |
| `templates.properties-1` | `upstream/vendor/test/templates.test.js:68` |
| `templates.string-methods` | `upstream/vendor/test/templates.test.js:72` |
| `templates.string-methods-2` | `upstream/vendor/test/templates.test.js:73` |
| `templates.rstrip` | `upstream/vendor/test/templates.test.js:74` |
| `templates.lstrip` | `upstream/vendor/test/templates.test.js:75` |
| `templates.split` | `upstream/vendor/test/templates.test.js:76` |
| `templates.split-2` | `upstream/vendor/test/templates.test.js:77` |
| `templates.split-3` | `upstream/vendor/test/templates.test.js:78` |
| `templates.split-4` | `upstream/vendor/test/templates.test.js:79` |
| `templates.split-5` | `upstream/vendor/test/templates.test.js:80` |
| `templates.replace` | `upstream/vendor/test/templates.test.js:81` |
| `templates.string-slicing` | `upstream/vendor/test/templates.test.js:84` |
| `templates.array-slicing` | `upstream/vendor/test/templates.test.js:87` |
| `templates.membership` | `upstream/vendor/test/templates.test.js:90` |
| `templates.membership-negation-1` | `upstream/vendor/test/templates.test.js:91` |
| `templates.membership-negation-2` | `upstream/vendor/test/templates.test.js:92` |
| `templates.membership-undefined` | `upstream/vendor/test/templates.test.js:93` |
| `templates.escaped-chars` | `upstream/vendor/test/templates.test.js:96` |
| `templates.substring-inclusion` | `upstream/vendor/test/templates.test.js:99` |
| `templates.filter-operator` | `upstream/vendor/test/templates.test.js:102` |
| `templates.filter-operator-2` | `upstream/vendor/test/templates.test.js:103` |
| `templates.filter-operator-3` | `upstream/vendor/test/templates.test.js:104` |
| `templates.filter-operator-4` | `upstream/vendor/test/templates.test.js:105` |
| `templates.filter-operator-5` | `upstream/vendor/test/templates.test.js:106` |
| `templates.filter-operator-6` | `upstream/vendor/test/templates.test.js:107` |
| `templates.filter-operator-7` | `upstream/vendor/test/templates.test.js:108` |
| `templates.filter-operator-8` | `upstream/vendor/test/templates.test.js:109` |
| `templates.filter-operator-9` | `upstream/vendor/test/templates.test.js:110` |
| `templates.filter-operator-10` | `upstream/vendor/test/templates.test.js:111` |
| `templates.filter-operator-11` | `upstream/vendor/test/templates.test.js:112` |
| `templates.filter-operator-12` | `upstream/vendor/test/templates.test.js:113` |
| `templates.filter-operator-13` | `upstream/vendor/test/templates.test.js:114` |
| `templates.filter-operator-14` | `upstream/vendor/test/templates.test.js:115` |
| `templates.filter-operator-15` | `upstream/vendor/test/templates.test.js:116` |
| `templates.filter-operator-16` | `upstream/vendor/test/templates.test.js:117` |
| `templates.filter-operator-17` | `upstream/vendor/test/templates.test.js:118` |
| `templates.filter-operator-18` | `upstream/vendor/test/templates.test.js:119` |
| `templates.filter-operator-19` | `upstream/vendor/test/templates.test.js:120` |
| `templates.tojson-separators-1` | `upstream/vendor/test/templates.test.js:122` |
| `templates.tojson-separators-2` | `upstream/vendor/test/templates.test.js:123` |
| `templates.tojson-separators-3` | `upstream/vendor/test/templates.test.js:124` |
| `templates.tojson-ensure-ascii` | `upstream/vendor/test/templates.test.js:125` |
| `templates.tojson-ensure-ascii-false` | `upstream/vendor/test/templates.test.js:126` |
| `templates.tojson-sort-keys` | `upstream/vendor/test/templates.test.js:127` |
| `templates.tojson-combined` | `upstream/vendor/test/templates.test.js:128` |
| `templates.tojson-indent-with-separators` | `upstream/vendor/test/templates.test.js:129` |
| `templates.filter-operator-dictsort-1` | `upstream/vendor/test/templates.test.js:130` |
| `templates.filter-operator-dictsort-2` | `upstream/vendor/test/templates.test.js:131` |
| `templates.filter-operator-dictsort-3` | `upstream/vendor/test/templates.test.js:132` |
| `templates.filter-operator-dictsort-4` | `upstream/vendor/test/templates.test.js:133` |
| `templates.filter-operator-dictsort-5` | `upstream/vendor/test/templates.test.js:134` |
| `templates.filter-operator-dictsort-6` | `upstream/vendor/test/templates.test.js:135` |
| `templates.sort-basic-numbers` | `upstream/vendor/test/templates.test.js:138` |
| `templates.sort-basic-strings` | `upstream/vendor/test/templates.test.js:139` |
| `templates.sort-reverse` | `upstream/vendor/test/templates.test.js:140` |
| `templates.sort-case-sensitive` | `upstream/vendor/test/templates.test.js:141` |
| `templates.sort-case-insensitive` | `upstream/vendor/test/templates.test.js:142` |
| `templates.sort-attribute-simple` | `upstream/vendor/test/templates.test.js:143` |
| `templates.sort-attribute-reverse` | `upstream/vendor/test/templates.test.js:144` |
| `templates.sort-attribute-dot-notation` | `upstream/vendor/test/templates.test.js:145` |
| `templates.sort-chained` | `upstream/vendor/test/templates.test.js:146` |
| `templates.sort-stable` | `upstream/vendor/test/templates.test.js:147` |
| `templates.sort-positional-reverse` | `upstream/vendor/test/templates.test.js:149` |
| `templates.sort-positional-reverse-case` | `upstream/vendor/test/templates.test.js:150` |
| `templates.sort-positional-all` | `upstream/vendor/test/templates.test.js:151` |
| `templates.sort-positional-mixed` | `upstream/vendor/test/templates.test.js:152` |
| `templates.sort-null-values` | `upstream/vendor/test/templates.test.js:154` |
| `templates.sort-undefined-values` | `upstream/vendor/test/templates.test.js:155` |
| `templates.sort-mixed-bool-num` | `upstream/vendor/test/templates.test.js:156` |
| `templates.filter-statements` | `upstream/vendor/test/templates.test.js:159` |
| `templates.boolean-numerical` | `upstream/vendor/test/templates.test.js:162` |
| `templates.boolean-strings` | `upstream/vendor/test/templates.test.js:163` |
| `templates.boolean-mixed` | `upstream/vendor/test/templates.test.js:164` |
| `templates.boolean-mixed-2` | `upstream/vendor/test/templates.test.js:165` |
| `templates.boolean-mixed-if` | `upstream/vendor/test/templates.test.js:166` |
| `templates.is-operator` | `upstream/vendor/test/templates.test.js:169` |
| `templates.is-operator-2` | `upstream/vendor/test/templates.test.js:170` |
| `templates.is-operator-3` | `upstream/vendor/test/templates.test.js:171` |
| `templates.is-operator-5` | `upstream/vendor/test/templates.test.js:173` |
| `templates.is-operator-6` | `upstream/vendor/test/templates.test.js:174` |
| `templates.short-circuit` | `upstream/vendor/test/templates.test.js:177` |
| `templates.short-circuit-1` | `upstream/vendor/test/templates.test.js:178` |
| `templates.namespace` | `upstream/vendor/test/templates.test.js:181` |
| `templates.namespace-1` | `upstream/vendor/test/templates.test.js:182` |
| `templates.namespace-2` | `upstream/vendor/test/templates.test.js:183` |
| `templates.object-operators` | `upstream/vendor/test/templates.test.js:186` |
| `templates.object-operators-1` | `upstream/vendor/test/templates.test.js:187` |
| `templates.object-operators-2` | `upstream/vendor/test/templates.test.js:188` |
| `templates.scope` | `upstream/vendor/test/templates.test.js:191` |
| `templates.scope-1` | `upstream/vendor/test/templates.test.js:192` |
| `templates.undefined-variables` | `upstream/vendor/test/templates.test.js:195` |
| `templates.undefined-access` | `upstream/vendor/test/templates.test.js:196` |
| `templates.null-variable` | `upstream/vendor/test/templates.test.js:199` |
| `templates.ternary-operator` | `upstream/vendor/test/templates.test.js:202` |
| `templates.ternary-operator-1` | `upstream/vendor/test/templates.test.js:203` |
| `templates.ternary-set` | `upstream/vendor/test/templates.test.js:204` |
| `templates.ternary-consecutive` | `upstream/vendor/test/templates.test.js:205` |
| `templates.ternary-shortcut` | `upstream/vendor/test/templates.test.js:206` |
| `templates.array-literals` | `upstream/vendor/test/templates.test.js:209` |
| `templates.tuple-literals` | `upstream/vendor/test/templates.test.js:212` |
| `templates.object-literals` | `upstream/vendor/test/templates.test.js:215` |
| `templates.object-literals-1` | `upstream/vendor/test/templates.test.js:216` |
| `templates.array-operators` | `upstream/vendor/test/templates.test.js:219` |
| `templates.macros` | `upstream/vendor/test/templates.test.js:222` |
| `templates.macros-1` | `upstream/vendor/test/templates.test.js:223` |
| `templates.macros-2` | `upstream/vendor/test/templates.test.js:224` |
| `templates.macros-3` | `upstream/vendor/test/templates.test.js:225` |
| `templates.macros-4` | `upstream/vendor/test/templates.test.js:226` |
| `templates.context-keywords` | `upstream/vendor/test/templates.test.js:229` |
| `templates.context-keywords-1` | `upstream/vendor/test/templates.test.js:230` |
| `templates.unpacking` | `upstream/vendor/test/templates.test.js:233` |
| `templates.whitespace-control-1` | `upstream/vendor/test/templates.test.js:236` |
| `templates.whitespace-control-2` | `upstream/vendor/test/templates.test.js:241` |
| `templates.whitespace-control-3` | `upstream/vendor/test/templates.test.js:242` |
| `templates.whitespace-control-4` | `upstream/vendor/test/templates.test.js:243` |
| `templates.whitespace-control-5` | `upstream/vendor/test/templates.test.js:244` |
| `templates.whitespace-control-6` | `upstream/vendor/test/templates.test.js:245` |
| `templates.whitespace-control-7` | `upstream/vendor/test/templates.test.js:246` |
| `templates.whitespace-control-8` | `upstream/vendor/test/templates.test.js:247` |
| `interpreter.whitespace-control-1` | `upstream/vendor/test/interpreter.test.js:21` |
| `interpreter.whitespace-control-2` | `upstream/vendor/test/interpreter.test.js:27` |
| `interpreter.whitespace-control-3` | `upstream/vendor/test/interpreter.test.js:35` |
| `interpreter.whitespace-control-4` | `upstream/vendor/test/interpreter.test.js:43` |
| `interpreter.whitespace-control-5` | `upstream/vendor/test/interpreter.test.js:54` |
| `interpreter.whitespace-control-6` | `upstream/vendor/test/interpreter.test.js:60` |
| `interpreter.whitespace-control-7` | `upstream/vendor/test/interpreter.test.js:66` |
| `interpreter.whitespace-control-8` | `upstream/vendor/test/interpreter.test.js:72` |
| `interpreter.whitespace-control-9` | `upstream/vendor/test/interpreter.test.js:78` |
| `interpreter.whitespace-control-10` | `upstream/vendor/test/interpreter.test.js:84` |
| `interpreter.whitespace-control-11` | `upstream/vendor/test/interpreter.test.js:90` |
| `interpreter.whitespace-control-12` | `upstream/vendor/test/interpreter.test.js:96` |
| `interpreter.whitespace-control-13` | `upstream/vendor/test/interpreter.test.js:104` |
| `interpreter.whitespace-control-14` | `upstream/vendor/test/interpreter.test.js:110` |
| `interpreter.whitespace-control-15` | `upstream/vendor/test/interpreter.test.js:116` |
| `interpreter.whitespace-control-16` | `upstream/vendor/test/interpreter.test.js:122` |
| `templates.error-missing-closing-curly-brace` | `upstream/vendor/test/templates.test.js:5441` |
| `templates.error-unclosed-string-literal` | `upstream/vendor/test/templates.test.js:5445` |
| `templates.error-unexpected-character` | `upstream/vendor/test/templates.test.js:5450` |
| `templates.error-invalid-quote-character` | `upstream/vendor/test/templates.test.js:5455` |
| `templates.error-unclosed-statement` | `upstream/vendor/test/templates.test.js:5462` |
| `templates.error-unclosed-expression` | `upstream/vendor/test/templates.test.js:5468` |
| `templates.error-unmatched-control-structure` | `upstream/vendor/test/templates.test.js:5474` |
| `templates.error-missing-variable-in-for-loop` | `upstream/vendor/test/templates.test.js:5480` |
| `templates.error-unclosed-parentheses-in-expression` | `upstream/vendor/test/templates.test.js:5486` |
| `templates.error-invalid-variable-name` | `upstream/vendor/test/templates.test.js:5492` |
| `templates.error-invalid-control-structure-usage` | `upstream/vendor/test/templates.test.js:5498` |
| `templates.error-undefined-function-call` | `upstream/vendor/test/templates.test.js:5506` |
| `templates.error-incorrect-function-call` | `upstream/vendor/test/templates.test.js:5514` |
| `templates.error-looping-over-non-iterable` | `upstream/vendor/test/templates.test.js:5524` |
| `templates.error-invalid-variable-assignment` | `upstream/vendor/test/templates.test.js:5534` |

## Schema exclusions

| Upstream fixture | Reason |
| --- | --- |
| `NUMBERS` | context supplies the add JavaScript function |
| `FUNCTIONS` | context supplies JavaScript functions |
| `OBJ_METHODS` | context supplies JavaScript methods |
| `IS_OPERATOR_4` | context supplies the custom testFunction JavaScript function |

## Source inventory

| Upstream source | Coverage | Evidence |
| --- | --- | --- |
| `test/templates.test.js` | differential-corpus | All serializable render and error vectors are generated; four function-context render vectors are named schema exclusions. |
| `test/interpreter.test.js` | differential-corpus | All whitespace-control vectors are generated through the direct lexer/parser/interpreter oracle path. |
| `test/format.test.js` | format-golden | src/test/resources/format/vectors.json |
| `test/utils.test.js` | direct-unit | src/test/java/se/alipsa/jmlx/jinja/internal/util/JsSliceTest.java, src/test/java/se/alipsa/jmlx/jinja/internal/util/PosixStrftimeTest.java, src/test/java/se/alipsa/jmlx/jinja/internal/runtime/InterpreterTest.java |
| `test/memory.test.js` | policy-excluded | Contains a model-derived Granite Guardian template. It cannot enter text-bearing corpus material without a fixture-specific source and license record under model-fixture-policy.md. |
| `test/e2e.test.js` | policy-excluded | Excluded by upstream-lock.json because it contains unapproved model templates and rendered output. |

Every vendored unit source and every policy-excluded test source is represented above.
## Runtime surface inventory

| Surface | Name | Differential-corpus evidence |
| --- | --- | --- |
| filter | abs | templates.filter-operator-3, self.runtime-filter-options, wp7.call-float-abs, wp7.runtime-filter-and-member-surface, wp7.runtime-inventory-core |
| filter | bool | wp7.runtime-filter-and-member-surface, wp7.runtime-inventory-core |
| filter | capitalize | templates.filter-operator-2, wp7.runtime-inventory-core |
| filter | default | templates.filter-operator-16, self.filters-basic, wp7.bare-default, wp7.default-surplus-spread, wp7.eager-default, wp7.empty-first-deferred-dereference, wp7.empty-first-default-truthiness, wp7.empty-first-all-absorbing-arguments, wp7.range-function-rendering-through-filters |
| filter | dictsort | templates.filter-operator-dictsort-1, templates.filter-operator-dictsort-2, templates.filter-operator-dictsort-3, templates.filter-operator-dictsort-4, templates.filter-operator-dictsort-5, templates.filter-operator-dictsort-6, self.runtime-object-builtin-filters, wp7.runtime-filter-and-member-surface |
| filter | first | self.runtime-sequence-filter-basics, wp7.call-first-empty-arguments, wp7.call-first-argument, wp7.call-tuple-first, wp7.empty-first-last, wp7.runtime-inventory-core, wp7.empty-first-last-defer-undefined, wp7.empty-first-macro-argument, wp7.empty-first-deferred-dereference, wp7.empty-first-container-dereference, wp7.empty-first-macro-default, wp7.empty-first-filter-default-arguments, wp7.empty-first-default-truthiness, wp7.empty-first-test-dereference, wp7.empty-first-tojson-dereference, wp7.empty-first-join-dereference, wp7.empty-first-or-dereference, wp7.empty-first-and-dereference, wp7.empty-first-unary-dereference, wp7.empty-first-equality-dereference, wp7.empty-first-member-dereference, wp7.empty-first-loop-predicate, wp7.empty-first-safe-store, wp7.empty-first-container-filter-read, wp7.empty-first-container-unique-read, wp7.empty-first-attribute-read, wp7.empty-first-absorbing-arguments, wp7.empty-first-callable-argument, wp7.empty-first-instanceof-tests, wp7.empty-first-all-absorbing-arguments, wp7.empty-first-replace-positional-count, wp7.deferred-object-member-builtin-fallback, wp7.deferred-namespace-member-builtin-fallback |
| filter | float | templates.filter-operator-14, self.filters-basic, self.filter-edge-cases, wp7.call-number-coercion, wp7.float-surplus-spread, wp7.eager-float, wp7.empty-first-all-absorbing-arguments, wp7.undefined-backed-string-coercions |
| filter | get | self.runtime-object-builtin-filters, wp7.get-keyword-bag, wp7.get-filter-missing-no-keywords, wp7.get-filter-present-no-keywords |
| filter | indent | templates.filter-operator-10, self.runtime-edge-indent-raw-truthiness, self.runtime-filter-options, wp7.eager-indent, wp7.runtime-inventory-core, wp7.empty-first-filter-default-arguments |
| filter | int | templates.filter-operator-14, self.filters-basic, wp7.call-number-coercion, wp7.int-surplus-spread, wp7.int-unknown-keywords, wp7.eager-int, wp7.filter-block-int-surplus, wp7.filter-block-eager, wp7.empty-first-all-absorbing-arguments, wp7.undefined-backed-string-coercions |
| filter | items | templates.filter-operator-6, self.items-filter, self.runtime-edge-undefined-object-key, wp7.call-items-ignores-arguments, wp7.call-items-evaluates-spread, wp7.namespace-object-surface |
| filter | join | templates.split, templates.split-2, templates.split-3, templates.split-5, self.filters-basic, self.filter-edge-cases, self.spread-call-arguments, wp7.call-tuple-join, wp7.join-surplus-spread, wp7.join-receiver-before-arguments, wp7.eager-join, wp7.empty-first-filter-default-arguments, wp7.empty-first-join-dereference, wp7.undefined-backed-string-coercions, wp7.undefined-backed-string-join-call, wp7.undefined-backed-string-join-call-positional-separator, wp7.undefined-backed-string-join-call-keyword-separator |
| filter | keys | self.runtime-object-builtin-filters, wp7.runtime-filter-and-member-surface |
| filter | last | self.runtime-sequence-filter-basics, wp7.empty-first-last, wp7.runtime-inventory-core, wp7.empty-first-last-defer-undefined |
| filter | length | templates.strings-2, templates.filter-operator, templates.filter-operator-2, templates.filter-operator-4, templates.filter-operator-5, templates.filter-operator-6, templates.filter-operator-11, templates.filter-operator-12, templates.filter-operator-17, templates.ternary-operator-1, templates.array-literals, templates.tuple-literals, templates.array-operators, self.runtime-edge-undefined-object-key, wp7.call-object-unknown, wp7.empty-first-container-unique-read, wp7.empty-first-attribute-read, wp7.empty-first-callable-argument, wp7.deferred-object-member-builtin-fallback, wp7.namespace-object-surface, wp7.tuple-array-surface, wp7.member-builtin-identity |
| filter | list | templates.filter-operator-9, templates.filter-operator-17, templates.sort-attribute-simple, templates.sort-attribute-reverse, templates.sort-attribute-dot-notation, templates.sort-chained, templates.sort-stable, templates.sort-positional-all, templates.sort-positional-mixed, wp7.empty-first-container-unique-read, wp7.empty-first-attribute-read, wp7.empty-first-all-absorbing-arguments, wp7.deferred-object-member-builtin-fallback, wp7.deferred-namespace-member-builtin-fallback, wp7.member-builtin-identity |
| filter | lower | templates.filter-operator-2, self.filters-basic |
| filter | map | templates.filter-operator-9, templates.sort-attribute-simple, templates.sort-attribute-reverse, templates.sort-attribute-dot-notation, templates.sort-chained, templates.sort-stable, templates.sort-positional-all, templates.sort-positional-mixed, self.runtime-filter-options, wp7.bare-map, wp7.call-map-no-attribute, wp7.call-map-positional-attribute, wp7.eager-map, wp7.runtime-inventory-core, wp7.empty-first-attribute-read, wp7.empty-first-all-absorbing-arguments, wp7.namespace-map-unknown-filter |
| filter | rejectattr | templates.filter-operator-11, templates.filter-operator-12, wp7.bare-rejectattr, wp7.empty-first-attribute-read, wp7.namespace-rejectattr-unknown-filter |
| filter | replace | templates.filter-operator-15, self.runtime-edge-replace-code-point, wp7.bare-replace, wp7.replace-filter-one-positional, wp7.replace-filter-count, wp7.filter-block-replace-one, wp7.eager-replace, wp7.replace-filter-zero-positional, wp7.replace-filter-keyword-only, wp7.replace-filter-four-positional, wp7.runtime-inventory-core, wp7.empty-first-all-absorbing-arguments, wp7.empty-first-replace-positional-count, wp7.namespace-replace-unknown-filter |
| filter | reverse | self.runtime-sequence-filter-basics, wp7.runtime-filter-and-member-surface, wp7.runtime-inventory-core, wp7.tuple-array-surface |
| filter | safe | templates.filter-operator-18, self.safe-and-boolean-tests, wp7.call-safe-string, wp7.call-safe-skips-arguments, wp7.filter-block-safe-call, wp7.empty-first-safe-store, wp7.range-function-rendering-through-filters |
| filter | selectattr | templates.for-loop-select-2, templates.filter-operator-4, templates.filter-operator-5, wp7.bare-selectattr, wp7.call-selectattr-empty-array, wp7.call-selectattr-nonobject-array, wp7.call-selectattr-string, wp7.selectattr-computed-argument, wp7.selectattr-spread-argument, wp7.selectattr-keyword-argument, wp7.selectattr-four-literals, wp7.empty-first-attribute-read, wp7.namespace-selectattr-unknown-filter |
| filter | sort | templates.filter-operator, templates.sort-basic-numbers, templates.sort-basic-strings, templates.sort-reverse, templates.sort-case-sensitive, templates.sort-case-insensitive, templates.sort-attribute-simple, templates.sort-attribute-reverse, templates.sort-attribute-dot-notation, templates.sort-chained, templates.sort-stable, templates.sort-positional-reverse, templates.sort-positional-reverse-case, templates.sort-positional-all, templates.sort-positional-mixed, templates.sort-null-values, templates.sort-undefined-values, templates.sort-mixed-bool-num, self.runtime-filter-options, wp7.eager-sort, wp7.runtime-inventory-core, wp7.sort-case-sensitive-undefined, wp7.tuple-array-surface |
| filter | string | templates.filter-operator-13, templates.filter-operator-19, wp7.namespace-object-surface, wp7.tuple-array-surface |
| filter | title | templates.filter-operator-2, self.runtime-edge-title-ascii-boundary, wp7.runtime-inventory-core |
| filter | tojson | templates.split-4, templates.filter-operator-7, templates.filter-operator-8, templates.filter-operator-9, templates.tojson-separators-1, templates.tojson-separators-2, templates.tojson-separators-3, templates.tojson-ensure-ascii, templates.tojson-ensure-ascii-false, templates.tojson-sort-keys, templates.tojson-combined, templates.tojson-indent-with-separators, self.filters-basic, self.filter-edge-cases, self.tojson-ensure-ascii, self.tojson-negative-indent, self.tojson-sort-undefined-key, self.tojson-negative-indent-scalars, self.tojson-nan-indent, self.tojson-undefined-slots-and-separators, wp7.eager-tojson, wp7.sort-case-sensitive-undefined, wp7.empty-first-container-dereference, wp7.empty-first-tojson-dereference, wp7.empty-first-attribute-read, wp7.empty-first-absorbing-arguments, wp7.empty-first-all-absorbing-arguments, wp7.deferred-object-member-builtin-fallback, wp7.deferred-namespace-member-builtin-fallback |
| filter | trim | templates.filter-operator-2, self.filters-basic, self.filter-edge-cases |
| filter | unique | templates.filter-operator-17, wp7.runtime-inventory-core, wp7.empty-first-container-unique-read, wp7.member-builtin-identity |
| filter | upper | templates.filter-operator-2, templates.filter-statements, self.filter-block, wp7.macro-break-crosses-filter-block |
| filter | values | self.runtime-object-builtin-filters, wp7.runtime-filter-and-member-surface |
| test | boolean | templates.is-operator-2, self.tests-basic |
| test | callable | self.runtime-sequence-filter-basics, wp7.empty-first-instanceof-tests |
| test | defined | templates.membership-undefined, templates.is-operator, templates.object-operators-1, templates.null-variable, self.tests-basic, wp7.empty-first-last-defer-undefined, wp7.undefined-backed-string-tests, wp7.empty-first-test-dereference, wp7.empty-first-safe-store |
| test | eq | wp7.runtime-test-eq-alias-requires-arguments |
| test | equalto | wp7.runtime-test-aliases-require-arguments |
| test | even | templates.is-operator-3, wp7.runtime-inventory-core |
| test | false | templates.is-operator-2, self.safe-and-boolean-tests |
| test | integer | templates.is-operator-3, wp7.runtime-inventory-core, wp7.empty-first-instanceof-tests |
| test | iterable | self.tests-basic, wp7.undefined-backed-string-tests |
| test | lower | templates.is-operator-5, wp7.lower-undefined-backed-string, wp7.runtime-inventory-core |
| test | mapping | templates.is-operator-6, self.mapping-test, wp7.empty-first-instanceof-tests |
| test | none | templates.object-operators-1, templates.null-variable, self.filters-basic, self.tests-basic |
| test | number | templates.is-operator-3, self.tests-basic, wp7.empty-first-instanceof-tests |
| test | odd | templates.is-operator-3, wp7.runtime-inventory-core |
| test | sequence | self.tests-basic, self.mapping-test, wp7.undefined-backed-string-tests, wp7.empty-first-instanceof-tests |
| test | string | self.tests-basic, wp7.undefined-backed-string-tests |
| test | true | templates.is-operator-2, self.safe-and-boolean-tests |
| test | undefined | self.tests-basic, wp7.undefined-backed-string-tests |
| test | upper | templates.is-operator-5, wp7.runtime-inventory-core, wp7.upper-undefined-backed-string |
| string member | capitalize | templates.filter-operator-2, wp7.runtime-inventory-core |
| string member | endswith | self.startswith-endswith, self.endswith-type, wp7.undefined-backed-string-coercions |
| string member | length | templates.strings-2, templates.filter-operator, templates.filter-operator-2, templates.filter-operator-4, templates.filter-operator-5, templates.filter-operator-6, templates.filter-operator-11, templates.filter-operator-12, templates.filter-operator-17, templates.ternary-operator-1, templates.array-literals, templates.tuple-literals, templates.array-operators, self.runtime-edge-undefined-object-key, wp7.call-object-unknown, wp7.empty-first-container-unique-read, wp7.empty-first-attribute-read, wp7.empty-first-callable-argument, wp7.deferred-object-member-builtin-fallback, wp7.namespace-object-surface, wp7.tuple-array-surface, wp7.member-builtin-identity |
| string member | lower | templates.string-methods, templates.filter-operator-2, self.filters-basic |
| string member | lstrip | templates.lstrip |
| string member | replace | templates.replace, templates.filter-operator-15, self.runtime-edge-replace-code-point, wp7.bare-replace, wp7.replace-filter-one-positional, wp7.replace-member-keyword-bag, wp7.replace-filter-count, wp7.eager-replace, wp7.replace-filter-zero-positional, wp7.replace-filter-keyword-only, wp7.replace-filter-four-positional, wp7.runtime-inventory-core, wp7.empty-first-all-absorbing-arguments, wp7.empty-first-replace-positional-count, wp7.namespace-replace-unknown-filter, wp7.undefined-backed-string-replace |
| string member | rstrip | templates.rstrip |
| string member | split | templates.variables-2, templates.split, templates.split-2, templates.split-3, templates.split-4, templates.split-5, self.runtime-edge-ecma-whitespace-split, self.runtime-filter-options, self.runtime-string-negative-maxsplit, wp7.split-maxsplit-keyword, wp7.split-sep-keyword, wp7.runtime-inventory-core, wp7.empty-first-all-absorbing-arguments |
| string member | startswith | self.startswith-endswith, self.startswith-arity, self.startswith-tuple-element-type, wp7.member-builtin-identity, wp7.undefined-backed-string-coercions |
| string member | strip | templates.string-methods, wp7.runtime-filter-and-member-surface |
| string member | title | templates.string-methods-2, templates.filter-operator-2, self.runtime-edge-title-ascii-boundary, wp7.runtime-inventory-core |
| string member | upper | templates.string-methods, templates.filter-operator-2, wp7.member-builtin-identity |
| object member | dictsort | templates.filter-operator-dictsort-1, templates.filter-operator-dictsort-2, templates.filter-operator-dictsort-3, templates.filter-operator-dictsort-4, templates.filter-operator-dictsort-5, templates.filter-operator-dictsort-6, self.runtime-edge-undefined-object-key, self.runtime-object-builtin-filters, wp7.dictsort-undefined-key-with-peer, wp7.runtime-filter-and-member-surface, wp7.runtime-inventory-core, wp7.empty-first-absorbing-arguments, wp7.namespace-object-surface |
| object member | get | templates.object-operators-1, self.runtime-object-builtin-filters, wp7.get-keyword-bag, wp7.get-filter-missing-no-keywords, wp7.get-filter-present-no-keywords, wp7.get-member-keyword-only, wp7.runtime-inventory-core, wp7.empty-first-absorbing-arguments, wp7.deferred-object-member-builtin-fallback, wp7.namespace-object-surface |
| object member | items | templates.filter-operator-6, templates.object-operators-2, self.items-filter, self.runtime-edge-undefined-object-key, wp7.call-items-ignores-arguments, wp7.call-items-evaluates-spread, wp7.deferred-object-member-builtin-fallback, wp7.namespace-object-surface, wp7.member-builtin-identity |
| object member | keys | self.runtime-edge-undefined-object-key, self.runtime-object-builtin-filters, wp7.runtime-filter-and-member-surface, wp7.runtime-inventory-core, wp7.deferred-object-member-builtin-fallback, wp7.namespace-object-surface, wp7.deferred-namespace-member-builtin-fallback, wp7.member-builtin-identity |
| object member | values | self.runtime-object-builtin-filters, wp7.runtime-filter-and-member-surface, wp7.runtime-inventory-core, wp7.namespace-object-surface |
| array member | length | templates.strings-2, templates.filter-operator, templates.filter-operator-2, templates.filter-operator-4, templates.filter-operator-5, templates.filter-operator-6, templates.filter-operator-11, templates.filter-operator-12, templates.filter-operator-17, templates.ternary-operator-1, templates.array-literals, templates.tuple-literals, templates.array-operators, self.runtime-edge-undefined-object-key, wp7.call-object-unknown, wp7.empty-first-container-unique-read, wp7.empty-first-attribute-read, wp7.empty-first-callable-argument, wp7.deferred-object-member-builtin-fallback, wp7.namespace-object-surface, wp7.tuple-array-surface, wp7.member-builtin-identity |
| global | False | templates.boolean-literals, templates.ternary-consecutive, self.tojson-ensure-ascii, wp7.runtime-global-aliases |
| global | None | templates.sort-null-values, wp7.runtime-global-aliases |
| global | True | templates.boolean-literals, templates.filter-operator-10, templates.ternary-set, interpreter.whitespace-control-1, interpreter.whitespace-control-2, interpreter.whitespace-control-3, interpreter.whitespace-control-4, self.tojson-ensure-ascii, wp7.runtime-global-aliases |
| global | false | templates.boolean-literals, templates.logical-and, templates.logical-or, templates.logical-not, templates.logical-not-not, templates.logical-and-or, templates.logical-and-not, templates.logical-or-not, templates.membership, templates.membership-negation-1, templates.membership-negation-2, templates.filter-operator-14, templates.filter-operator-16, templates.tojson-ensure-ascii-false, templates.sort-case-insensitive, templates.sort-positional-reverse-case, templates.sort-positional-all, templates.sort-mixed-bool-num, templates.boolean-mixed, templates.boolean-mixed-2, templates.is-operator-2, templates.short-circuit, templates.namespace-1, templates.namespace-2, templates.scope, templates.scope-1, templates.ternary-operator, templates.ternary-shortcut, self.filters-basic, self.ternary-expressions, self.safe-and-boolean-tests, self.runtime-edge-indent-raw-truthiness, wp7.eager-sort, wp7.eager-indent, wp7.eager-default, wp7.runtime-global-aliases |
| global | none | templates.split-5, templates.object-operators-1, templates.null-variable, self.filters-basic, self.tests-basic, self.filter-edge-cases, self.ternary-expressions, self.loose-nil-equality, self.mapping-test, self.tojson-negative-indent-scalars, self.safe-and-boolean-tests, self.runtime-filter-options, self.runtime-string-negative-maxsplit, wp7.bare-null-unknown, wp7.eager-sort, wp7.runtime-global-aliases, wp7.empty-first-equality-dereference |
| global | raise_exception | templates.short-circuit, templates.short-circuit-1, self.ternary-expressions, wp7.call-safe-skips-arguments, wp7.eager-int, wp7.eager-tojson, wp7.join-receiver-before-arguments, wp7.positional-after-keyword, wp7.filter-block-eager, wp7.eager-float, wp7.eager-sort, wp7.eager-join, wp7.eager-indent, wp7.eager-map, wp7.eager-default, wp7.eager-replace, wp7.raise-exception-function-bare-rendering, wp7.raise-exception-function-message, wp7.raise-exception-nested-function-message |
| global | range | templates.for-loop-range-negative-step, self.context-global-collision, self.ternary-expressions, self.spread-invalid-receiver, self.runtime-sequence-filter-basics, wp7.bare-function-unknown, wp7.runtime-global-aliases, wp7.empty-first-callable-argument, wp7.range-function-rendering-through-filters, wp7.range-function-bare-rendering, wp7.raise-exception-function-message, wp7.raise-exception-nested-function-message |
| global | strftime_now | self.strftime-month-names, self.strftime-default-instant-zone, self.strftime-unknown-directive, self.strftime-terminal-percent, self.strftime-non-ascii-literal, self.strftime-date-boundary, self.strftime-zero-padding-utc, self.strftime-literal-vector, wp7.strftime-missing-format, wp7.strftime-keyword-format, wp7.strftime-namespace-format, wp7.strftime-now-function-bare-rendering |
| global | true | templates.boolean-literals, templates.logical-and, templates.logical-or, templates.logical-not, templates.logical-not-not, templates.logical-and-or, templates.logical-and-not, templates.logical-or-not, templates.binop-expr-1, templates.strings-3, templates.membership, templates.membership-negation-1, templates.membership-negation-2, templates.filter-operator-7, templates.filter-operator-14, templates.filter-operator-16, templates.tojson-ensure-ascii, templates.tojson-sort-keys, templates.tojson-combined, templates.filter-operator-dictsort-3, templates.filter-operator-dictsort-5, templates.filter-operator-dictsort-6, templates.sort-reverse, templates.sort-case-sensitive, templates.sort-attribute-reverse, templates.sort-chained, templates.sort-positional-reverse, templates.sort-positional-reverse-case, templates.sort-positional-mixed, templates.sort-mixed-bool-num, templates.boolean-mixed, templates.boolean-mixed-2, templates.boolean-mixed-if, templates.is-operator-2, templates.short-circuit-1, templates.scope, templates.scope-1, templates.ternary-operator, templates.ternary-operator-1, templates.ternary-shortcut, templates.array-literals, templates.whitespace-control-3, interpreter.whitespace-control-13, interpreter.whitespace-control-14, interpreter.whitespace-control-15, interpreter.whitespace-control-16, templates.error-incorrect-function-call, self.if-set-persists, self.filters-basic, self.tests-basic, self.ternary-expressions, self.tojson-sort-undefined-key, self.safe-and-boolean-tests, wp7.bare-boolean-unknown, wp7.default-surplus-spread, wp7.runtime-filter-and-member-surface, wp7.runtime-inventory-core, wp7.sort-case-sensitive-undefined, wp7.runtime-global-aliases, wp7.empty-first-default-truthiness |

## Error-family inventory

| Category | Differential-corpus evidence |
| --- | --- |
| ARITY | self.startswith-arity, wp7.empty-first-macro-argument |
| EXPLICIT_RAISE | wp7.eager-int, wp7.eager-tojson, wp7.filter-block-eager, wp7.eager-float, wp7.eager-sort, wp7.eager-join, wp7.eager-indent, wp7.eager-map, wp7.eager-default, wp7.eager-replace, wp7.raise-exception-function-message, wp7.raise-exception-nested-function-message |
| SYNTAX | templates.error-missing-closing-curly-brace, templates.error-unclosed-string-literal, templates.error-unexpected-character, templates.error-invalid-quote-character, templates.error-unclosed-statement, templates.error-unclosed-expression, templates.error-unmatched-control-structure, templates.error-missing-variable-in-for-loop, templates.error-unclosed-parentheses-in-expression, templates.error-invalid-variable-name, templates.error-invalid-control-structure-usage, templates.error-invalid-variable-assignment, wp7.positional-after-keyword, wp7.positional-after-keyword-macro, wp7.macro-bare-break, wp7.macro-bare-continue, wp7.call-block-bare-break, wp7.call-block-bare-continue, wp7.parser-missing-statement-name |
| TYPE | templates.error-undefined-function-call, templates.error-incorrect-function-call, templates.error-looping-over-non-iterable, self.object-key-type, self.tuple-unpack-loop-item, self.spread-invalid-receiver, self.endswith-type, self.startswith-tuple-element-type, wp7.call-safe-string, wp7.call-safe-skips-arguments, wp7.call-items-evaluates-spread, wp7.call-first-empty-arguments, wp7.call-first-argument, wp7.call-tuple-first, wp7.bare-numeric-unknown, wp7.bare-float-unknown, wp7.bare-boolean-unknown, wp7.bare-null-unknown, wp7.bare-undefined-unknown, wp7.bare-function-unknown, wp7.bare-tuple-unknown, wp7.bare-object-unknown, wp7.call-object-unknown, wp7.call-float-abs, wp7.bare-selectattr, wp7.bare-rejectattr, wp7.bare-map, wp7.bare-replace, wp7.bare-default, wp7.call-map-no-attribute, wp7.call-map-positional-attribute, wp7.call-selectattr-nonobject-array, wp7.call-selectattr-string, wp7.filter-block-safe-call, wp7.join-receiver-before-arguments, wp7.selectattr-computed-argument, wp7.selectattr-spread-argument, wp7.selectattr-keyword-argument, wp7.replace-filter-one-positional, wp7.replace-member-keyword-bag, wp7.get-keyword-bag, wp7.split-maxsplit-keyword, wp7.split-sep-keyword, wp7.filter-block-replace-one, wp7.get-member-keyword-only, wp7.replace-filter-zero-positional, wp7.replace-filter-keyword-only, wp7.empty-first-last, wp7.dictsort-undefined-key-with-peer, wp7.lower-undefined-backed-string, wp7.strftime-missing-format, wp7.strftime-keyword-format, wp7.strftime-namespace-format, wp7.runtime-test-aliases-require-arguments, wp7.runtime-test-eq-alias-requires-arguments, wp7.upper-undefined-backed-string, wp7.empty-first-deferred-dereference, wp7.empty-first-container-dereference, wp7.empty-first-default-truthiness, wp7.empty-first-test-dereference, wp7.empty-first-tojson-dereference, wp7.empty-first-join-dereference, wp7.empty-first-or-dereference, wp7.empty-first-and-dereference, wp7.empty-first-unary-dereference, wp7.empty-first-equality-dereference, wp7.empty-first-member-dereference, wp7.empty-first-loop-predicate, wp7.empty-first-container-filter-read, wp7.empty-first-container-unique-read, wp7.empty-first-callable-argument, wp7.undefined-backed-string-tostring-add, wp7.undefined-backed-string-tostring-concat, wp7.empty-first-replace-positional-count, wp7.namespace-map-unknown-filter, wp7.namespace-selectattr-unknown-filter, wp7.namespace-rejectattr-unknown-filter, wp7.namespace-replace-unknown-filter, wp7.undefined-backed-string-join-call, wp7.undefined-backed-string-join-call-positional-separator, wp7.undefined-backed-string-join-call-keyword-separator, wp7.undefined-backed-string-membership, wp7.undefined-backed-string-index, wp7.undefined-backed-string-slice, wp7.undefined-backed-string-replace |
| UNDEFINED_OR_ACCESS | Excluded: unknown variables and missing members render as undefined in the pinned public Template API. |
| VALUE | self.context-global-collision, self.tojson-negative-indent |

