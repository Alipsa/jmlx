"""Required native suite inventory shared by CI and local release verification."""

suites = {
    'jmlx-ffi': ['NativeLoaderMissingMetallibTest', 'NativeLoaderExtractionFallbackTest'],
    'jmlx-core': ['DecoderAttentionTest', 'MoeMlpTest', 'SwitchGluTest', 'MoeRoutingProbeTest', 'RopeFreqsProbeTest',
                  'KVCacheTest', 'KVCacheMemoryPlateauTest', 'RecurrentCacheTest',
                  'GatedDeltaRuleTest', 'GatedDeltaNetTest', 'GatedDeltaMemoryTest',
                  'HybridDecoderBlockTest',
                  'KVCacheQuantizationProbeTest', 'PackedAttentionProbeTest', 'AttentionMaskTest',
                  'GroupedQueryAttentionTest', 'MultiHeadAttentionTest',
                  'BatchAttentionMaskTest', 'BatchRopeProbeTest', 'StridedViewSafetyTest',
                  'Phase71OracleTest', 'Phase71DefaultModeTest', 'Phase71ContractsTest', 'EncoderAttentionTest',
                  'MLXMemoryLeakTest', 'MLXIOSelectionScopeTest', 'CategoricalTempScopeTest',
                  'MultiAxisRopeTest', 'DecoderRotationSeamTest', 'SingleRowDecodeEquivalenceTest'],
    'jmlx-models': ['DecoderRefactorGoldenTest', 'MistralModelTest', 'GemmaModelTest',
                    'Phi3ModelTest', 'MixtralModelTest', 'Llama31ModelTest', 'Qwen3ModelTest',
                    'QuantizedCheckpointProbeTest', 'QuantizedDecoderTest', 'DecoderCacheLifecycleTest',
                    'Qwen35WeightsTest', 'Qwen35TextGoldenTest', 'Qwen35PackedGoldenTest',
                    'Qwen35VisionGoldenTest', 'Qwen35GenerationTest', 'Qwen35MemoryTest',
                    'Qwen35DefaultPrecisionTest',
                    'DecoderCacheCapacityTest', 'DecoderSlidingWindowTest',
                    'DecoderStackContractTest', 'DecoderForwardPoisoningTest',
                    'QuantizedKvFeasibilityProbeTest',
                    'BatchGenerationSchedulerTest', 'BatchSchedulerFailureTest',
                    'BatchSchedulerLifecycleTest', 'BatchSchedulerMemoryTest',
                    'SchedulerOperationTest', 'SchedulerInferenceClientTest',
                    'BatchStepEquivalenceTest', 'BertGoldenTest', 'T5GoldenTest', 'Seq2SeqMemoryTest', 'Seq2SeqGenerationTest', 'EncoderDefaultPrecisionTest',
                    'ModalityRejectionTest', 'Idefics3VisionTowerSmokeTest',
                    'SmolVlmGoldenTest', 'SmolVlmGenerationTest', 'SmolVlmDefaultPrecisionTest', 'VisionLanguageMemoryTest',
                    'DecoderSeamTest', 'VisualFeatureInjectionTest',
                    'Qwen3VlVisionTowerGoldenTest', 'Qwen3VlVisionPeakTest',
                    'Qwen3VlVisionTowerValidationTest',
                    'Qwen3VlModelLoadTest', 'Qwen3VlModelAffineLoadTest',
                    'Qwen3VlGenerationTest', 'Qwen3VlMemoryTest', 'Qwen3VlCacheEquivalenceTest',
                    'Qwen3VlGoldenTest', 'Qwen3VlDefaultPrecisionTest',
                    'Phi3VModelTest', 'Phi3VGoldenTest', 'Phi3VDefaultPrecisionTest',
                     'Phi3VRopeReferenceTest', 'Phi3VGenerationTest', 'Phi3VMemoryTest',
                      'BaselineAdaptersTest', 'BaselineWorkerProtocolTest',
                      'BaselineWorkerProcessTest'],
}
# Golden references run in their own full-float32 JVM. Batch equivalence must
# execute in both precision modes. Probe (b) remains evidence only.
float32_suites = ['DecoderRefactorGoldenTest', 'MistralModelTest', 'GemmaModelTest',
                  'Phi3ModelTest', 'MixtralModelTest', 'Llama31ModelTest',
                  'DecoderSlidingWindowTest', 'BertGoldenTest', 'T5GoldenTest', 'SmolVlmGoldenTest',
                  'Qwen3VlVisionTowerGoldenTest',
                  'Qwen3VlTextOnlyEquivalenceTest', 'Qwen3VlModelFloatGoldenTest',
                  'Qwen3VlGoldenTest', 'Phi3VGoldenTest', 'Qwen35WeightsTest',
                  'Qwen35TextGoldenTest', 'Qwen35PackedGoldenTest', 'Qwen35VisionGoldenTest']
model_results = {name: ['float32GoldenTest'] for name in float32_suites}
model_results['BatchStepEquivalenceTest'] = ['test', 'float32GoldenTest']
model_results['Qwen3VlCacheEquivalenceTest'] = ['test', 'float32GoldenTest']
model_results['DecoderSeamTest'] = ['test', 'float32GoldenTest']
model_results['VisualFeatureInjectionTest'] = ['test', 'float32GoldenTest']
model_results['WorkerStreamProbeTest'] = ['threadProbeTest']
core_results = {name: ['test', 'float32GoldenTest'] for name in
                ['DecoderAttentionTest', 'MultiHeadAttentionTest', 'MoeMlpTest', 'SwitchGluTest',
                 'MultiAxisRopeTest', 'DecoderRotationSeamTest']}
core_results['Phase71OracleTest'] = ['float32GoldenTest']
core_results['EncoderAttentionTest'] = ['float32GoldenTest']
core_results['GatedDeltaRuleTest'] = ['float32GoldenTest']
core_results['GatedDeltaNetTest'] = ['float32GoldenTest']
core_results['HybridDecoderBlockTest'] = ['float32GoldenTest']
results_dirs = {'jmlx-models': model_results, 'jmlx-core': core_results}
results_dirs['jmlx-ffi'] = {
    'NativeLoaderMissingMetallibTest': ['loaderGuardTest'],
    'NativeLoaderExtractionFallbackTest': ['extractionFallbackTest'],
}
suites['jmlx-models'].append('WorkerStreamProbeTest')
