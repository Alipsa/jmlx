package se.alipsa.jmlx.models;

record Seq2SeqMetadata(String modelType, int vocabSize, int numHiddenLayers, int numEncoderLayers)
    implements ModelMetadata {}
