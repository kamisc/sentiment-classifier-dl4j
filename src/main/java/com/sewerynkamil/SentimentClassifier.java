package com.sewerynkamil;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.io.File;

public class SentimentClassifier {
    public static void main(String[] args) {
        File gModel = new File("data/GoogleNews-vectors-negative300.bin.gz");
        Word2Vec wordVectors = WordVectorSerializer.readWord2VecModel(gModel);
    }
}