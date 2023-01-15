package com.sewerynkamil;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import java.io.File;

public class SentimentClassifier {
    public static void main(String[] args) throws Exception {
        File gModel = new File("data/GoogleNews-vectors-negative300.bin.gz");
        Word2Vec wordVectors = WordVectorSerializer.readWord2VecModel(gModel);

        TwitterIterator train = new TwitterIterator("data", wordVectors, 128, 35, true);
        TwitterIterator test = new TwitterIterator("data", wordVectors, 128, 35, false);

        int inputNeurons = train.inputColumns();
        int outputNeurons = train.totalOutcomes();
        int nEpochs = 5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new RmsProp(0.001))
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .list()
                .layer(new LSTM.Builder().nIn(inputNeurons).nOut(200).activation(Activation.TANH).build())
                .layer(new LSTM.Builder().nIn(200).nOut(200).activation(Activation.TANH).build())
                .layer(new LSTM.Builder().nIn(200).nOut(200).activation(Activation.TANH).build())
                .layer(new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(outputNeurons).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        System.out.println("Started training...");
        net.setListeners(new ScoreIterationListener(1), new EvaluativeListener(test, 1, InvocationType.EPOCH_END));
        net.fit(train, nEpochs);

        System.out.println("Evaluating...");
        Evaluation eval = net.evaluate(test);
        System.out.println(eval.stats());
    }
}