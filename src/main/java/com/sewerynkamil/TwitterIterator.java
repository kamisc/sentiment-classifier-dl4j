package com.sewerynkamil;

import org.apache.commons.lang.ArrayUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.util.*;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class TwitterIterator implements DataSetIterator {
    private final Word2Vec wordVectors;
    private final int batchSize;
    private final int vectorSize;
    private final int truncateLength;
    private final Map<Integer, List<String>> categoryData = new HashMap<>();
    private final String filename;
    private final TokenizerFactory tokenizerFactory;
    private final String[] categories = new String[]{"neutral", "positive", "negative"};
    private final int textId = 1;
    private final int categoryIdx;

    private int cursor = 0;
    private int totalTweets = 0;
    private float[] proportions = new float[]{0.0f, 0.0f, 0.0f};
    private int[] cursors = new int[3];
    private int maxLength;

    public TwitterIterator(
            String dataDirectory,
            Word2Vec wordVectors,
            int batchSize,
            int truncateLength,
            boolean train) throws Exception {
        this.batchSize = batchSize;
        this.wordVectors = wordVectors;
        this.vectorSize = this.wordVectors.getWordVector(this.wordVectors.vocab().wordAtIndex(0)).length;
        this.truncateLength = truncateLength;
        this.filename = "data/" + (train ? "train" : "test") + ".csv";
        this.categoryIdx = train ? 3 : 2;

        this.tokenizerFactory = new DefaultTokenizerFactory();
        this.tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        this.populateData();
    }

    private void populateData() throws IOException, InterruptedException {
        RecordReader rr = new CSVRecordReader(1, ',', '"');
        rr.initialize(new FileSplit(new File(this.filename)));
        while (rr.hasNext()) {
            List<Writable> record = rr.next();
            String text = record.get(this.textId).toString();
            int category = ArrayUtils.indexOf(this.categories, record.get(this.categoryIdx).toString());
            this.categoryData.computeIfAbsent(category, k -> new ArrayList<>());
            this.categoryData.get(category).add(text);
            this.totalTweets++;
        }

        for (int i = 0; i < 3; i++) {
            this.proportions[i] = (float) this.categoryData.get(i).size() / (float) this.totalTweets;
            this.cursors[i] = 0;
        }
    }

    @Override
    public DataSet next(int i) {
        if (this.cursor >= this.totalTweets) throw new NoSuchElementException();
        return nextDataSet(i);
    }

    private DataSet nextDataSet(int num) {
        List<String> tweets = new ArrayList<>(num);
        List<Integer> category = new ArrayList<>(num);

        int count = 0;
        for (int i = 0; i < 3 && count < num; i++) {
            int catSize = Math.round(this.proportions[i] * num);
            while (catSize > 0 && this.cursors[i] < this.categoryData.get(i).size() && count < num) {
                tweets.add(this.categoryData.get(i).get(this.cursors[i]));
                category.add(i);
                this.cursors[i]++;
                this.cursor++;
                catSize--;
                count++;
            }
        }

        long seed = System.nanoTime();
        Collections.shuffle(tweets, new Random(seed));
        Collections.shuffle(category, new Random(seed));

        List<List<String>> allTokens = new ArrayList<>(tweets.size());
        this.maxLength = 0;
        for (String tweet : tweets) {
            List<String> tokens = this.tokenizerFactory.create(tweet).getTokens();
            List<String> knownTokens = new ArrayList<>();
            for (String token : tokens) {
                if (this.wordVectors.hasWord(token)) {
                    knownTokens.add(token);
                }
            }
            allTokens.add(knownTokens);
            this.maxLength = Math.max(maxLength, knownTokens.size());
        }

        this.maxLength = Math.min(this.maxLength, this.truncateLength);

        INDArray features = Nd4j.create(tweets.size(), this.vectorSize, this.maxLength);
        INDArray labels = Nd4j.create(tweets.size(), 3, this.maxLength);

        INDArray featuresMask = Nd4j.zeros(tweets.size(), this.maxLength);
        INDArray labelsMask = Nd4j.zeros(tweets.size(), this.maxLength);

        for (int i = 0; i < tweets.size(); i++) {
            List<String> tokens = allTokens.get(i);
            for (int j = 0; j < tokens.size() && j < this.maxLength; j++) {
                String token = tokens.get(j);
                INDArray vector = this.wordVectors.getWordVectorMatrix(token);
                features.put(new INDArrayIndex[]{point(i), all(), point(j)}, vector);
                featuresMask.putScalar(new int[]{i, j}, 1.0);
            }
            int idx = category.get(i);
            int lastIdx = Math.min(tokens.size(), this.maxLength);
            labels.putScalar(new int[]{i, idx, lastIdx - 1}, 1.0);
            labelsMask.putScalar(new int[]{i, lastIdx - 1}, 1.0);
        }

        return new DataSet(features, labels, featuresMask, labelsMask);
    }

    @Override
    public int inputColumns() {
        return this.vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return 3;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        this.cursor = 0;
        this.cursors = new int[]{0, 0, 0};
    }

    @Override
    public int batch() {
        return this.batchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return Arrays.asList(this.categories);
    }

    @Override
    public boolean hasNext() {
        return this.cursor < this.totalTweets;
    }

    @Override
    public DataSet next() {
        return this.next(this.batchSize);
    }
}
