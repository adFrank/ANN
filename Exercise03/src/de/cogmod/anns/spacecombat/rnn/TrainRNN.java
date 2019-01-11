package de.cogmod.anns.spacecombat.rnn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import de.cogmod.anns.misc.BasicLearningListener;
import de.cogmod.anns.spacecombat.Missile;
import de.cogmod.anns.spacecombat.RandomMissileTrajectories;

public class TrainRNN {

	final static double TRAINING_RATIO = 0.8;
	
	public static void main(String[] args) {
        TrajectorySample[] samples =  RandomMissileTrajectories.getSavedSamples();
        int numberOfTrainingSamples = (int)(samples.length * TRAINING_RATIO);
        int numberOfTestSamples = samples.length - numberOfTrainingSamples;
        
        
        double[][][] input = new double[numberOfTrainingSamples][][];
        double[][][] target = new double[numberOfTrainingSamples][][];
        for (int i = 0; i < target.length; i++) {
        	input[i] = samples[i].getInput();
        	target[i] = samples[i].getTarget();
		}
        
        double[][][] input_test = new double[numberOfTestSamples][][];
        double[][][] target_test = new double[numberOfTestSamples][][];
        for (int i = 0; i < target_test.length; i++) {
        	input_test[i] = samples[i + numberOfTrainingSamples].getInput();
        	target_test[i] = samples[i + numberOfTrainingSamples].getTarget();
		}
        
        Random rnd = new Random(1234);
        RecurrentNeuralNetwork rnn = new RecurrentNeuralNetwork(5, 16, 8, 3);
        rnn.initializeWeights(rnd, 0.1);
        rnn.rebufferOnDemand((int)Missile.MAX_LIFETIME);
        rnn.setBias(0, false);
        rnn.setBias(1, false);
        rnn.setBias(2, false);
        rnn.setBias(3, false);
        int epochs = 1000;
        double learningrate = 1e-6;
        double momentumrate = 0.9;
        rnn.trainStochastic(rnd, input, target, epochs, learningrate, momentumrate, new BasicLearningListener());
        
        testRNN(rnn, input_test, target_test);
        double[] weights = new double[rnn.getWeightsNum()];
        rnn.readWeights(weights);
        saveWeights(weights);
	}
	
	private static void testRNN(RecurrentNeuralNetwork rnn, double[][][] input_test, double[][][] target_test) {
		double testError = 0.0;
		for (int i = 0; i < target_test.length; i++) {
			testError += RecurrentNeuralNetwork.RMSE(rnn.forwardPass(input_test[i]), target_test[i]);
		}
		testError /= target_test.length;
		System.out.println("Test error -> :" + testError);
	}
	
	private static void saveWeights(double[] weights) {
		PrintWriter out;
		try {
			out = new PrintWriter("data/trained_rnn_weights.txt");
			for (int i = 0; i < weights.length; i++) {
				out.println(weights[i]);
			}
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public static double[] loadTrainedWeights() {
		File file = new File("data/trained_rnn_weights.txt");
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(file));
			String st; 
			ArrayList <Double> weights = new ArrayList<>();
			while ((st = br.readLine()) != null) {
			    Matcher matcher = Pattern.compile("[-+]?\\d*\\.?\\d+([eE][-+]?\\d+)?").matcher(st);

			    while (matcher.find())
			    {
			        double element = Double.parseDouble(matcher.group());
			        weights.add(element);
			    }
		 	}
			br.close();
			double[] output = new double[weights.size()];
			for (int i = 0; i < weights.size(); i++) {
				output[i] = weights.get(i);
			}
			
			return output;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return null;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
	}

}
