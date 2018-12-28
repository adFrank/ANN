package de.cogmod.anns.spacecombat.rnn;

import java.util.Random;

import de.cogmod.anns.misc.BasicLearningListener;
import de.cogmod.anns.spacecombat.RandomMissileTrajectories;

public class TrainRNN {

	public static void main(String[] args) {
        TrajectorySample[] samples =  RandomMissileTrajectories.getSavedSamples();
        double[][][] input = new double[samples.length][][];
        double[][][] target = new double[samples.length][][];
        for (int i = 0; i < target.length; i++) {
        	input[i] = samples[i].getInput();
        	target[i] = samples[i].getTarget();
		}
        Random rnd = new Random(1234);
        RecurrentNeuralNetwork rnn = new RecurrentNeuralNetwork(5, 16, 8, 16, 3);
        rnn.initializeWeights(rnd, 0.1);
        rnn.setBias(0, false);
        rnn.setBias(1, false);
        rnn.setBias(2, false);
        rnn.setBias(3, false);
        rnn.setBias(4, false);
        int epochs = 10;
        double learningrate = 1e-8;
        double momentumrate = 1e-4;
        rnn.trainStochastic(rnd, input, target, epochs, learningrate, momentumrate, new BasicLearningListener());
	}

}
