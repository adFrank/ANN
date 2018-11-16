package de.cogmod.anns.exercisesheet01;

import java.util.Random;

import de.cogmod.anns.exercisesheet01.misc.BasicLearningListener;
import de.cogmod.anns.exercisesheet01.misc.Spiral;
import de.cogmod.anns.exercisesheet01.misc.TrajectoryGenerator;

public class VanishingGradient {
	private static Random rnd = new Random(100L);

	public static void main(String[] args) {
		
		final TrajectoryGenerator gen = new Spiral();
        final int trainlength         = 100;
        // 
        //
        final double[][][] inputs  = new double[1][trainlength][1];
        final double[][][] targets = new double[1][1][1];
        //
        // set only the first value of the input sequence to 1.0.
        //
        for (int i = 0; i < inputs[0].length; i++) {
			inputs[0][i][0] = 1;
		}
        
    	targets[0][0][0] = 1;
        //
        // set up network. biases are used by default, but
        // be deactivated using net.setBias(layer, false),
        // where layer gives the layer index (1 = is the first hidden layer).
        //
        final RecurrentNeuralNetwork net = new RecurrentNeuralNetwork(1, 1, 1);
        //
        // we disable all biases.
        //
        net.setBias(1, false);
        net.setBias(2, false);
        //
        // perform training.
        //
        final int epochs = 1;
        final double learningrate = 0.0001;
        final double momentumrate = 0.1;
        //
        // generate initial weights and prepare the RNN buffer
        // for BPTT over the required number of time steps.
        //
        net.initializeWeights(rnd, 0.1);
        net.rebufferOnDemand(trainlength);
        //
        // start the training.
        //
        final double error = net.trainStochastic(
            rnd, 
            inputs,
            targets,
            epochs,
            learningrate,
            momentumrate,
            new BasicLearningListener()
        );
        //
        System.out.println();
        System.out.println("final error: " + error);

	}

}
