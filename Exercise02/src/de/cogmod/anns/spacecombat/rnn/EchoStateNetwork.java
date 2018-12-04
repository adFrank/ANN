package de.cogmod.anns.spacecombat.rnn;

import static de.cogmod.anns.spacecombat.rnn.ReservoirTools.map;
import static de.cogmod.anns.spacecombat.rnn.ReservoirTools.multiply;

import java.util.Arrays;
import java.util.Random;

import de.jannlab.optimization.BasicOptimizationListener;
import de.jannlab.optimization.Objective;
import de.jannlab.optimization.optimizer.DifferentialEvolution;
import de.jannlab.optimization.optimizer.DifferentialEvolution.Mutation;

/**
 * @author Sebastian Otte
 */
public class EchoStateNetwork extends RecurrentNeuralNetwork {

    private double[][] inputweights;
    private double[][] reservoirweights;
    private double[][] outputweights;
    
    public double[][] getInputWeights() {
        return this.inputweights;
    }
    
    public double[][] getReservoirWeights() {
        return this.reservoirweights;
    }
    
    public double[][] getOutputWeights() {
        return this.outputweights;
    }
    
    public EchoStateNetwork(
        final int input,
        final int reservoirsize,
        final int output
    ) {
        super(input, reservoirsize, output);
        //
        this.inputweights     = this.getWeights()[0][1];
        this.reservoirweights = this.getWeights()[1][1];
        this.outputweights    = this.getWeights()[1][2];
        //
    }
    
    @Override
    public void rebufferOnDemand(int sequencelength) {
        super.rebufferOnDemand(1);
    }
    
    /**
     * Returns the 
     */
    public double[] output() {
        //
        final double[][][] act = this.getAct();
        final int outputlayer  = this.getOutputLayer(); 
        //
        final int n = act[outputlayer].length;
        //
        final double[] result = new double[n];
        final int t = Math.max(0, this.getLastInputLength() - 1);
        //
        for (int i = 0; i < n; i++) {
            result[i] = act[outputlayer][i][t];
        }
        //
        return result;
    }
    
    /**
     * This is an ESN specific forward pass realizing 
     * an oscillator by means of an output feedback via
     * the input layer. This method requires that the input
     * layer size matches the output layer size. 
     */
    public double[] forwardPassOscillator() {
        //
        // this method causes an additional copy operation
        // but it is more readable from outside.
        //
        final double[] output = this.output();
        return this.forwardPass(output);
    }
    
    /**
     * Overwrites the current output with the given target.
     */
    public void teacherForcing(final double[] target) {
        //
        final double[][][] act = this.getAct();
        final int outputlayer  = this.getOutputLayer(); 
        //
        final int n = act[outputlayer].length;
        //
        final int t = this.getLastInputLength() - 1;
        //
        for (int i = 0; i < n; i++) {
            act[outputlayer][i][t] = target[i];
        }
    }
    
    /**
     * ESN training algorithm. 
     */
    public double trainESN(
        final double[][] sequence,
        final int washout,
        final int training,
        final int test
    ) {
        //
        // ESN training algorithm 
        //
    	double[][][] act = this.getAct();
    	int outputlayer = this.getOutputLayer();
    	int hiddenlayer = outputlayer-1;
    	int numberOfHiddenNeurons = act[hiddenlayer].length;
    	int numberOfOutputNeurons = act[outputlayer].length;
    	
    	// 1. washout 
    	for (int t = 0; t < washout; t++) {
    		
    		// ignore output during washout
    		this.forwardPassOscillator();
    		
    		double[] target = sequence[t];
    		this.teacherForcing(target);
    	}
    	
    	
    	// 1. training with teacher forcing
    	
    	// Activation matrix is of dimension timesteps x hidden units
    	double[][] activationMatrix = new double[training][numberOfHiddenNeurons];
    	
    	// output matrix is of dimension timesteps x output units
    	double[][] desiredOutputs = new double[training][numberOfOutputNeurons];
    	
    	// weight out matrix is of dimension (hidden units x output units) 
    	double[][] weightOut = new double[numberOfHiddenNeurons][numberOfOutputNeurons];
    	
    	for (int i = 0; i < training; i++) {
    		this.forwardPassOscillator();
    		act = this.getAct();
    		
    		// iterate over reservoir
    		for (int j = 0; j < act[hiddenlayer].length; j++) {
    			// TODO: is index 0 correct here? Can time index be ignored?
    			activationMatrix[i][j] = act[hiddenlayer][j][0];
    		}
    		double[] target = sequence[i+washout];
    		desiredOutputs[i] = target;
    		this.teacherForcing(target);
    	}
    	
    	double trainingError = 0.0;
    	
    	
    	if(ReservoirTools.solveSVD(activationMatrix, desiredOutputs, weightOut)) {
    		// optimal weight out was successfully computed
    		this.outputweights = weightOut;
    		
//    		int arity = (numberOfOutputNeurons + numberOfHiddenNeurons) * numberOfHiddenNeurons;
//    		int flatWeightsIndexBoundary = numberOfOutputNeurons + numberOfHiddenNeurons;
    		
    		Objective objective = this.defineObjective(washout, training, numberOfHiddenNeurons, numberOfOutputNeurons, sequence);
    		System.out.println(ReservoirTools.matrixAsString(evolutionaryOptimization(objective, numberOfHiddenNeurons, numberOfOutputNeurons)));
    		
    	} else {
    		System.out.println("Ax = B could not be solved");
    	}
    	
    	System.out.println("TRAIN ESN ERROR");
    	System.out.println(trainingError / (numberOfOutputNeurons*training));
    	// return MSE error
    	return trainingError / (numberOfOutputNeurons*training);
    }
    
    
    private Objective defineObjective(
		int washout,
		int training,
		int numberOfHiddenNeurons,
		int numberOfOutputNeurons,
		double[][] sequence)
    {
    	
    	// w_fb is a matrix of size output neurons x hidden neurons
    	// w_rec is a matrix of size hidden neurons x hidden neurons
    	// thus, total matrix to optimize is of size (output + hidden) x hidden neurons
    	
    	
        final Objective objective = new Objective() {
            //
            @Override
            public int arity() {
                return (numberOfOutputNeurons + numberOfHiddenNeurons) * numberOfHiddenNeurons;
            }
            @Override
            /**
             * This is the callback method that is called from the 
             * optimizer to compute the "fitness" of a particular individual.
             */
            public double compute(double[] values, int offset) {
                //
                // the parameters for which the optimizer requests a fitness
                // value or stored in values starting at the given offset
                // with the length that is given via arity(), namely, sizex.
                //
            	
            	// values contains the flattened array and is of size arity()
            	// the input weights in this array range from 0 to output neurons * hidden neurons
            	int numberOfInputWeights = numberOfOutputNeurons*numberOfHiddenNeurons;
                final double[][] tentative_inputweights = new double[numberOfOutputNeurons][numberOfHiddenNeurons];
                final double[] flatInputWeights = Arrays.copyOfRange(values, offset, offset+numberOfInputWeights);
                map(flatInputWeights, 0, tentative_inputweights);
                
                // write tentative input weights
                inputweights = tentative_inputweights; 
                
                
                // the recurrent weights in this array range from output neurons * hidden neurons + 1 to end of values
                int numberOfRecurrentWeights = numberOfHiddenNeurons*numberOfHiddenNeurons;
                final double[][] tentative_recurrentweights = new double[numberOfHiddenNeurons][numberOfHiddenNeurons];
                final double[] flatRecurrentWeights = Arrays.copyOfRange(values, offset+numberOfInputWeights, offset+numberOfInputWeights+numberOfRecurrentWeights);
                map(flatRecurrentWeights, 0, tentative_recurrentweights);
                
                reservoirweights = tentative_recurrentweights;
                
                //
                // Evaluate current sample.
                //
                
        		// washout 
            	for (int t = 0; t < washout; t++) {
            		
            		// ignore output during washout
            		forwardPassOscillator();
            		
            		double[] target = sequence[t];
            		teacherForcing(target);
            	}
            	
            	double error = 0.0;

            	// training without teacher forcing
            	for (int i = 0; i < training; i++) {
            		double[] output = forwardPassOscillator();
            		double[] target = sequence[i+washout];
            		
            		// iterate over reservoir
            		for (int j = 0; j < output.length; j++) {
            			error += Math.pow((output[j] - target[j]), 2);
            		}
            	}
                
                return Math.sqrt( (1.0 / 1.5) * error);
            }
        };
        return objective;
    }
    
    private double[][] evolutionaryOptimization(Objective f, int numberOfHiddenNeurons, int numberOfOutputNeurons) {
    	final DifferentialEvolution optimizer = new DifferentialEvolution();
        //
        // The same parameters can be used for reservoir optimization.
        //
        optimizer.setF(0.4);
        optimizer.setCR(0.6);
        optimizer.setPopulationSize(10);
        optimizer.setMutation(Mutation.CURR2RANDBEST_ONE);
        //
        optimizer.setInitLbd(-0.1);
        optimizer.setInitUbd(0.1);
        //
        // Obligatory things...
        // 
        optimizer.setRnd(new Random(1234));
        optimizer.setParameters(f.arity());
        optimizer.updateObjective(f);
        //
        // for observing the optimization process.
        //
        // optimizer.addListener(new BasicOptimizationListener());
        //
        optimizer.initialize();
        //
        // go!
        //
        optimizer.iterate(1000, 0.0);
        //
        // read the best solution.
        //
        final double[] solution = new double[f.arity()];
        final double[][] output = new double[numberOfHiddenNeurons + numberOfOutputNeurons][numberOfHiddenNeurons];
        optimizer.readBestSolution(solution, 0);
        map(solution, 0, output);
        return output;
    }
}