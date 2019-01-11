package de.cogmod.anns.spacecombat;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;

import de.cogmod.anns.math.Vector3d;
import de.cogmod.anns.spacecombat.rnn.RecurrentNeuralNetwork;
import de.cogmod.anns.spacecombat.rnn.TrainRNN;

public class MissileController {

	final int PREDICTED_FRAMES = 5;
	final int OPTIMIZATION_STEPS = 20;
	
	private RecurrentNeuralNetwork rnn1;
	private RecurrentNeuralNetwork rnn2;
	private Random rnd;
	
	private Vector3d prev_pos;
	private LinkedList<double[]> policy;
	
	public MissileController() {
		policy = new LinkedList<>();
		for (int i = 0; i < PREDICTED_FRAMES; i++) {
			policy.add(new double[]{0, 0});
		}
		
		double[] trainedWeights = TrainRNN.loadTrainedWeights();
		rnn1 = new RecurrentNeuralNetwork(5, 20, 3);

		rnd = new Random(1234);
        rnn1.initializeWeights(rnd, 0.1);
        rnn1.rebufferOnDemand(PREDICTED_FRAMES);

        rnn1.setBias(0, false);
        rnn1.setBias(1, false);
        rnn1.setBias(2, false);
  
        
        rnn1.writeWeights(trainedWeights);
        prev_pos = new Vector3d(0,-1,0);
	}
	
	public double[] optimizeMotorCommands(Missile missile, Vector3d[] enemytrajectoryprediction) {
		
		
		Vector3d sampleTarget = new Vector3d(34, 49, 76);
		
		double[][] input_seq = new double[PREDICTED_FRAMES][2];
		
		Vector3d currentPos = missile.getPosition().copy();
		Vector3d currentVel = Vector3d.sub(currentPos, prev_pos);
		this.prev_pos = currentPos.copy();

		
		double[] initialVelocity = new double[] {
				currentVel.x,
				currentVel.y,
				currentVel.z
		};
		
		for (int i = 0; i < input_seq.length; i++) {
			input_seq[i] = policy.get(i);
		}
		
		
		for (int i = 0; i < OPTIMIZATION_STEPS; i++) {
			
			
			double[][] output = rnn1.recursiveForwardPass(initialVelocity, input_seq);
			double[][] positions = new double[input_seq.length][3];
			Vector3d lastPos = currentPos;
			
			// Compute predicted positions in each time steps
			for (int t = 0; t < PREDICTED_FRAMES; t++) {
				Vector3d t_vel = new Vector3d(output[t][0], output[t][1], output[t][2]);
				Vector3d t_pos = Vector3d.add(lastPos, t_vel);
				positions[t] = new double[]{
					t_pos.x,
					t_pos.y,
					t_pos.z
				};
				lastPos = t_pos;
			}
			
			double[][] target_seq = new double[PREDICTED_FRAMES][3];
			
			// Compute target sequence by calculating discrepancy between prediction and target. Afterwards normalize discrepancy and multiply with predicted velocity
			for (int t = 0; t < PREDICTED_FRAMES; t++) {
				Vector3d discrepancy = Vector3d.sub(sampleTarget, new Vector3d(positions[t][0], positions[t][1], positions[t][2]));
				Vector3d normalizedDiscrepancy = Vector3d.normalize(discrepancy);
				Vector3d t_vel = new Vector3d(output[t][0], output[t][1], output[t][2]);
				Vector3d t_target_vel = Vector3d.mul(normalizedDiscrepancy, t_vel.length());
				
				target_seq[t] = new double[]{
					t_target_vel.x,
					t_target_vel.y,
					t_target_vel.z
				};
			}
			
			
		
			rnn1.backwardPass(target_seq);

			double[][][] act = rnn1.getAct();
			double[][][][] weights = rnn1.getWeights();
			double[][][] delta = rnn1.getDelta();
			
			
			// loop over first hidden layer and sum up weighted input deltas
			
			for (int t = 0; t < PREDICTED_FRAMES; t++) {
				double delta_x = 0;
				double delta_y = 0;
				for (int j = 0; j < act[1].length; j++) {
					delta_x += weights[0][1][3][j] * delta[1][j][t];
					delta_y += weights[0][1][4][j] * delta[1][j][t];
				}
				double[] prev_command = this.policy.get(t);
				this.policy.set(t, new double[]{
						prev_command[0] += delta_x,
						prev_command[1] += delta_y
				});
				
			}
		}
		
		rnn1.reset();
		
		double[] optimizedCommand = this.policy.removeFirst();
		this.policy.add(new double[]{ 0, 0});
		
		return optimizedCommand;
		
	}
	
    private double mapToRange(double value, double oldMin, double oldMax, double newMin, double newMax) {
    	return (value - oldMin) * Math.abs(newMax - newMin) / (Math.abs(oldMax - oldMin)) + newMin;
    }
}
