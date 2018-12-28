package de.cogmod.anns.spacecombat.rnn;

import java.util.List;

public class TrajectorySample {

	private double[][] input;
	private double[][] target;
	
	public TrajectorySample(double[][] input, double[][] target) {
		this.input = input;
		this.target = target;
	}
	
	public TrajectorySample(List<double[]> input, List<double[]> target) {
		this.input = new double[input.size()][];
		for (int i = 0; i < input.size(); i++) {
			this.input[i] = input.get(i);
		}
		this.target = new double[target.size()][];
		for (int i = 0; i < target.size(); i++) {
			this.target[i] = target.get(i);
		}
	}

	public double[][] getInput() {
		return input;
	}

	public double[][] getTarget() {
		return target;
	}
}
