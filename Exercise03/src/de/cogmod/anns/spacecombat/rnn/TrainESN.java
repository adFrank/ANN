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

public class TrainESN {

	public static void main(String[] args) {
		
		String path = "src/de/cogmod/anns/spacecombat/resources/recorded_trajectory.txt";
		double[][] sequence = readRecordedTrajectory(path);
		
		EchoStateNetwork esn = new EchoStateNetwork(3, 40, 3);
		esn.initializeWeights(new Random(1234), 0.1);
		esn.setBias(0, false);
		esn.setBias(1, false);
		esn.setBias(2, false);
		
		int washout = 200;
		int training = 400;
		int test = 400;

		esn.trainESN(sequence, washout, training, test);
		double weights[] = new double[esn.getWeightsNum()];
		
		
		// read final weights and save to file
		esn.readWeights(weights);
		PrintWriter out;
		try {
			out = new PrintWriter("src/de/cogmod/anns/spacecombat/resources/trained_weights.txt");
			for (int i = 0; i < weights.length; i++) {
				out.println(weights[i]);
			}
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	private static double[][] readRecordedTrajectory(String path) {
		File file = new File(path); 
		double[][] sequence = new double[1000][3];
		int idx = 0;
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(file));
			String st; 
			while ((st = br.readLine()) != null) {
				ArrayList<Double> coordinates = new ArrayList<>();
			    Matcher matcher = Pattern.compile( "[-+]?\\d*\\.?\\d+([eE][-+]?\\d+)?").matcher(st);

			    while ( matcher.find() )
			    {
			        double element = Double.parseDouble(matcher.group());
			        coordinates.add(element);
			    }

			    for (int i = 0; i < coordinates.size(); i++  ) {
			    	sequence[idx][i] = coordinates.get(i);
			    }
				idx++;
		 	}
			br.close();
			return sequence;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return null;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
	}
	
	public static double[] loadTrainedWeights() {
		File file = new File("src/de/cogmod/anns/spacecombat/resources/trained_weights.txt");
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
