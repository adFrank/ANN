package de.cogmod.anns.spacecombat;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.swing.Timer;

import de.cogmod.anns.math.Vector3d;
import de.cogmod.anns.spacecombat.rnn.RecurrentNeuralNetwork;
import de.cogmod.anns.spacecombat.rnn.TrainRNN;
import de.cogmod.anns.spacecombat.rnn.TrajectorySample;

public class RandomMissileTrajectories {
	
	private SpaceSimulation sim;
	private SpaceSimulationGUI simgui;
	
	public RandomMissileTrajectories(SpaceSimulation sim, SpaceSimulationGUI simgui) {
		this.sim = sim;
		this.simgui = simgui;
		
		
		final double fps    = 30.0;
        final double dtmsec = 1000.0 / fps;

        Timer timer = new Timer((int)(dtmsec), new ActionListener() {
        	
        	int framecount = 0;
        	int actionRepeat = 5;
        	Random rnd = new Random(10);
        	double rotx = mapToRange(rnd.nextDouble(), 0, 1, -1, 1);
        	double roty = mapToRange(rnd.nextDouble(), 0, 1, -1, 1);
        	ArrayList<Vector3d> missileCoordinates = new ArrayList<>();
        	ArrayList<double[]> motorCommands = new ArrayList<>();
        	//
            @Override
            public void actionPerformed(final ActionEvent e) {
            	List<Missile> missiles = sim.getMissiles();
            	if (missiles.size() == 0) {
            		writeToFile(missileCoordinates, motorCommands);
            		missileCoordinates = new ArrayList<>();
            		motorCommands = new ArrayList<>();
            		sim.launchMissile();
            	}
            	Missile missile = sim.getMissiles().get(sim.getMissiles().size() - 1);
            	
            	Vector3d currCoords = new Vector3d(missile.getPosition().x, missile.getPosition().y, missile.getPosition().z);
            	missileCoordinates.add(currCoords);
            	
            	framecount++;
                //
                if (framecount % actionRepeat == 0) {
                	framecount = 0;
                	double p = rnd.nextDouble();
                	if (p < 0.5) {
                		rotx = mapToRange(rnd.nextDouble(), 0, 1, -1, 1);
                		roty = mapToRange(rnd.nextDouble(), 0, 1, -1, 1);
                	}
                }
                
                if (
                    missile!= null && 
                    missile.isLaunched() && 
                    !missile.isDestroyed()
                ) {
                    missile.adjust(rotx, roty);
                    double[] cmds = new double[]{ rotx, roty};
                    motorCommands.add(cmds);
                } else {
                	
                }
            }
        });
         sim.launchMissile();
         timer.start();
	}
    
    private double mapToRange(double value, double oldMin, double oldMax, double newMin, double newMax) {
    	return (value - oldMin) * Math.abs(newMax - newMin) / (Math.abs(oldMax - oldMin)) + newMin;
    }
    
    private void writeToFile(List<Vector3d> coords, List<double[]> cmds) {
    	int count = new File("data/missile_flights/").list().length + 1;
    	PrintWriter out;
		try {
			out = new PrintWriter("data/missile_flights/" + count + ".txt");
			for(int t = 0; t < coords.size(); t++ ) {
				double[] currCmds = cmds.get(t);
				Vector3d currVec = coords.get(t);
				Vector3d prevVec = currVec;
				if (t > 0) {
					prevVec = coords.get(t-1);
				}
				
				if (t < coords.size() - 1) {
					Vector3d targetVec = coords.get(t+1);
					Vector3d deltaInput = Vector3d.sub(currVec, prevVec);
					Vector3d deltaTarget = Vector3d.sub(targetVec, currVec);
					out.println(deltaInput.x + " | " + deltaInput.y + " | " + deltaInput.z + " | " +  currCmds[0] + " | " + currCmds[1] + ":" + deltaTarget.x + " | " + deltaTarget.y + " | " + deltaTarget.z);				
				}
	    	}
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
    	
    }

    private static TrajectorySample readTrajectoryFromFile(int number) {
    	File file = new File("data/missile_flights/" + number + ".txt");
    	final String pattern = "[-+]?\\d*\\.?\\d+([eE][-+]?\\d+)?";
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(file));
			String st;
			double[] input = new double[5];
			double[] target = new double[3];
			ArrayList<double[]> input_sequence = new ArrayList<>();
			ArrayList<double[]> target_sequence = new ArrayList<>();
			while ((st = br.readLine()) != null) {
			    
			    String[] input_target = st.split(":");
			    Matcher matcher = Pattern.compile(pattern).matcher(input_target[0]);
			    
			    int idx = 0;
			    while (matcher.find())
			    {
			        input[idx] = Double.parseDouble(matcher.group());
			        idx++;
			    }
			    
			    idx = 0;
			    matcher = Pattern.compile(pattern).matcher(input_target[1]);
			    while (matcher.find())
			    {
			    	target[idx] = Double.parseDouble(matcher.group());
			        idx++;
			    }
			 
			    input_sequence.add(input.clone());
			    target_sequence.add(target.clone());
			}
			br.close();
			return new TrajectorySample(input_sequence, target_sequence);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return null;
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
    }
    
    public static TrajectorySample[] getSavedSamples() {
    	int numberOfSamples = new File("data/missile_flights/").list().length;
    	TrajectorySample[] result = new TrajectorySample[numberOfSamples];
    	for (int i = 0; i < numberOfSamples; i++) {
    		result[i] = readTrajectoryFromFile(i+1);
    	}
    	return result;
    }
}
