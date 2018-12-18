package de.cogmod.anns.spacecombat;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.swing.Timer;

import de.cogmod.anns.math.Vector3d;

public class RandomMissileTrajectories {
	
	private SpaceSimulation sim;
	private SpaceSimulationGUI simgui;
	
	public RandomMissileTrajectories(SpaceSimulation sim, SpaceSimulationGUI simgui) {
		this.sim = sim;
		this.simgui = simgui;
		
		
		final double fps    = 30.0;
        final double dtmsec = 1000.0 / fps;
        final double n = 10;
        
        
        
        
        Timer timer = new Timer((int)(dtmsec), new ActionListener() {
        	
        	int framecount = 0;
        	int actionRepeat = 5;
        	Random rnd = new Random(10);
        	double rotx = mapToRange(rnd.nextDouble(), 0, 1, -1, 1);
        	double roty = mapToRange(rnd.nextDouble(), 0, 1, -1, 1);
        	int missileCount = 0;
        	ArrayList<Vector3d> missileCoordinates = new ArrayList<>();
        	//
            @Override
            public void actionPerformed(final ActionEvent e) {
            	List<Missile> missiles = sim.getMissiles();
            	if (missiles.size() == 0) {
            		writeToFile(missileCoordinates, missileCount);
            		missileCount++;
            		missileCoordinates = new ArrayList<>();
            		sim.launchMissile();
            	}
            	Missile missile = sim.getMissiles().get(sim.getMissiles().size() - 1);
            	
            	missileCoordinates.add(new Vector3d(missile.getPosition().x, missile.getPosition().y, missile.getPosition().z));
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
                } else {
                	
                }
            }
        });
        sim.launchMissile();
        timer.start();
	}
	
    public void generateRandomMissileFlights(int n) {
    	Random rnd = new Random(1234);
    	double[][] missileAdjustments = new double[n][2];
    	for (int i = 0; i < n; i++) {
    		double xAdjustment = mapToRange(rnd.nextDouble(), 0, 1, -1, 1);
    		double yAdjustment = mapToRange(rnd.nextDouble(), 0, 1, -1, 1);
    		
    		missileAdjustments[i][0] = xAdjustment;
    		missileAdjustments[i][1] = yAdjustment;
    	}
        final double fps    = 30.0;
        final double dtmsec = 1000.0 / fps;
    }
    
    private double mapToRange(double value, double oldMin, double oldMax, double newMin, double newMax) {
    	return (value - oldMin) * Math.abs(newMax - newMin) / (Math.abs(oldMax - oldMin)) + newMin;
    }
    
    private void writeToFile(List<Vector3d> list, int count) {
    	PrintWriter out;
		try {
			out = new PrintWriter("src/de/cogmod/anns/spacecombat/resources/" + count + ".txt");
			for(Vector3d vec: list) {
				out.println(vec.x + " | " + vec.y + " | " + vec.z);
	    	}
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
    	
    }
}
