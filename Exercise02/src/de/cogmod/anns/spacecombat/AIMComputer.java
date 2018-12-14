package de.cogmod.anns.spacecombat;

import java.util.Random;

import de.cogmod.anns.math.Vector3d;
import de.cogmod.anns.spacecombat.examples.TeacherForcingExample;
import de.cogmod.anns.spacecombat.rnn.EchoStateNetwork;

/**
 * @author Sebastian Otte
 */
public class AIMComputer implements SpaceSimulationObserver {

    private EnemySpaceShip enemy        = null;
    private boolean        targetlocked = false;
    
    private Vector3d[] enemytrajectoryprediction;
    private EchoStateNetwork esn1;
    private EchoStateNetwork esn2;
    
    public AIMComputer() {
    	esn1 = new EchoStateNetwork(3, 40, 3);
    	esn2 = new EchoStateNetwork(3, 40, 3);
    	double[] trainedWeights = TrainESN.loadTrainedWeights();

    	// Load in trained ESN
    	esn1.initializeWeights(new Random(1234), 0.1);
    	esn2.initializeWeights(new Random(1234), 0.1);
    	esn1.writeWeights(trainedWeights);
    	esn2.writeWeights(trainedWeights);
    	
    	esn1.setBias(0, false);
    	esn1.setBias(1, false);
    	esn1.setBias(2, false);
    	esn2.setBias(0, false);
    	esn2.setBias(1, false);
    	esn2.setBias(2, false);
    }
    
    public Vector3d[] getEnemyTrajectoryPrediction() {
        return this.enemytrajectoryprediction;
    }
    
    public boolean getTargetLocked() {
        return this.targetlocked;
    }
    
    public EnemySpaceShip getTarget() {
        return this.enemy;
    }
    
    public void releaseTarget() {
        synchronized (this) {
            this.enemy = null;
            this.targetlocked = false;
        }
    }
    
    public void lockTarget(final EnemySpaceShip enemy) {
        synchronized (this) {
            this.enemy        = enemy;
            this.targetlocked = true;
        }
    }
    


    private Vector3d[] generateFutureProjection(final int timesteps) {

    	final Vector3d[] result = new Vector3d[timesteps]; 

    	// Copy activations from esn1 to esn2
    	double[][][] act = esn1.getAct().clone();
    	for (int j = 0; j < act.length; j++) {
        	for (int i = 0; i < act[j].length; i++) {
        		esn2.act[j][i][0] = act[j][i][0];
        	}
        }
    	
    	// predict trajectory
        for (int t = 0; t < timesteps; t++) {
        	double[] output = esn2.forwardPassOscillator();
            result[t] = Vector3d.add(enemy.getOrigin(), new Vector3d(output[0], output[1], output[2]));
        }
        return result;
    }
    
    @Override
    public void simulationStep(final SpaceSimulation sim) {
    	
        //
        synchronized (this) {
            //
            if (!this.targetlocked) return;
            

            final Vector3d enemyrelativepostion = sim.getEnemy().getRelativePosition();
            double[] currentPos = {
            	enemyrelativepostion.x,
                enemyrelativepostion.y,
                enemyrelativepostion.z
            };
            
            // ESN1 is in permanent washout state
            esn1.forwardPassOscillator();
            esn1.teacherForcing(currentPos);

            this.enemytrajectoryprediction = this.generateFutureProjection(100);
        }

    }
	
    
	
}