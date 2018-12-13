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
    
    // maybe only required for dummy trajectory.
    private Random rnd = new Random(1234);

    private Vector3d[] generateDummyFutureProjection(final int timesteps) {
        //
        Vector3d last           = this.enemy.getRelativePosition();
        final Vector3d dir      = new Vector3d();
        final Vector3d[] result = new Vector3d[timesteps]; 
        //
        for (int t = 0; t < timesteps; t++) {
            dir.x += rnd.nextGaussian() * 0.1;
            dir.y += rnd.nextGaussian() * 0.1;
            dir.z += rnd.nextGaussian() * 0.1;
            //
            Vector3d.normalize(dir, dir);
            //
            dir.x *= 1.0;
            dir.y *= 1.0;
            dir.z *= 1.0;
            
            final Vector3d current = Vector3d.add(last, dir);
            result[t] = Vector3d.add(current, enemy.getOrigin());
            last = current;
        }
        return result;
    }
    
    private Vector3d[] generateFutureProjection(final int timesteps, double[][][] act) {
        //
        Vector3d last           = this.enemy.getRelativePosition();
        final Vector3d dir      = new Vector3d();
        final Vector3d[] result = new Vector3d[timesteps]; 
        //
        
        for (int j = 0; j < act.length; j++) {
        	for (int i = 0; i < act[j].length; i++) {
        		esn2.act[j][i][0] = act[j][i][0];
        	}
        }
        // double output[] = esn2.forwardPassOscillator();
        
        
        	// esn2.teacherForcing(currentPos);

        for (int t = 0; t < timesteps; t++) {
        	double[] output = esn2.forwardPassOscillator();
        	System.out.println(output[0]);
            dir.x += (output[0]);
            dir.y += (output[1]);
            dir.z += (output[2]);
            //System.out.println(dir.x);
            //
            Vector3d.normalize(dir, dir);
            //

            final Vector3d current = Vector3d.add(last, dir);
            result[t] = Vector3d.add(current, enemy.getOrigin());
            last = current;
        }
        return result;
    }
    
    @Override
    public void simulationStep(final SpaceSimulation sim) {
    	
        //
        synchronized (this) {
            //
            if (!this.targetlocked) return;
            
            //
            // update trajectory prediction RNN (teacher forcing)
            //
            final Vector3d enemyposition = sim.getEnemy().getPosition();
            final Vector3d enemyrelativepostion = sim.getEnemy().getRelativePosition();
            double[] currentPos = {
            	enemyrelativepostion.x,
                enemyrelativepostion.y,
                enemyrelativepostion.z
            };
            
            double[] output = esn1.forwardPassOscillator();
            // System.out.println(enemyrelativepostion.x + " | " + output[0]);
            esn1.teacherForcing(currentPos);
            double[][][] act = esn1.getAct().clone();
            // ...
            //
            // use copy of the RNN to generate future projection
            //
            this.enemytrajectoryprediction = this.generateFutureProjection(100, act);
            
            //
            // will be later extended for missile control
            //
            
            // 1. ESN dauer washout mit aktueller position des T-Fighters
            // 2. ESN predicted die zukunft indem aktueller hidden state von ESN1 reingepackt wird und es die nächsten Werte voraussieht
            
        }

    }
	
    
	
}