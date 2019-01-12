package de.cogmod.anns.spacecombat;

import java.util.List;

import de.cogmod.anns.math.Vector3d;
import de.cogmod.anns.spacecombat.AIMComputer;
import de.cogmod.anns.spacecombat.EnemySpaceShip;
import de.cogmod.anns.spacecombat.SpaceSimulation;
import de.cogmod.anns.spacecombat.rnn.EchoStateNetwork;
import de.jannlab.io.Serializer;

/**
 * @author Sebastian Otte
 */
public class AIMComputer implements SpaceSimulationObserver {

    public final static int PREDICTION_LENGTH = 100;
    
    private EnemySpaceShip enemy        = null;
    private boolean        targetlocked = false;
    
    private Vector3d[] enemytrajectoryprediction;
    
    private EchoStateNetwork enemyesn;
    private EchoStateNetwork enemyesncopy;
    
    private int lastMissileHashCode = -1;
    
    private Vector3d optimalTarget;
    private boolean missileLaunched = false;
    
    private MissileController mcon;
    
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
            this.enemyesn.reset();
        }
    }
    
    private Vector3d[] generateESNFutureProjection(final int timesteps) {
        //
        // copy current state of the online ESN into the projection ESN.
        //
        final double[][][] source = this.enemyesn.getAct();
        final double[][][] dest   = this.enemyesncopy.getAct();
        //
        for (int l = 0; l < source.length; l++) {
            if (source[l] != null) {
                for (int i = 0; i < source[l].length; i++) {
                    for (int t = 0; t < source[l][i].length; t++) {
                        dest[l][i][t] = source[l][i][t];
                    }
                }
            }
        }
        //
        // perform projection.
        //
        final Vector3d[] result = new Vector3d[timesteps]; 
        //
        for (int t = 0; t < timesteps; t++) {
            final double[] output = this.enemyesncopy.forwardPassOscillator();
            final Vector3d pos = new Vector3d(
                output[0],
                output[1],
                output[2]
            );
            //
            Vector3d.add(pos, this.enemy.getOrigin(), pos);
            //
            result[t] = pos;
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
            final Vector3d enemyrelativeposition = sim.getEnemy().getRelativePosition();
            //
            final double[] update = {
                enemyrelativeposition.x,
                enemyrelativeposition.y,
                enemyrelativeposition.z
            };
            //
            this.enemyesn.forwardPassOscillator();
            this.enemyesn.teacherForcing(update);
            //
            // use copy of the RNN to generate future projection
            //
            this.enemytrajectoryprediction = this.generateESNFutureProjection(PREDICTION_LENGTH);
            //
            // grab the most recently launched missile that is alive.
            //
            final Missile currentMissile = lastActiveMissile(sim);
            //
            //
            
            if (currentMissile != null) {
            	int currentMissileHashCode = currentMissile.hashCode();
            	
            	// compute optimal target only once a new missile was launched
            	if(!missileLaunched || (lastMissileHashCode != currentMissileHashCode)) {
            		missileLaunched = true;
            		lastMissileHashCode = currentMissileHashCode;
            		
            		optimalTarget = mcon.findFirstReachableTarget(
            				enemytrajectoryprediction, 
            				50); // ignore the first 50 predicted target positions as they will be very likely out of reach in the given time
            		// fire straight if no reachable target was found
            		if (optimalTarget == null) {
            			optimalTarget = new Vector3d(0, -1, 150);
            		}
            	}
            	// optimize command and apply it to the missile
            	double[] commandToApply =  mcon.optimizeMotorCommands(currentMissile, optimalTarget);
            	currentMissile.adjust(commandToApply[0], commandToApply[1]);
            } else {
            	missileLaunched = false;
            	lastMissileHashCode = -1;
            }
        }
    }
    
    /**
     * Returns the most recently launched missile within the simulation, but only
     * if it is still "alive". Otherwise the method returns null. 
     */
    private Missile lastActiveMissile(final SpaceSimulation sim) {
        final List<Missile> missiles = sim.getMissiles();
        if (missiles.size() > 0) {
            final Missile lastMissile = missiles.get(missiles.size() - 1);
            if (lastMissile.isLaunched() && !lastMissile.isDestroyed()) {
                return lastMissile;
            }
        }
        return null;
    }
    
    public AIMComputer() {
        try {
            //
            // load esn.
            //
            final int reservoirsize = 30;
            this.enemyesn     = new EchoStateNetwork(3, reservoirsize, 3);
            this.enemyesncopy = new EchoStateNetwork(3, reservoirsize, 3);
            //
            final String weightsfile = (
                "data/esn-3-" + 
                reservoirsize + "-3.weights"
            );
            final double[] weights = Serializer.read(weightsfile);
            //
            this.enemyesn.writeWeights(weights);
            this.enemyesncopy.writeWeights(weights);
            //
            mcon = new MissileController();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }   
}