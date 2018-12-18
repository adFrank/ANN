package de.cogmod.anns.spacecombat;


import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import javax.swing.Timer;

import de.cogmod.anns.math.Vector3d;


/**
 * @author Sebastian Otte
 */
public class SpaceSimulation {
	
    public static final double HIT_DISTANCE  = 1.0;
    public static final long   EXPLOSION_AGE = 150; 
    
    private EnemySpaceShip enemy;
    private List<Missile>  missiles;

    private long globaltime;
    
    private List<Explosion> explosions;

    private List<SpaceSimulationObserver> observer;
    
    
    public void addObserver(final SpaceSimulationObserver obs) {
        this.observer.add(obs);
    }
    
    public EnemySpaceShip getEnemy() {
        return this.enemy;
    }
    
    public List<Missile> getMissiles() {
        return this.missiles;
    }
    
    public Missile launchMissile() {
    	generateRandomMissileFlights(100);
        //
        final Missile missile = new Missile();
        missile.launch();
        missile.getPosition().y = -1.0; 
        this.missiles.add(missile);
        //
        return missile;
    }
    
    private void deleteDeadMissiles() {
        final List<Missile> delete = new LinkedList<Missile>();
        for (Missile missile : this.missiles) {
            if (missile.isDestroyed()) {
                delete.add(missile);
            }
        }
        this.missiles.removeAll(delete);
    }
    
    public List<Explosion> getExplosions() {
        return this.explosions;
    }
    
    public long getGlobalTime() {
        return this.globaltime;
    }
    
    private void deleteDeadExplosions() {
        final List<Explosion> delete = new LinkedList<Explosion>();
        for (Explosion explosion : this.explosions) {
            if ((this.globaltime - explosion.getStartTime()) > EXPLOSION_AGE) {
                delete.add(explosion);
            }
        }
        this.explosions.removeAll(delete);
    }
    
    public void update() {
        this.enemy.update();
        for (Missile missile : this.missiles) {
            missile.update();
            if (
                !this.enemy.isDestroyed() && 
                Vector3d.sub(
                    missile.getPosition(),
                    this.enemy.getPosition()
                ).length() <= HIT_DISTANCE
            ) {
                this.enemy.destroy();
                missile.destroy();
                this.explosions.add(
                    new Explosion(this.enemy.getPosition(), this.globaltime)
                );
            }
        }
        this.deleteDeadMissiles();
        this.deleteDeadExplosions();
        //
        for (SpaceSimulationObserver obs : this.observer) {
            obs.simulationStep(this);
        }
        //
        this.globaltime++;
    }
    
    public void reset() {
        this.enemy.reset();
        this.missiles.clear();
        this.explosions.clear();
        //
        this.enemy.getOrigin().x = 0.0;
        this.enemy.getOrigin().y = 2.0;
        this.enemy.getOrigin().z = 20.0;
        //
        this.globaltime = 0L;
    }
    
    public SpaceSimulation() {
        this.enemy      = new EnemySpaceShip();
        this.missiles   = new ArrayList<Missile>();
        this.explosions = new ArrayList<Explosion>();
        this.observer   = new ArrayList<SpaceSimulationObserver>();
        this.reset();
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
    
    
    
	
}