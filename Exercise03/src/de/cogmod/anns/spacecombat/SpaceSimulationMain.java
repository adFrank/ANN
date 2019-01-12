package de.cogmod.anns.spacecombat;

/**
 * @author Sebastian Otte
 */
public class SpaceSimulationMain {
    
    public static void main(String[] args) {
        //
        final SpaceSimulation sim       = new SpaceSimulation();
        final AIMComputer aimcontroller = new AIMComputer();
        sim.addObserver(aimcontroller);
        //
        final SpaceSimulationGUI simgui = new SpaceSimulationGUI(
            sim,
            aimcontroller
        );
        //
        simgui.setVisible(true);
        simgui.start();
        
        // * UNCOMMENT NEXT LINE TO RECORD MORE RANDOM MISSILE FLIGHTS
        // RandomMissileTrajectories rmt =  new RandomMissileTrajectories(sim, simgui);
    } 
    
}