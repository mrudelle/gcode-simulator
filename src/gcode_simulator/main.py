import datetime
import sys
from gcode_simulator import GCodeSimulator, GrblSettings
from viz import plot_trace


if __name__ == '__main__':
    # Example usage
    settings = GrblSettings(
        max_rate_x=3000,      # mm/min
        max_rate_y=3000,      # mm/min
        max_accel_x=800,   # mm/s^2
        max_accel_y=800,    # mm/s^2
        junction_deviation=0.01,  # mm
    )
    
    simulator = GCodeSimulator(settings, trace=True)
    
    test_gcode = sys.stdin.read()
    
    time, bounds = simulator.estimate_time(test_gcode)

    print(f"Estimated execution time: {datetime.timedelta(seconds=round(time))}")
    print(f"Bounds:")
    print(f"  X: {bounds.min_x:.1f} to {bounds.max_x:.1f} (width: {bounds.width:.1f}mm)")
    print(f"  Y: {bounds.min_y:.1f} to {bounds.max_y:.1f} (height: {bounds.height:.1f}mm)")

    plot_trace(simulator.trace_nodes, simulator.bounds)