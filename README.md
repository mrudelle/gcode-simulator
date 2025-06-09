# gcode-simulator

Python package for analyzing and simulating gcode tool-path for plotters, CNC, 3D printers and more.

- Accurate time estimation with junction deviation modeling
- Visualization of tool paths with feed rates

Sample output, the plot on the left and the toolpath on the right. (The y axis is inverted on my machine)
![Feed Rate Visualization Closeup](doc/imgs/feed_rate_viz_closeup.png)

## Installation

```bash
pip install gcode-simulator
```

## Usage

gcode-simulator come with both a cli and a python API.

### Command Line Interface

Analyze a G-code file to estimate execution time and visualize the tool path:

```bash
gcode-simulator path/to/your/file.gcode
```

#### Options

```
--max-rate-x FLOAT       Maximum feed rate for X axis in mm/min ($110) [default: 3000.0]
--max-rate-y FLOAT       Maximum feed rate for Y axis in mm/min ($111) [default: 3000.0]
--max-accel-x FLOAT      Maximum acceleration for X axis in mm/s^2 ($120) [default: 800.0]
--max-accel-y FLOAT      Maximum acceleration for Y axis in mm/s^2 ($121) [default: 800.0]
--junction-deviation FLOAT  Junction deviation in mm ($11) [default: 0.01]
--grbl-version TEXT      GRBL firmware version [default: 0.9i]
--visualize / --no-visualize  Display a visualization of the toolpath [default: no-visualize]
--json-output / --no-json-output  Output results in JSON format [default: no-json-output]
--help                   Show this message and exit.
```

### Python API

#### GCodeSimulator

```python
from gcode_simulator import GCodeSimulator, GrblSettings

# Configure GRBL settings
settings = GrblSettings(
    max_rate_x=3000.0,  # mm/min
    max_rate_y=3000.0,  # mm/min
    max_accel_x=800.0,  # mm/s^2
    max_accel_y=800.0,  # mm/s^2
    junction_deviation=0.01,  # mm
    grbl_version='0.9i'
)

# Create simulator with tracing enabled for visualization
simulator = GCodeSimulator(settings, trace=True)

# Load G-code from a file
with open('path/to/your/file.gcode', 'r') as f:
    gcode = f.read()

# Estimate execution time and get boundaries
time_seconds, bounds = simulator.estimate_time(gcode)

print(f"Estimated execution time: {time_seconds:.2f} seconds")
print(f"Width: {bounds.width:.2f}mm, Height: {bounds.height:.2f}mm")
```

#### Visualization

```python
from gcode_simulator import GCodeSimulator, GrblSettings
from gcode_simulator.viz import plot_trace

# Setup and run the simulator as shown above
simulator = GCodeSimulator(settings, trace=True)
time_seconds, bounds = simulator.estimate_time(gcode)

# Visualize the tool path with color-coded feed rates
plot_trace(simulator.trace_nodes, bounds)
```

The visualization shows the G-code path with color-coded feed rates and a grid for scale. Each line segment is colored according to its feed rate, with a color bar indicating the feed rate values.

![Feed Rate Visualization](doc/imgs/feed_rate_viz.png)

![Feed Rate Visualization Closeup](doc/imgs/feed_rate_viz_closeup.png)
