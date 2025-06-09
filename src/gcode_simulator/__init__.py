"""
GCode Simulator - Accurately analyze and simulate gcode tool-path for plotters, CNCs, 3D printers and more.

This package provides tools for simulating and visualizing GCode execution,
taking into account machine acceleration and other parameters.
"""

__version__ = "0.2.1"

from .gcode_simulator import GCodeSimulator, GrblSettings
from .viz import plot_trace

__all__ = ["GCodeSimulator", "GrblSettings", "plot_trace"]