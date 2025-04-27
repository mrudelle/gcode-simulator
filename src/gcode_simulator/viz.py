import math
from gcode_simulator import Bounds, TraceNode
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_trace(trace_nodes: list[TraceNode], bounds: Bounds, cmap_name='plasma'):  

  trace_nodes = interpolate_trace_nodes(trace_nodes, max_distance=0.1)

  # Extract coordinates and feed values
  x = np.array([node.x for node in trace_nodes])
  y = np.array([node.y for node in trace_nodes])
  feeds = np.array([node.feed for node in trace_nodes])
  
  # Create figure and axis
  fig, ax = plt.subplots(figsize=(10, 6))
  
  # Add grid lines
  # Calculate grid line positions with a padding of 10mm
  min_x, max_x = int(min(bounds.min_x, x.min()) - 10), int(max(bounds.max_x, x.max()) + 10)
  min_y, max_y = int(min(bounds.min_y, y.min()) - 10), int(max(bounds.max_y, y.max()) + 10)
  
  # Ensure we start and end on multiple of 5 for clean grid
  min_x = 5 * (min_x // 5)
  min_y = 5 * (min_y // 5)
  max_x = 5 * (max_x // 5 + 1)
  max_y = 5 * (max_y // 5 + 1)
  
  # Create minor grid lines (1cm)
  minor_grid_x = np.arange(min_x, max_x + 1, 1)
  minor_grid_y = np.arange(min_y, max_y + 1, 1)
  ax.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.5)
  
  # Create major grid lines (5cm)
  major_grid_x = np.arange(min_x, max_x + 1, 5)
  major_grid_y = np.arange(min_y, max_y + 1, 5)
  ax.grid(True, which='major', color='lightgray', linestyle='-', linewidth=1.0)
  
  # Set both minor and major ticks
  ax.set_xticks(major_grid_x, minor=False)
  ax.set_yticks(major_grid_y, minor=False)
  ax.set_xticks(minor_grid_x, minor=True)
  ax.set_yticks(minor_grid_y, minor=True)

  # Automatically adjust the spacing of major tick labels
  ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='both'))
  ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='both'))
  
  # Create a set of points in the form of (x, y)
  points = np.array([x, y]).T.reshape(-1, 1, 2)
  
  # Create line segments
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  
  # Create a continuous norm to map from feed values to colors
  norm = plt.Normalize(feeds.min(), feeds.max())
  
  # Create a LineCollection with the specified colormap
  lc = LineCollection(segments, cmap=plt.get_cmap(cmap_name), norm=norm, linewidth=2)
  
  # Set the values used for colormapping
  lc.set_array(feeds)
  
  # Add the collection to the plot
  line = ax.add_collection(lc)
  
  # Add a colorbar
  cbar = plt.colorbar(line, ax=ax)
  cbar.set_label('Feed Rate')
  
  # Set limits and labels
  ax.set_xlim(min_x, max_x)
  ax.set_ylim(min_y, max_y)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')

  # Add scatter points to see exact positions
  # sc = ax.scatter(x, y, c=feeds, cmap=cmap, s=30, zorder=3)

  plt.axis('scaled')

  plt.tight_layout()
  plt.show()


def interpolate_trace_nodes(trace_nodes: list[TraceNode], max_distance: float = 1.0) -> list[TraceNode]:
  """
  Interpolate the trace nodes so that each node is at most max_distance apart,
  evenly spread along the path between two consecutive nodes.
  
  Parameters:
  - trace_nodes: List of TraceNode objects
  - max_distance: Maximum distance between consecutive nodes in mm
  
  Returns:
  - List of interpolated TraceNode objects
  """
  if len(trace_nodes) < 2:
      return trace_nodes
  
  interpolated_nodes = [trace_nodes[0]]
  
  for i in range(1, len(trace_nodes)):
      start_node = trace_nodes[i-1]
      end_node = trace_nodes[i]
      
      # Calculate the distance between the two nodes
      dx = end_node.x - start_node.x
      dy = end_node.y - start_node.y
      distance = math.sqrt(dx*dx + dy*dy)
      
      # Calculate the time difference between nodes
      time_diff = end_node.time - start_node.time
      
      # If the distance is already less than or equal to max_distance, just add the end node
      if distance <= max_distance:
          interpolated_nodes.append(end_node)
          continue
      
      # Calculate how many points we need to add
      num_segments = math.ceil(distance / max_distance)
      
      # Create interpolated points
      for j in range(1, num_segments):
          fraction = j / num_segments
          
          # Linear interpolation for position, feed rate, and time
          interp_x = start_node.x + dx * fraction
          interp_y = start_node.y + dy * fraction
          interp_feed = start_node.feed + (end_node.feed - start_node.feed) * fraction
          interp_time = start_node.time + time_diff * fraction
          
          interpolated_node = TraceNode(
              x=interp_x,
              y=interp_y,
              feed=interp_feed,
              time=interp_time
          )
          interpolated_nodes.append(interpolated_node)
      
      # Add the end node
      interpolated_nodes.append(end_node)
  
  return interpolated_nodes