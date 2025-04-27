import math
from .gcode_simulator import Bounds, TraceNode
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_trace(trace_nodes: list[TraceNode], bounds: Bounds, cmap_name='plasma'):  

  trace_nodes = interpolate_trace_nodes(trace_nodes, max_distance=0.1)

  x = np.array([node.x for node in trace_nodes])
  y = np.array([node.y for node in trace_nodes])
  feeds = np.array([node.feed for node in trace_nodes])
  
  fig, ax = plt.subplots(figsize=(10, 6))
  
  min_x, max_x = int(min(bounds.min_x, x.min()) - 10), int(max(bounds.max_x, x.max()) + 10)
  min_y, max_y = int(min(bounds.min_y, y.min()) - 10), int(max(bounds.max_y, y.max()) + 10)
  
  # Ensure we start and end on multiple of 5 for clean grid
  min_x = 50 * (min_x // 50)
  min_y = 50 * (min_y // 50)
  max_x = 50 * (max_x // 50 + 1)
  max_y = 50 * (max_y // 50 + 1)
  
  # Create minor grid lines (1cm)
  minor_grid_x = np.arange(min_x, max_x + 1, 10)
  minor_grid_y = np.arange(min_y, max_y + 1, 10)
  ax.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.5)
  
  # Create major grid lines (5cm)
  major_grid_x = np.arange(min_x, max_x + 1, 50)
  major_grid_y = np.arange(min_y, max_y + 1, 50)
  ax.grid(True, which='major', color='lightgray', linestyle='-', linewidth=1.0)
  
  ax.set_xticks(major_grid_x, minor=False)
  ax.set_yticks(major_grid_y, minor=False)
  ax.set_xticks(minor_grid_x, minor=True)
  ax.set_yticks(minor_grid_y, minor=True)

  ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='both', steps=[1, 5, 10]))
  ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='both', steps=[1, 5, 10]))
  
  # Create a set of points in the form of (x, y)
  points = np.array([x, y]).T.reshape(-1, 1, 2)
  
  # Create line segments
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  
  norm = plt.Normalize(feeds.min(), feeds.max())
  
  lc = LineCollection(segments, cmap=plt.get_cmap(cmap_name), norm=norm, linewidth=2)
  
  # Set the values used for colormapping
  lc.set_array(feeds)
  
  line = ax.add_collection(lc)
  
  cbar = plt.colorbar(line, ax=ax)
  cbar.set_label('Feed Rate')
  
  ax.set_xlim(min_x, max_x)
  ax.set_ylim(min_y, max_y)
  ax.set_xlabel('X (mm)')
  ax.set_ylabel('Y (mm)')

  # sc = ax.scatter(x, y, c=feeds, cmap=cmap, s=30, zorder=3)

  # enforce aspect ratio
  plt.axis('scaled')

  plt.tight_layout()
  plt.show()


def interpolate_trace_nodes(trace_nodes: list[TraceNode], max_distance: float = 1.0) -> list[TraceNode]:
  """
  Interpolate the trace nodes so that each node is at most max_distance apart,
  evenly spread along the path between two consecutive nodes.

  The feed at any location is linearly interpolated.
  
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
      
      dx = end_node.x - start_node.x
      dy = end_node.y - start_node.y
      distance = math.sqrt(dx*dx + dy*dy)
      
      time_diff = end_node.time - start_node.time
      
      if distance <= max_distance:
          interpolated_nodes.append(end_node)
          continue
      
      num_segments = math.ceil(distance / max_distance)
      
      for j in range(1, num_segments):
          fraction = j / num_segments
          
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
      
      interpolated_nodes.append(end_node)
  
  return interpolated_nodes