import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt
from typing import cast
import scipy.ndimage as nd

def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
  """
  Takes a mesh and centers and normalizes it
  
  :param mesh: Mesh to normalize
  :type mesh: trimesh.Trimesh
  :returns: A mesh centered at the origin (0, 0, 0) scaled down to fit in 1 unit
  :rtype: Trimesh
  """
  # Canter mesh
  center = mesh.centroid
  mesh.apply_translation(-center)
  
  # Scale it down to a scale of 1
  max_extent = np.max(mesh.extents)
  if max_extent > 0:
      mesh.apply_scale(1.0 / max_extent)
  return mesh

def loadNormalizedMesh(objectPath: str) -> trimesh.Trimesh:
  """
  Loads an object from a file, handles Scene/Mesh differences,
  and returns a centered, unit-scaled Trimesh object.

  1. Load mesh from path
  2. Cast to Trimesh
  3. Normalize mesh
  
  :param objectPath: Relative path to .obj file
  :type objectPath: str
  :returns: A mesh normalized and centered at origin
  :rtype: Trimesh
  """
  # 1. Load mesh from path
  loaded_data = trimesh.load(objectPath)

  loaded_data = trimesh.creation.icosphere()

  # Unwrap scene if multiple objects
  # THIS SHOULDN"T HAPPEN since the .obj files should
  # be one object not a scene
  if isinstance(loaded_data, trimesh.Scene):
      print(f"Collapsed Scene into single Mesh: {objectPath}")
      mesh = loaded_data.dump(concatenate=True)
  else:
      mesh = loaded_data

  # 2. Cast
  mesh = cast(trimesh.Trimesh, mesh)

  if len(mesh.vertices) == 0:
    raise ValueError(f"Error: The mesh '{objectPath}' has 0 vertices! It is empty.")
  
  print(f"Sanitizing mesh... ({len(mesh.vertices)} vertices)")
  mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

  # 3. Normalize and center
  normalizedMesh = normalize_mesh(mesh)
  
  return normalizedMesh

def look_at(eye: list[float] | np.ndarray, 
            target: list[float] | np.ndarray, 
            up: list[float] | np.ndarray = [0, 0, 1]) -> np.ndarray:
  """
  Returns a 4x4 Camera View Matrix (Extrinsic Matrix) looking at a specific target.

  1. Construct x, y, z axis normals to the camera "box"
  2. Add to camera matrix

  :param eye: The [x, y, z] position of the camera
  :type eye: list[float] | np.ndarray
  :param target: The [x, y, z] point the camera should look at.
  :type target: list[float] | np.ndarray
  :param up: The "Up" vector for the camera (default is Z-axis [0, 0, 1]).
              This prevents the camera from rolling sideways.
  :type up: list[float] | np.ndarray
  :return: A 4x4 homogenous transformation matrix representing the camera pose.
  :rtype: np.ndarray
  """
  eye = np.array(eye).astype(float)
  target = np.array(target).astype(float)
  up = np.array(up).astype(float)

  # 1. Construct Camera rotation vectors
  # Forward Vector (Z-axis)
  # Camera looks down -Z, so Z points FROM target TO eye.
  z_axis = eye - target      
  z_axis = z_axis / np.linalg.norm(z_axis)
  # Right Vector (X-axis)
  x_axis = np.cross(up, z_axis) 
  x_axis = x_axis / np.linalg.norm(x_axis)
  # Up Vector (Y-axis)
  y_axis = np.cross(z_axis, x_axis)
  
  # Construct camera matrix
  mat = np.eye(4)
  mat[:3, 0] = x_axis  # Column 0: Right
  mat[:3, 1] = y_axis  # Column 1: Up
  mat[:3, 2] = z_axis  # Column 2: Forward
  mat[:3, 3] = eye     # Column 3: Position
  
  return mat

def render_mesh(mesh: trimesh.Trimesh, eye_pos=[1.5, 1.5, 1.5], target_pos=[0,0,0]):
  """
  Renders the mesh from a specific angle and returns the 2D image and Z-buffer.
  
  1. Convert mesh to pyrender mesh
  2. Setup camera to look at object
  3. Initialize light
  4. Render scene and capture color and depth map

  :param mesh: The normalized Trimesh object
  :param eye_pos: Where the camera is [x, y, z]
  :param target_pos: What the camera is looking at [x, y, z]
  :returns: tuple (color_image, depth_map)
  """
  # 1. Convert to Pyrender Object
  mesh_render = pyrender.Mesh.from_trimesh(mesh)
  scene = pyrender.Scene(bg_color=[255, 255, 255]) # White background easier masking
  scene.add(mesh_render)

  # 2. Setup Camera using look_at()
  # YFOV = ~60 degrees 
  camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
  camera_pose = look_at(eye_pos, target_pos)
  scene.add(camera, pose=camera_pose)

  # 3. Setup Light (Attached to Camera)
  # Intensity=3.0 This is good enough
  light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                              innerConeAngle=np.pi/16.0,
                              outerConeAngle=np.pi/6.0)
  scene.add(light, pose=camera_pose)

  # 4. Render scene and capture color and depth map
  # 512x512 is reccomended by stable diffusion
  r = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=512)
  color, depth = r.render(scene) # type: ignore Not sure why it's giving error here
  
  # Free memory
  r.delete()
  
  return color, depth

def extract_features(image_2d, depth_map):
  """
  Gets the rim mask (occluding contour), centroid, rim eccentricty, and
  rim depth of the 2d image of the object from the depth map

  1. Creates the silhoutte (occluding contour)
  2. Finds the center of mass of silhoutte
  3. Gets the rim from silhoutte
  4. Calculate eccentricity at points
  5. Extract rim depth and eccentricity map
  
  :param image_2d: (H, W, 3) RGB image from the renderer
  :param depth_map: (H, W) Float array of distances
  :returns: A dictionary containing masks and the raw paired data for plotting
  """
  # 1. Create the Silhouette Mask
  # This squishes the image into two dimensions
  # then sums up the color value of each pixel.
  # If the pixel isn't white i.e all values sum
  # aren't 255 * 3 = 765 then it's the object.
  mask = np.sum(image_2d, axis=2) < 765
  
  if not np.any(mask):
    print("Warning: No object detected in image!")
    return None

  # 2. Find the Center of Mass (Centroid) of the Silhouette
  # We calculate the average x and y position of all black pixels
  y_coords, x_coords = np.where(mask)
  centroid_y = np.mean(y_coords)
  centroid_x = np.mean(x_coords)
  
  # 3. Detect the Rim
  # We shrink the mask by 2 pixels and subtract it from the original.
  # Original - Shrunken = The Border.
  shrunk_1 = nd.binary_erosion(mask, iterations=1)
  shrunk_2 = nd.binary_erosion(mask, iterations=2)
  rim_mask = np.logical_xor(shrunk_1, shrunk_2)
  # Odd depth = 0 points messed up map
  rim_mask = rim_mask & (depth_map > 0)
  
  # 4. Calculate "Eccentricity" (Squared Radial Distance)
  # Create a grid of coordinate values for the whole image
  H, W = mask.shape
  x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))

  # Normalize from pixels to -1 to 1
  x_norm = (x_grid - centroid_x) / (W / 2.0)
  y_norm = (y_grid - centroid_y) / (H / 2.0)
  
  # Calculate distance (x - cx)^2 + (y - cy)^2 for every pixel
  # This is more hijacking from numpy
  dist_sq_map = x_norm**2 + y_norm**2
  
  # 5. Extract Data Points
  # get depth and eccentricity at the rim pixels
  rim_eccentricity = dist_sq_map[rim_mask]
  rim_depth = depth_map[rim_mask]
  
  # Some aliasing issues causing points
  # in the background to be part of the
  # silhoutte so only include points
  # Whose distance is greater than 0
  # valid_points = rim_depth > 0

  # rim_eccentricity = rim_eccentricity[valid_points]
  # rim_depth = rim_depth[valid_points]
  
  return {
    "rim_mask": rim_mask,
    "centroid": (centroid_x, centroid_y),
    "eccentricity_map": dist_sq_map,
    "data_x": rim_eccentricity, # The r^2 values
    "data_y": rim_depth         # The Depth values
  }

def plot_binned_trend(features):
  """
  Collapses the noisy scatter plot into a clean trend line by 
  calculating the average Depth for each Eccentricity 'bin'.
  """
  x = features['data_x']
  y = features['data_y']
  
  # 1. Create Bins (0.0 to 1.0 with step 0.02)
  bins = np.arange(0, np.max(x) + 0.02, 0.02)
  
  # 2. Calculate the Mean Depth for each bin
  # 'digitize' tells us which bin each pixel belongs to
  indices = np.digitize(x, bins)
  
  mean_depths = []
  mean_ecc = []
  
  for i in range(1, len(bins)):
    # Find all points in this bin
    mask = indices == i
    if np.sum(mask) > 0:
      # Calculate mean depth for this slice
      mean_depths.append(np.mean(y[mask]))
      # Calculate mean eccentricity (center of bin)
      mean_ecc.append(bins[i-1] + 0.01)
          
  # 3. Plot it
  plt.figure(figsize=(10, 6))
  
  # Background: The Raw Scatter (faint)
  plt.scatter(x, y, alpha=0.1, s=1, c='gray', label='Raw Data (Pixels)')
  
  # Foreground: The Trend Line
  plt.plot(mean_ecc, mean_depths, 'r-', linewidth=3, marker='o', label='Mean Trend (Paper Style)')
  
  # Aesthetics
  plt.title("Binned Trend: Eccentricity vs Depth")
  plt.xlabel("Squared Eccentricity (r^2)")
  plt.ylabel("Depth (Z)")
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.show(block=True)

def objectAnalyzer(objectPath: str):
  # 1. Load Data
  normalizedMesh = loadNormalizedMesh(objectPath)

  # 2. Define Camera (Isometric-ish view)
  cam_position = [1.2, 0.3, 0.3]
  
  # 3. Render
  image_2d, depth_map_ground_truth = render_mesh(normalizedMesh, eye_pos=cam_position)

  # 4. Feature Extraction
  features = extract_features(image_2d, depth_map_ground_truth)
  
  if features is None: return

  # --- VISUAL DEBUGGER ---
  # This is very hard to visualize in my head
  plt.figure(figsize=(18, 5))

  # PANEL 1: The Depth Map
  plt.subplot(1, 3, 1)
  
  # Mask out the background (Depth=0) so it appears white
  masked_depth = np.ma.masked_where(depth_map_ground_truth == 0, depth_map_ground_truth)
  
  plt.imshow(masked_depth, cmap='turbo')
  plt.colorbar(label="Depth (Meters)")
  
  # Draw Crosshair centroid
  cx, cy = features['centroid']
  plt.plot(cx, cy, 'w+', markersize=5, markeredgewidth=2, label='Centroid')
  plt.legend(loc='upper right')
  plt.title("1. Depth Heatmap")
  plt.xlabel("Blue = Close to Camera | Red = Far Away")

  # PANEL 2: The Masked Rim
  plt.subplot(1, 3, 2)
  
  # Create the masked array for the rim
  visual_debug = np.zeros_like(features['eccentricity_map'])
  visual_debug[features['rim_mask']] = features['data_x']
  visual_debug_masked = np.ma.masked_where(~features['rim_mask'], visual_debug)
  
  plt.imshow(visual_debug_masked, cmap='turbo', interpolation='nearest')
  plt.title("2. Rim Eccentricity")
  plt.xlabel("Blue = Center of Image | Red = Edge of Image")

  # PANEL 3: The Scatter Plot
  plt.subplot(1, 3, 3)
  plt.scatter(features['data_x'], features['data_y'], 
                c=features['data_x'], cmap='turbo', 
                alpha=0.5, s=5)
  
  plt.xlabel("Eccentricity (Normalized r^2)")
  plt.ylabel("Depth (Meters)")
  plt.title("3. Width vs Depth")
  plt.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.show(block=True)

  plot_binned_trend(features)
  
  return features

def main():
  mesh_path = "./objects/cow.obj" # Change to use different objects

  objectAnalyzer(mesh_path)

if __name__ == "__main__":
  main()