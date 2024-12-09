import argparse
import numpy as np
from stl import mesh
import os

def read_stl(file_path):
    """Reads an STL file and extracts all points."""
    stl_mesh = mesh.Mesh.from_file(file_path)
    points = np.vstack([stl_mesh.v0, stl_mesh.v1, stl_mesh.v2])
    return points

def sample_points(points, n):
    """Samples N unique points from a set of points."""
    n = min(len(points), n)  # Ensure we don't sample more points than available
    indices = np.random.choice(len(points), size=n, replace=False)
    sampled_points = points[indices]
    return sampled_points

def write_stl(points, output_file):
    """Creates an STL file with degenerated mesh from sampled points."""
    # Degenerate triangles (same vertex repeated)
    degenerate_triangles = np.array([[p, p, p] for p in points])
    new_mesh = mesh.Mesh(np.zeros(degenerate_triangles.shape[0], dtype=mesh.Mesh.dtype))
    for i, triangle in enumerate(degenerate_triangles):
        new_mesh.v0[i] = triangle[0]
        new_mesh.v1[i] = triangle[1]
        new_mesh.v2[i] = triangle[2]
    new_mesh.save(output_file)

def main():
    parser = argparse.ArgumentParser(description="Sample points from an STL file and save as a new STL with degenerated mesh.")
    parser.add_argument("stl_file_path", type=str, help="Path to the input STL file")
    parser.add_argument("N", type=int, help="Number of points to sample")
    args = parser.parse_args()

    stl_file_path = args.stl_file_path
    n_points = args.N

    if not os.path.exists(stl_file_path):
        print(f"Error: File {stl_file_path} does not exist.")
        return

    # Process STL file
    print(f"Reading STL file: {stl_file_path}")
    points = read_stl(stl_file_path)
    print(f"Total points in STL: {len(points)}")

    print(f"Sampling {n_points} points...")
    sampled_points = sample_points(points, n_points)

    output_file = f"sampled_{n_points}_points.stl"
    print(f"Saving sampled points to {output_file}")
    write_stl(sampled_points, output_file)
    print("Done.")

if __name__ == "__main__":
    main()
