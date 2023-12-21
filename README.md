# 3D_restruction

Reconstruct 3D objects in a single view

## Compute Chamfer distance

### 1. Install dependencies

```bash
pip install open3d numpy scipy
```

### 2. You should modify the path in chamfer_distance.py

```python
filename = 'iPad_71'
```

### 3. Run the code

```bash
python chamfer_distance.py
```

This will display the two point clouds and the distance between them. (Orignal, Rotated, Aligned)

### 4. Further Develop

You can modify the code in chamfer_distance_utils.py, mainly the alignment function.
