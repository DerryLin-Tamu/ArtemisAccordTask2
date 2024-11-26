import matplotlib.pyplot as plt
import numpy as np
from compute_homography import compute_transforms, apply_forward, apply_reverse
from PIL import Image

def generate_spiral_waypoints(corner0, corner1, corner2, corner3, step_size = 40, threshold = 1.5):
    """
    Returns waypoints for a square spiral inward in the direction based on the order of the corners provided.

    The `threshold` variable is a factor used to determine the minimum distance between two waypoints for an
    intermediate waypoint to be added between them.
    """
    side0 = np.array(corner1) - np.array(corner0)
    dim0 = np.linalg.norm(side0) # first dimension of the spiral
    u = side0 / dim0 # unit vector for first direction
    side1 = np.array(corner2) - np.array(corner1)
    dim1 = np.linalg.norm(side1) # second dimension of the spiral
    v = side1 / dim1 # unit vector for second direction
    waypoints = [corner0]

    pos = np.array(corner0)
    steps0 = abs(dim0 / step_size)
    steps1 = abs(dim1 / step_size)
    steps = (steps0 + steps1) / 2 # assume we're searching a roughly square area

    direction = 0
    for i in range(int(steps)):
        for _ in range(2):  # Two lines per step increment
            w = None
            side = None
            dir = 1
            if direction % 4 == 0:
                w = u
                side = side0
            elif direction % 4 == 1:
                w = v
                side = side1
            elif direction % 4 == 2:
                w = u
                side = side0
                dir = -1
            elif direction % 4 == 3:
                w = v
                side = side1
                dir = -1
            target = pos + dir * (side - w * step_size * i)
            while np.linalg.norm(target - pos) > step_size * threshold:
                pos += dir * w * step_size
                waypoints.append(pos.tolist())
            waypoints.append(target.tolist())
            pos = target
            direction += 1
    
    return waypoints

def plot_search(path, grid_size_km=8):

    grid_size_m = grid_size_km * 1000
    lla_tr = np.array([-89.33738, -133.78184])
    lla_br = np.array([-89.40591, -133.34033])
    lla_bl = np.array([-89.40708, -139.93771])
    lla_tl = np.array([-89.33843, -139.69528])

    filename = "lunar_sar_aoi_hires.png" # This is Calvin's rotated image
    image = Image.open(filename)
    image_array = np.array(image, dtype=np.uint16)
    img = image_array.copy()
    height, width = img.shape[:2]
    qTL = np.array([0, 0], dtype="float32")
    qTR = np.array([0, width-1], dtype="float32")
    qBR = np.array([height-1, width-1], dtype="float32")
    qBL = np.array([height-1, 0], dtype="float32")
    ll_search_tl = np.array([-89.39, -134.5], dtype="float32")
    ll_search_tr = np.array([-89.34, -134.5], dtype="float32")
    ll_search_br = np.array([-89.34, -139.0], dtype="float32")
    ll_search_bl = np.array([-89.39, -139.0], dtype="float32")
    H_ll_px, H_px_ll = compute_transforms(lla_tl, lla_tr, lla_br, lla_bl, qTL, qTR, qBR, qBL)
    xy_search_tl = [-317.154, 735.680]
    xy_search_tr = [1199.211, 795.979]
    xy_search_br = [1199.962, -776.878]
    xy_search_bl = [-316.460, -718.026]
    H_xy_ll, H_ll_xy = compute_transforms(xy_search_tl, xy_search_tr, xy_search_br, xy_search_bl, ll_search_tl, ll_search_tr, ll_search_br, ll_search_bl)
    path_ll = [apply_forward(H_xy_ll, point) for point in path]
    path_px = [apply_forward(H_ll_px, point) for point in path_ll]

    # Grab x and y coordinates 
    x_coord = [point[0] for point in path_px]
    y_coord = [point[1] for point in path_px]

    crater_px = apply_forward(H_ll_px, np.array([-89.36867, -136.51529], dtype="float32"))
    spawn_px = apply_forward(H_ll_px, np.array([-89.405715, -134.533586], dtype="float32"))
    square1_tl_px = np.array([220, 355], dtype="float32")
    square1_tr_px = apply_forward(H_ll_px, np.array([-89.39, -134.5], dtype="float32"))
    square1_br_px = apply_forward(H_ll_px, np.array([-89.38819, -135.741814], dtype="float32"))
    square1_bl_px = np.array([215, 260], dtype="float32")
    square1_px = [square1_tr_px, square1_tl_px, square1_bl_px, square1_br_px]
    square1_ll = [apply_reverse(H_px_ll, point) for point in square1_px]
    square1_xy = [apply_reverse(H_ll_xy, point) for point in square1_ll]
    for point in square1_xy:
        print(point)
    pois = [crater_px, spawn_px, square1_tl_px, square1_tr_px, square1_br_px, square1_bl_px]

    square1 = generate_spiral_waypoints(square1_tr_px, square1_tl_px, square1_bl_px, square1_br_px)

    plt.figure(figsize=(8, 8))  # 8x8 km grid
    plt.imshow(img)
    plt.scatter(x_coord, y_coord, s=0.5)
    plt.scatter([point[0] for point in pois], [point[1] for point in pois], s=2, c='#ff0000')
    plt.scatter([point[0] for point in square1], [point[1] for point in square1], s=2, c='#00ff00')
    plt.title("Square Search Path")
    plt.xlabel("X Coordinate (meters)")
    plt.ylabel("Y Coordinate (meters)")

    plt.xlim(0, grid_size_m)
    plt.ylim(0, grid_size_m)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# path = generate_spiral_waypoints([-317.154, 735.680], [1199.211, 795.979], [1199.962, -776.878], [-316.460, -718.026], 100)
# plot_search(path)