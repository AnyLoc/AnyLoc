
import matplotlib
matplotlib.use('Agg')
import os
import cv2
import math
import numpy as np
from glob import glob
from skimage import io
from os.path import join
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import defaultdict
from staticmap import StaticMap, Polygon
from matplotlib.colors import ListedColormap


def _lon_to_x(lon, zoom):
    if not (-180 <= lon <= 180): lon = (lon + 180) % 360 - 180
    return ((lon + 180.) / 360) * pow(2, zoom)


def _lat_to_y(lat, zoom):
    if not (-90 <= lat <= 90): lat = (lat + 90) % 180 - 90
    return (1 - math.log(math.tan(lat * math.pi / 180) + 1 / math.cos(lat * math.pi / 180)) / math.pi) / 2 * pow(2, zoom)


def _download_map_image(min_lat=45.0, min_lon=7.6, max_lat=45.1, max_lon=7.7, size=2000):
    """"Download a map of the chosen area as a numpy image"""
    mean_lat = (min_lat + max_lat) / 2
    mean_lon = (min_lon + max_lon) / 2
    static_map = StaticMap(size, size)
    static_map.add_polygon(
        Polygon(((min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat)), None, '#FFFFFF'))
    zoom = static_map._calculate_zoom()
    static_map = StaticMap(size, size)
    image = static_map.render(zoom, [mean_lon, mean_lat])
    print(
        f"You can see the map on Google Maps at this link www.google.com/maps/place/@{mean_lat},{mean_lon},{zoom - 1}z")
    min_lat_px, min_lon_px, max_lat_px, max_lon_px = \
        static_map._y_to_px(_lat_to_y(min_lat, zoom)), \
        static_map._x_to_px(_lon_to_x(min_lon, zoom)), \
        static_map._y_to_px(_lat_to_y(max_lat, zoom)), \
        static_map._x_to_px(_lon_to_x(max_lon, zoom))
    assert 0 <= max_lat_px < min_lat_px < size and 0 <= min_lon_px < max_lon_px < size
    return np.array(image)[max_lat_px:min_lat_px, min_lon_px:max_lon_px], static_map, zoom


def get_edges(coordinates, enlarge=0):
    """
    Send the edges of the coordinates, i.e. the most south, west, north and
        east coordinates.
    :param coordinates: A list of numpy.arrays of shape (Nx2)
    :param float enlarge: How much to increase the coordinates, to enlarge
        the area included between the points
    :return: a tuple with the four float
    """
    min_lat, min_lon, max_lat, max_lon = (*np.concatenate(coordinates).min(0), *np.concatenate(coordinates).max(0))
    diff_lat = (max_lat - min_lat) * enlarge
    diff_lon = (max_lon - min_lon) * enlarge
    inc_min_lat, inc_min_lon, inc_max_lat, inc_max_lon = \
        min_lat - diff_lat, min_lon - diff_lon, max_lat + diff_lat, max_lon + diff_lon
    return inc_min_lat, inc_min_lon, inc_max_lat, inc_max_lon


def _create_map(coordinates, colors=None, dot_sizes=None, legend_names=None, map_intensity=0.6):
    dot_sizes = dot_sizes if dot_sizes is not None else [10] * len(coordinates)
    colors = colors if colors is not None else ["r"] * len(coordinates)
    assert len(coordinates) == len(dot_sizes) == len(colors), \
        f"The number of coordinates must be equals to the number of colors and dot_sizes, but they're " \
        f"{len(coordinates)}, {len(colors)}, {len(dot_sizes)}"

    # Add two dummy points to slightly enlarge the map
    min_lat, min_lon, max_lat, max_lon = get_edges(coordinates, enlarge=0.1)
    coordinates.append(np.array([[min_lat, min_lon], [max_lat, max_lon]]))
    # Download the map of the chosen area
    map_img, static_map, zoom = _download_map_image(min_lat, min_lon, max_lat, max_lon)

    scatters = []
    fig = plt.figure(figsize=(map_img.shape[1] / 100, map_img.shape[0] / 100), dpi=1000)
    for i, coord in enumerate(coordinates):
        for i in range(len(coord)):  # Scale latitudes because of earth's curvature
            coord[i, 0] = -static_map._y_to_px(_lat_to_y(coord[i, 0], zoom))
    for coord, size, color in zip(coordinates, dot_sizes, colors):
        scatters.append(plt.scatter(coord[:, 1], coord[:, 0], s=size, color=color))

    if legend_names != None:
        plt.legend(scatters, legend_names, scatterpoints=10000, loc='lower left',
                   ncol=1, framealpha=0, prop={"weight": "bold", "size": 30})

    min_lat, min_lon, max_lat, max_lon = get_edges(coordinates)
    plt.ylim(min_lat, max_lat)
    plt.xlim(min_lon, max_lon)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    fig.canvas.draw()
    plot_img = np.array(fig.canvas.renderer._renderer)
    plt.close()

    plot_img = cv2.resize(plot_img[:, :, :3], map_img.shape[:2][::-1], interpolation=cv2.INTER_LANCZOS4)
    map_img[(map_img.sum(2) < 444)] = 188  # brighten dark pixels
    map_img = (((map_img / 255) ** map_intensity) * 255).astype(np.uint8)  # fade map
    mask = (plot_img.sum(2) == 255 * 3)[:, :, None]  # mask of plot, to find white pixels
    final_map = map_img * mask + plot_img * (~mask)
    return final_map


def _get_coordinates_from_dataset(dataset_folder, extension="jpg"):
    """
    Takes as input the path of a dataset, such as "datasets/st_lucia/images"
    and returns
        [("train/database", [[45, 8.1], [45.2, 8.2]]), ("train/queries", [[45, 8.1], [45.2, 8.2]])]
    """
    images_paths = glob(join(dataset_folder, "**", f"*.{extension}"), recursive=True)
    if len(images_paths) != 0:
        print(f"I found {len(images_paths)} images in {dataset_folder}")
    else:
        raise ValueError(f"I found no images in {dataset_folder} !")

    grouped_gps_coords = defaultdict(list)

    for image_path in images_paths:
        full_path = os.path.dirname(image_path)
        full_parent_path, parent_dir = os.path.split(full_path)
        parent_parent_dir = os.path.split(full_parent_path)[1]

        # folder_name is for example "train - database"
        folder_name = " - ".join([parent_parent_dir, parent_dir])

        gps_coords = image_path.split("@")[5], image_path.split("@")[6]
        grouped_gps_coords[folder_name].append(gps_coords)

    grouped_gps_coords = sorted([(k, np.array(v).astype(np.float64))
                                 for k, v in grouped_gps_coords.items()])
    return grouped_gps_coords


def build_map_from_dataset(dataset_folder, dot_sizes=None):
    """dataset_folder is the path that contains the 'images' folder."""
    grouped_gps_coords = _get_coordinates_from_dataset(join(dataset_folder, "images"))
    SORTED_FOLDERS = ["train - database", "train - queries", "val - database", "val - queries",
                      "test - database", "test - queries"]
    try:
        grouped_gps_coords = sorted(grouped_gps_coords, key=lambda x: SORTED_FOLDERS.index(x[0]))
    except ValueError:
        pass  # This dataset has different folder names than the standard train-val-test database-queries.
    coordinates = []
    legend_names = []
    for folder_name, coords in grouped_gps_coords:
        legend_names.append(f"{folder_name} - {len(coords)}")
        coordinates.append(coords)

    colors = cm.rainbow(np.linspace(0, 1, len(legend_names)))
    colors = ListedColormap(colors)
    colors = colors.colors
    if len(legend_names) == 1:
        legend_names = None  # If there's only one folder, don't show the legend
        colors = np.array([[1, 0, 0]])

    map_img = _create_map(coordinates, colors, dot_sizes, legend_names)

    print(f"Map image resolution: {map_img.shape}")
    dataset_name = os.path.basename(os.path.abspath(dataset_folder))
    io.imsave(join(dataset_folder, f"map_{dataset_name}.png"), map_img)
