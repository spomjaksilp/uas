from numba import typed, types

type_coordinate = types.int16[:]
type_coordinate_matrix = types.int16[:, :]
type_path_matrix = types.float32[:, :]
type_site_matrix = types.boolean[:, :]
