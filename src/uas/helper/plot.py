"""

"""
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
# from uas import Lattice


def add_gaussian(image, sigma, amplitude, center):
    x, y = np.mgrid[0:image.shape[0], 0:image.shape[1]].astype(np.float)
    x -= center[0]
    y -= center[1]
    image += amplitude * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return image


def plot_lattice(lattice):
    """

    :param lattice:
    :return:
    """
    # rescale to 0.1µm
    lattice.rescale(np.array((.2e-6, .2e-6)))
    image = np.zeros(lattice.value.shape)
    for coord in lattice.coordinates:
        image = add_gaussian(image=image, sigma=1.5, amplitude=1, center=coord)
    return image


def animate_timeline(timeline):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10, True)
    camera = Camera(fig)
    for timestep in timeline:
        ax.imshow(plot_lattice(timestep["lattice"]))
        ax.text(0, -0.05, f"{timestep['time'] * 1e3:.1f}ms", transform=ax.transAxes)
        camera.snap()
    plt.tight_layout()
    plt.title(f"Completion time {timeline[-1]['time'] * 1e3:.1f}ms")
    animation = camera.animate()
    animation.save('animation.mp4', dpi=240)


