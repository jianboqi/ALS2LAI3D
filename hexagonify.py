from PIL import Image, ImageDraw
import numpy as np
import math
import random


def hexagon_corners(center, size):
    x = center[0]
    y = center[1]

    w = math.sqrt(3) * size
    h = 2 * size

    return [
        (x - w / 2, y - h / 4),
        (x, y - h / 2),
        (x + w / 2, y - h / 4),
        (x + w / 2, y + h / 4),
        (x, y + h / 2),
        (x - w / 2, y + h / 4)
    ]


def rectangle_corners(center, w, h):
    x = center[0]
    y = center[1]

    return [
        (x - w / 2, y - h / 2),
        (x + w / 2, y - h / 2),
        (x + w / 2, y + h / 2),
        (x - w / 2, y + h / 2)
    ]


def hexagonify(image, hexagon_size):
    img_w, img_h = image.shape[1], image.shape[0]

    draw_image = Image.new("I", (img_w, img_h))
    draw = ImageDraw.Draw(draw_image)

    w = math.sqrt(3) * hexagon_size
    h = 2 * hexagon_size

    # numer of hexagons horizontally and vertically
    num_hor = int(img_w / w) + 2
    num_ver = int(img_h / h * 4 / 3) + 2

    pixel_id = 1
    for i in range(0, num_hor * num_ver):
        column = i % num_hor
        row = i // num_hor
        even = row % 2  # the even rows of hexagons has w/2 offset on the x-axis compared to odd rows.

        p = hexagon_corners((column * w + even * w / 2, row * h * 3 / 4), hexagon_size)

        # compute the average color of the hexagon, use a rectangle approximation.
        raw = rectangle_corners((column * w + even * w / 2, row * h * 3 / 4), w, h)
        r = []
        for points in raw:
            np0 = int(np.clip(points[0], 0, img_w))
            np1 = int(np.clip(points[1], 0, img_h))
            r.append((np0, np1))
        slice = image[r[0][1]:r[3][1], r[0][0]:r[1][0]]
        if not slice.size == 0:
            color = np.average(slice, axis=(0, 1))
            if color != 0:
                draw.polygon(p, fill=pixel_id)
                pixel_id += 1
    return np.array(draw_image)


if __name__ == "__main__":
    image = np.random.rand(500, 500)
    im = hexagonify(image, 10)
    import matplotlib.pyplot as plt
    plt.imshow(im)
    plt.show()