from __future__ import annotations

__all__ = ["Image"]

from pathlib import Path
from typing import NoReturn, cast

from ._logging import logger
from .colourspace import ASSColor, Opacity
from .geometry import PointCartesian2D
from .ptypes import AnyPath
from .shape import Pixel


class Image:
    path: Path

    def __init__(self, image: AnyPath) -> None:
        self.path = Path(image)

    @logger.catch
    def to_ass(self) -> NoReturn:
        raise NotImplementedError

    def to_pixels(self) -> list[Pixel]:
        """
        Convert current image file to a list of Pixel
        It is strongly recommended to create a dedicated style for pixels,
        thus, you will write less tags for line in your pixels,
        which means less size for your .ass file.

        Style suggested as an=7, bord=0, shad=0

        :return:            List of Pixel
        """
        from skimage.io import imread as skimage_imread

        img_rgb = skimage_imread(str(self.path))
        rows, columns = img_rgb.shape[:2]
        # Handle grayscale vs color
        if img_rgb.ndim == 2:
            return [
                Pixel(
                    PointCartesian2D(float(co), float(ro)),
                    Opacity(1.0),
                    ASSColor(cast(tuple[str, str, str], (int(img_rgb[ro, co]),) * 3)),
                )
                for ro in range(rows)
                for co in range(columns)
            ]

        # Handle RGB/RGBA
        return [
            Pixel(
                PointCartesian2D(float(co), float(ro)),
                Opacity(1.0),
                # skimage returns RGB, ASSColor expects (B,G,R) or similar based on existing cv2 usage
                ASSColor((int(img_rgb[ro, co, 2]), int(img_rgb[ro, co, 1]), int(img_rgb[ro, co, 0]))),
            )
            for ro in range(rows)
            for co in range(columns)
        ]
