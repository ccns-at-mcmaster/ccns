"""
Methods to 'mask' a two dimensional array. These methods are intended to be used to exclude data from reduction.
Intended for use at the MacSANS laboratory at McMaster University.

    Author: Devin Burke

(c) Copyright 2022, McMaster University
"""
import numpy as np


def _rectangular_mask(mask, row_width, column_width, origin):
    """
    This acts on an array and sets the values within a rectangular area of pixels to zero. The shape of the rectangle
    is specified by input parameters with an origin point at a corner.

    :param mask: A 2D numpy array.
    :param row_width: The width of the rectangle in pixels. Must be an integer.
    :param column_width: The width of the column in pixels. Must be an integer.
    :param origin: A (row, col) tuple of indices corresponding to the origin of the rectangle, located at one of the
                   corners. The rectangle is 'drawn' outward from this point.
    :return:
    """
    row_width = int(row_width)
    column_width = int(column_width)

    row_indices = range(origin[0], origin[0] + row_width + 1, 1)
    column_indices = range(origin[1], origin[1] + column_width + 1, 1)

    for y, row in enumerate(mask):
        for x, _ in enumerate(row):
            if (x in column_indices) and (y in row_indices):
                mask[y][x] = 0
    return


def _circle_mask(mask, radius, center):
    """
    This acts on an array and sets the values within a circular area of pixels to zero. The shape of the circle
    is specified by input parameters with the origin point at the center.

    :param mask: A 2D numpy array.
    :param radius: The radius of the circle in pixels. Must be an integer.
    :param center: A (row, col) tuple of indices corresponding to the center point of the circle.
    :return:
    """
    radius = float(radius)
    if radius <= 0.0:
        raise Exception('The circle radius must be greater than zero.')
    radius_squared = radius * radius

    for y, row in enumerate(mask):
        for x, _ in enumerate(row):
            dx = x - center[1]
            dy = y - center[0]
            distance_squared = float(dx * dx + dy * dy)

            if distance_squared <= radius_squared:
                mask[y][x] = 0
    return


def _ring_mask(mask, outer_radius, inner_radius, center):
    """
    This acts on an array and sets the values within a ring of pixels to zero. The shape of the ring
    is specified by input parameters with the origin point at the center.

    :param mask: A 2D numpy array.
    :param outer_radius: The outer radius of the ring in pixels. Must be an integer.
    :param inner_radius: The inner radius of the ring in pixels. Must be an integer.
    :param center: A (row, col) tuple of indices corresponding to the center point of the ring.
    :return:
    """
    if inner_radius >= outer_radius:
        raise Exception('The inner radius of the ring mask must be less than the outer radius.')
    inner_radius = float(inner_radius)
    outer_radius = float(outer_radius)
    outer_radius_squared = outer_radius * outer_radius
    inner_radius_squared = inner_radius * inner_radius

    for y, row in enumerate(mask):
        for x, _ in enumerate(row):
            dx = x - center[1]
            dy = y - center[0]
            distance_squared = float(dx * dx + dy * dy)

            if inner_radius_squared <= distance_squared <= outer_radius_squared:
                mask[y][x] = 0
    return


def _irregular_mask(mask, pixels):
    """
    This acts on an array and sets the values within an arbitrary list of pixels to zero.

    :param mask: A 2D numpy array.
    :param pixels: A list of (row, col) tuples whose indices correspond to pixels that will make up the mask feature.
    :return:
    """
    for pixel in pixels:
        if (type(pixel) is tuple) and (type(pixel[0]) is int) and (type(pixel[1]) is int):
            mask[pixel[0]][pixel[1]] = 0
        else:
            raise Exception('The list of masked pixels must be composed of integer (row, col) tuples.')
    return


def get_mask(mask_shape, array_shape=(147, 147), x_width=None, y_width=None, origin=None, inner_radius=None,
             outer_radius=None, irregular_pixels=None):
    """
    This method returns a boolean array of shape array_shape. This boolean array has '0' values within a geometric area
    defined by input parameters in the shape of a rectangle, a circle, or a ring. This is a mask which can be applied to
    a data array to ignore part of the data during reduction. If an 'irregular' mask shape is specified, a mask can be
    returned to exclude an arbitrary list of pixels.

    :param mask_shape: The desired geometric shape of the '0' values of the mask. Must one of 'rectangle', 'circle',
                       'ring', or 'irregular'.
    :param array_shape: The desired shape of the return mask array. This is (147, 147) by default.
    :param x_width: The column width in pixels used to create a rectangular mask. Must be an integer.
    :param y_width: The row width in pixels used to create a rectangular mask. Must be an integer.
    :param origin: A (row, col) tuple of indices corresponding to the placement of the origin point of the mask.
    :param inner_radius: The inner radius in pixels used to create a ring mask. Must be an integer.
    :param outer_radius: The outer radius in pixels used to create a ring mask. Must be an integer.
    :param irregular_pixels: A list of (row, col) tuples containing indices that correspond to pixels. A mask made from
                             this list will return '0' values at each specified pixel.
    :return:
    """

    possible_mask_shapes = ['rectangle', 'circle', 'ring', 'irregular']
    mask = np.ones(array_shape)

    if mask_shape.lower() not in possible_mask_shapes:
        raise Exception("The mask shape must be either 'rectangle', 'circle', 'ring', or 'irregular'.")
    if mask_shape.lower() == 'rectangle':
        if (not y_width) or (not x_width) or (not origin):
            raise Exception("A rectangular mask requires x and y dimensional widths as well as an origin point.")
        _rectangular_mask(mask, y_width, x_width, origin)
        return mask

    if mask_shape.lower() == 'circle':
        if (not outer_radius) or (not origin):
            raise Exception("A circular mask requires a radius and an origin (center) point.")
        _circle_mask(mask, outer_radius, origin)
        return mask

    if mask_shape.lower() == 'ring':
        if (not outer_radius) or (not inner_radius) or (not origin):
            raise Exception("A ring mask requires an inner and outer radius as well as an origin (center) point.")
        _ring_mask(mask, outer_radius, inner_radius, origin)
        return mask

    if mask_shape.lower() == 'irregular':
        if not irregular_pixels:
            raise Exception("An irregular masks requires a list of (row, col) tuples representing pixels.")
        _irregular_mask(mask, irregular_pixels)
        return mask


def apply_mask(data, mask):
    if data.shape == mask.shape:
        for y, row in enumerate(data):
            for x, _ in enumerate(row):
                if not mask[y][x]:
                    data[y][x] = 0
    else:
        raise Exception('The data array and the mask array must have the same shape.')
