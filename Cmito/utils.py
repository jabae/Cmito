# Import necessary packages
import itertools
import operator


# Chunk boundaries with overlap
def chunk_bboxes_overlap(vol_size, chunk_size, overlap=(0, 0, 0), offset=None, mip=0):

    if mip > 0:
        mip_factor = 2 ** mip
        vol_size = (vol_size[0]//mip_factor,
                    vol_size[1]//mip_factor,
                    vol_size[2])

        chunk_size = (chunk_size[0]//mip_factor,
                      chunk_size[1]//mip_factor,
                      chunk_size[2])

        overlap = (overlap[0]//mip_factor,
                   overlap[1]//mip_factor,
                   overlap[2])

        if offset is not None:
            offset = (offset[0]//mip_factor,
                      offset[1]//mip_factor,
                      offset[2])

    x_bnds = bounds1D_overlap(vol_size[0], chunk_size[0], overlap[0])
    y_bnds = bounds1D_overlap(vol_size[1], chunk_size[1], overlap[1])
    z_bnds = bounds1D_overlap(vol_size[2], chunk_size[2], overlap[2])

    bboxes = [tuple(zip(xs, ys, zs))
              for (xs, ys, zs) in itertools.product(x_bnds, y_bnds, z_bnds)]

    if offset is not None:
        bboxes = [(tuple(map(operator.add, bb[0], offset)),
                   tuple(map(operator.add, bb[1], offset)))
                  for bb in bboxes]

    return bboxes


def chunk_bboxes_2d(vol_size, chunk_size, overlap=(0, 0), offset=None, mip=0):

    if mip > 0:
        mip_factor = 2 ** mip
        vol_size = (vol_size[0]//mip_factor,
                    vol_size[1]//mip_factor)

        chunk_size = (chunk_size[0]//mip_factor,
                      chunk_size[1]//mip_factor)

        overlap = (overlap[0]//mip_factor,
                   overlap[1]//mip_factor)

        if offset is not None:
            offset = (offset[0]//mip_factor,
                      offset[1]//mip_factor)

    x_bnds = bounds1D_overlap(vol_size[0], chunk_size[0], overlap[0])
    y_bnds = bounds1D_overlap(vol_size[1], chunk_size[1], overlap[1])
    
    bboxes = [tuple(zip(xs, ys))
              for (xs, ys) in itertools.product(x_bnds, y_bnds)]

    if offset is not None:
        bboxes = [(tuple(map(operator.add, bb[0], offset)),
                   tuple(map(operator.add, bb[1], offset)))
                  for bb in bboxes]

    return bboxes


# Boundaries with overlap 1D
def bounds1D_overlap(full_width, step_size, overlap=0):

    assert step_size > 0, "invalid step_size: {}".format(step_size)
    assert full_width > 0, "invalid volume_width: {}".format(full_width)
    assert overlap >= 0, "invalid overlap: {}".format(overlap)

    start = 0
    end = step_size

    bounds = []
    while end < full_width:
        bounds.append((start, end))

        start += step_size - overlap
        end = start + step_size

    # last window
    end = full_width
    bounds.append((start, end))

    return bounds