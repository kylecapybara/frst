"""
Fast Radial Symmetry Transform [1] for bead detection.

should be a drop-in alternative to cv2.HoughCircles;
returns circles as an (N, 3) float32 array of (x, y, r),

[1] G. Loy and A. Zelinsky, “A Fast Radial Symmetry Transform for Detecting Points of Interest,” in Computer Vision — ECCV 2002, vol. 2350, A. Heyden, G. Sparr, M. Nielsen, and P. Johansen, Eds., in Lecture Notes in Computer Science, vol. 2350. , Berlin, Heidelberg: Springer Berlin Heidelberg, 2002, pp. 358–368. doi: 10.1007/3-540-47969-4_24.


@incollection{heyden_fast_2002,
	title = {A Fast Radial Symmetry Transform for Detecting Points of Interest},
	volume = {2350},
	isbn = {978-3-540-43745-1 978-3-540-47969-7},
	doi = {10.1007/3-540-47969-4_24},
	series = {Lecture Notes in Computer Science},
	pages = {358--368},
	booktitle = {Computer Vision — {ECCV} 2002},
	publisher = {Springer Berlin Heidelberg},
	author = {Loy, Gareth and Zelinsky, Alexander},
	editor = {Heyden, Anders and Sparr, Gunnar and Nielsen, Mads and Johansen, Peter},
	editorb = {Goos, Gerhard and Hartmanis, Juris and Van Leeuwen, Jan},
	editorbtype = {redactor},
	date = {2002},
}
"""

import cv2
import numpy as np


def _gradients(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def radial_symmetry_circles(
    gray,
    min_radius=40,
    max_radius=120,
    radius_step=2,
    alpha=2.0,
    gradient_threshold_pct=85.0,
    peak_threshold=0.12,
    min_separation=None,
    max_circles=None,
):
    """Detect circles with FRST.

    gray: uint8 grayscale image. 
    
    Gradients are taken from it directly, 
    so no Canny/edge detection step is needed.
    """
    gray = np.asarray(gray)
    gx, gy = _gradients(gray)
    mag = np.hypot(gx, gy)
    if mag.max() <= 0:
        return np.empty((0, 3), dtype=np.float32)

    # keep only strong-gradient pixels
    # these are the candidate edge points
    thresh = np.percentile(mag[mag > 0], gradient_threshold_pct)
    ys, xs = np.nonzero(mag >= thresh)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    m = mag[ys, xs]
    ux = gx[ys, xs] / m
    uy = gy[ys, xs] / m

    h, w = gray.shape[:2]
    radii = np.arange(int(min_radius), int(max_radius) + 1, int(radius_step))
    if radii.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    kn = float(radii.size)
    acc_shape = (h, w)

    # Orientation projection O_n and magnitude projection M_n per radius.
    # Vote at p + r*g and p - r*g (dark-on-light and light-on-dark beads).
    S = np.zeros(acc_shape, dtype=np.float32)
    best_r = np.zeros(acc_shape, dtype=np.float32)
    best_val = np.zeros(acc_shape, dtype=np.float32)

    for n in radii:
        O = np.zeros(acc_shape, dtype=np.float32)
        M = np.zeros(acc_shape, dtype=np.float32)
        for sign in (1.0, -1.0):
            px = np.round(xs + sign * n * ux).astype(int)
            py = np.round(ys + sign * n * uy).astype(int)
            ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            np.add.at(O, (py[ok], px[ok]), sign)
            np.add.at(M, (py[ok], px[ok]), m[ok])

        # Clip orientation projection to kn, as in the original FRST.
        Ob = np.clip(np.abs(O), 0, kn)
        Fn = (M / max(M.max(), 1e-6)) * (Ob / kn) ** alpha
        sigma = max(0.5, 0.25 * n)
        Fn = cv2.GaussianBlur(Fn, (0, 0), sigma)

        better = Fn > best_val
        best_val[better] = Fn[better]
        best_r[better] = n
        S += Fn

    S = cv2.GaussianBlur(S, (0, 0), 1.0)
    if S.max() <= 0:
        return np.empty((0, 3), dtype=np.float32)

    # Non-maximum suppression over a window about the minimum radius.
    win = max(3, int(min_radius) | 1)
    local_max = cv2.dilate(S, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (win, win)))
    peaks = (S == local_max) & (S >= peak_threshold * S.max())
    py, px = np.nonzero(peaks)
    scores = S[py, px]
    order = np.argsort(-scores)
    px, py, scores = px[order], py[order], scores[order]

    sep = float(min_separation) if min_separation is not None else float(min_radius)
    accepted = []
    for x, y, s in zip(px, py, scores):
        r = best_r[y, x]
        if r <= 0:
            continue
        if any(np.hypot(x - ax, y - ay) < sep for ax, ay, _ in accepted):
            continue
        accepted.append((float(x), float(y), float(r)))
        if max_circles and len(accepted) >= max_circles:
            break

    if not accepted:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(accepted, dtype=np.float32)
