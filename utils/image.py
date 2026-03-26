# pyre-unsafe
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

from tw.camera import CameraTW
from tw.obb import ObbTW
from tw.pose import PoseTW
from tw.tensor_utils import tensor2string, unpad_string

# Some globals for opencv drawing functions.
BLU = (255, 0, 0)
GRN = (0, 255, 0)
RED = (0, 0, 255)
WHT = (255, 255, 255)
BLK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_PT = (5, 15)
FONT_SZ = 0.5
FONT_TH = 1.0


def string2color(string):
    string = string.lower()
    if string == "white":
        return WHT
    elif string == "green":
        return GRN
    elif string == "red":
        return RED
    elif string == "black":
        return BLK
    elif string == "blue":
        return BLU
    else:
        raise ValueError("input color string %s not supported" % string)


def put_text(
    img: np.ndarray,
    text: str,
    scale: float = 1.0,
    line: int = 0,
    # pyre-fixme[9]: color has type `Tuple[tuple[Any, ...], str]`; used as
    #  `Tuple[int, int, int]`.
    color: Tuple[Tuple, str] = WHT,
    font_pt: Optional[Tuple[int, int]] = None,
    # pyre-fixme[9]: truncate has type `int`; used as `None`.
    truncate: int = None,
):
    """Writes text with a shadow in the back at various lines and autoscales it.

    Args:
        image: image HxWx3 or BxHxWx3, should be uint8 for anti-aliasing to work
        text: text to write
        scale: 0.5 for small, 1.0 for normal, 1.5 for big font
        line: vertical line to write on (0: first, 1: second, -1: last, etc)
        color: text color, tuple of BGR integers between 0-255, e.g. (0,0,255) is red,
               can also be a few strings like "white", "black", "green", etc
        truncate: if not None, only show the first N characters
    Returns:
        image with text drawn on it

    """
    if len(img.shape) == 4:  # B x H x W x 3
        for i in range(len(img)):
            img[i] = put_text(img[i], text, scale, line, color, font_pt, truncate)
    else:  # H x W x 3
        if truncate and len(text) > truncate:
            text = text[:truncate] + "..."  # Add "..." to denote truncation.
        height = img.shape[0]
        scale = scale * (height / 320.0)
        wht_th = max(int(FONT_TH * scale), 1)
        blk_th = 2 * wht_th
        text_ht = 15 * scale
        if not font_pt:
            font_pt = int(FONT_PT[0] * scale), int(FONT_PT[1] * scale)
            font_pt = font_pt[0], int(font_pt[1] + line * text_ht)
        if line < 0:
            font_pt = font_pt[0], int(font_pt[1] + (height - text_ht * 0.5))
        cv2.putText(img, text, font_pt, FONT, FONT_SZ * scale, BLK, blk_th, lineType=16)

        if isinstance(color, str):
            color = string2color(color)

        cv2.putText(
            img, text, font_pt, FONT, FONT_SZ * scale, color, wht_th, lineType=16
        )
    return img


def rotate_image90(image: np.ndarray, k: int = 3):
    """Rotates an image and then re-allocates memory to avoid problems with opencv
    Input:
        image: numpy image, HxW or HxWxC
        k: number of times to rotate by 90 degrees counter clockwise
    Returns
        rotated image: numpy image, HxW or HxWxC
    """
    return np.ascontiguousarray(np.rot90(image, k=k))


def normalize(img, robust=0.0, eps=1e-6):
    if isinstance(img, torch.Tensor):
        vals = img.view(-1).cpu().numpy()
    elif isinstance(img, np.ndarray):
        vals = img.flatten()

    if robust > 0.0:
        v_min = np.quantile(vals, robust)
        v_max = np.quantile(vals, 1.0 - robust)
    else:
        v_min = vals.min()
        v_max = vals.max()
    # make sure we are not dividing by 0
    dv = max(eps, v_max - v_min)
    # normalize to 0-1
    img = (img - v_min) / dv
    if isinstance(img, torch.Tensor):
        img = img.clamp(0, 1)
    elif isinstance(img, np.ndarray):
        img = img.clip(0, 1)
    return img


def torch2cv2(
    img: Union[np.ndarray, torch.Tensor],
    rotate: bool = False,
    rgb2bgr: bool = True,
    ensure_rgb: bool = False,
    robust_quant: float = 0.0,
):
    """
    Converts numpy/torch float32 image [0,1] CxHxW to numpy uint8 [0,255] HxWxC

    Args:
        img: image CxHxW float32 image
        rotate: if True, rotate image 90 degrees
        rgb2bgr: convert image to BGR
        ensure_rgb: ensure RGB if True (i.e. replicate the single color channel 3 times)
        robust_quant: quantile to robustly copute min and max for normalization of the image.
    """

    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            img = img[0]
        img = img.data.cpu().numpy()
    if img.ndim == 2:
        img = img[np.newaxis, :, :]

    # CxHxW -> HxWxC
    img = img.transpose(1, 2, 0)
    img_cv2 = (img * 255.0).astype(np.uint8)

    if rgb2bgr:
        img_cv2 = img_cv2[:, :, ::-1]
    if rotate:
        img_cv2 = rotate_image90(img_cv2)
    else:
        img_cv2 = np.ascontiguousarray(img_cv2)
    if ensure_rgb and img_cv2.shape[2] == 1:
        img_cv2 = img_cv2[:, :, 0]
    if ensure_rgb and img_cv2.ndim == 2:
        img_cv2 = np.stack([img_cv2, img_cv2, img_cv2], -1)
    return img_cv2


# --- 3D/2D rendering functions (merged from render.py) ---

AXIS_COLORS_RGB = {
    0: (255, 0, 0),  # red
    3: (0, 255, 0),  # green
    8: (0, 0, 255),  # blue
}  # use RGB for xyz axes respectively


def draw_bb3_lines(
    viz,
    T_world_cam: PoseTW,
    cam: CameraTW,
    obbs: ObbTW,
    draw_cosy: bool,
    T: int,
    line_type=cv2.LINE_AA,
    thickness=1,
    prob_color=False,
    colors=None,
):
    bb3corners_world = obbs.T_world_object * obbs.bb3edge_pts_object(T)
    bb3corners_cam = T_world_cam.inverse() * bb3corners_world
    B = bb3corners_cam.shape[0]
    pt3s_cam = bb3corners_cam.view(B, -1, 3)
    pt2s, valids = cam.project(pt3s_cam)
    sem_ids = obbs.sem_id.int()
    # reshape to lines each composed of T segments
    pt2s = pt2s.round().int().view(B * 12, T, 2)
    valids = valids.view(B * 12, T)
    for line in range(pt2s.shape[0]):
        line_id = line % 12
        obb_id = line // 12
        sem_id = sem_ids[obb_id]
        if colors is not None:
            color = colors[obb_id]
        elif prob_color:
            prob = float(1.0 - obbs[obb_id].prob)
            max_val = 0.5
            min_val = 0.05
            prob = (prob - min_val) / (max_val - min_val)
            prob = max(0.0, min(1.0, prob))
            val = np.uint8([[int(prob * 255)]])
            bgr = cv2.applyColorMap(val, cv2.COLORMAP_JET)[0, 0]
            color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        else:
            color = obbs[obb_id].color
            if (color == -1).all():
                color = 255, 255, 255
            else:
                color = (
                    int(round(float(color[2] * 255))),
                    int(round(float(color[1] * 255))),
                    int(round(float(color[0] * 255))),
                )

        for i in range(T - 1):
            j = i + 1
            if valids[line, i] and valids[line, j]:
                # check if we should color this line in a special way
                if draw_cosy and line_id in AXIS_COLORS_RGB:
                    color = AXIS_COLORS_RGB[line_id]
                pt1 = (
                    int(round(float(pt2s[line, i, 0]))),
                    int(round(float(pt2s[line, i, 1]))),
                )
                pt2 = (
                    int(round(float(pt2s[line, j, 0]))),
                    int(round(float(pt2s[line, j, 1]))),
                )
                cv2.line(
                    viz,
                    pt1,
                    pt2,
                    color,
                    thickness,
                    lineType=line_type,
                )


def draw_bb3s(
    viz,
    T_world_rig: PoseTW,
    cam: CameraTW,
    obbs: ObbTW,
    draw_bb3_center=False,
    draw_label=False,
    draw_cosy=False,
    draw_score=False,
    render_obb_corner_steps=10,
    line_type=cv2.LINE_AA,
    rotate_label=True,
    white_backing_line=False,
    already_rotated=False,
    prob_color=False,
    colors=None,
    texts=None,
    text_sz=0.35,
    thickness=1,
):
    if obbs.shape[0] == 0:
        return viz

    if already_rotated:
        viz = cv2.rotate(viz, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Get pose of camera.
    T_world_cam = T_world_rig.float() @ cam.T_camera_rig.inverse()

    # draw semantic colors
    draw_bb3_lines(
        viz,
        T_world_cam,
        cam,
        obbs,
        draw_cosy=draw_cosy,
        T=render_obb_corner_steps,
        line_type=cv2.LINE_AA,
        thickness=thickness,
        prob_color=prob_color,
        colors=colors,
    )

    if draw_label or draw_bb3_center or texts is not None:
        bb3center_cam = T_world_cam.inverse() * obbs.bb3_center_world
        bb2center_im, valids = cam.unsqueeze(0).project(bb3center_cam.unsqueeze(0))
        bb2center_im, valids = bb2center_im.squeeze(0), valids.squeeze(0)
        for idx, (pt2, valid) in enumerate(zip(bb2center_im, valids)):
            if valid:
                center = (int(pt2[0]), int(pt2[1]))
                if draw_bb3_center:
                    cv2.circle(viz, center, 3, (255, 0, 0), 1, lineType=line_type)

                if draw_label or texts is not None:
                    height = viz.shape[0]
                    sem_id = int(obbs.sem_id.squeeze(-1)[idx])

                    if texts is not None:
                        text = texts[idx]
                    else:
                        text = obbs.text[idx]
                        if (text == -1).all():
                            text = str(sem_id)
                        else:
                            text = unpad_string(tensor2string(obbs.text[idx].byte()))
                    if colors is not None:
                        text_clr = colors[idx]
                    else:
                        text_clr = (200, 200, 200)

                    # rot 90 degree before drawing the text
                    if rotate_label:
                        viz = cv2.rotate(viz, cv2.ROTATE_90_CLOCKWISE)
                        center_rot90 = (height - center[1], center[0])
                        x, y = center_rot90
                    else:
                        x, y = center
                    ((txt_w, txt_h), _) = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_DUPLEX, text_sz, 1
                    )

                    ## Show text on top of the 3d boxes
                    put_text(viz, text, scale=text_sz, font_pt=(x, y), color=text_clr)
                    if draw_score and obbs.prob is not None:
                        score = float(obbs.prob.squeeze(-1)[idx])
                        score_text = f"prob={score:.2f}"
                        score_pos = (x, y + int(txt_h + 0.5))
                        put_text(
                            viz,
                            score_text,
                            scale=text_sz,
                            font_pt=score_pos,
                            color=text_clr,
                        )

                    if rotate_label:
                        viz = cv2.rotate(viz, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if already_rotated:
        viz = cv2.rotate(viz, cv2.ROTATE_90_CLOCKWISE)

    return viz


def render_bb2(img, bb2s, scale=1.0, clr=(0, 255, 0), rotated=False, texts=None):
    if rotated:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if texts is not None:
        assert len(texts) == len(bb2s)

    if isinstance(clr, tuple):
        colors = [clr] * len(bb2s)
    else:
        colors = clr

    for i, bb2 in enumerate(bb2s):
        # draw a rectangle
        xmin = int(round(float(bb2[0])))
        xmax = int(round(float(bb2[1])))
        ymin = int(round(float(bb2[2])))
        ymax = int(round(float(bb2[3])))
        cc = colors[i]
        cv2.rectangle(
            img, (xmin, ymin), (xmax, ymax), cc, int(round(scale * 1)), lineType=16
        )
        if texts is not None and not rotated:
            # Place text in the center of the bounding box
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            put_text(img, texts[i], scale=0.35, font_pt=(center_x, center_y), color=cc)

    if rotated:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if texts is not None:
            for i, bb2 in enumerate(bb2s):
                xmin = int(round(float(bb2[0])))
                xmax = int(round(float(bb2[1])))
                ymin = int(round(float(bb2[2])))
                ymax = int(round(float(bb2[3])))
                W = img.shape[1]  # Width of rotated image = Height of original
                # After 90° CW rotation: original (x,y) -> (H_orig - 1 - y, x)
                # Center of original box: ((xmin+xmax)/2, (ymin+ymax)/2)
                # Maps to rotated: (H_orig - 1 - (ymin+ymax)/2, (xmin+xmax)/2)
                # H_orig = W (width of rotated image)
                center_x = W - 1 - (ymin + ymax) // 2
                center_y = (xmin + xmax) // 2
                cc = colors[i]
                put_text(
                    img, texts[i], scale=0.35, font_pt=(center_x, center_y), color=cc
                )
    return img


def render_depth_patches(sdp_median, rotated, HH, WW):
    sdp_median = sdp_median[None]
    sdp_median = torch.nn.functional.interpolate(
        sdp_median, size=(HH, WW), mode="nearest"
    )
    sdp_median = sdp_median[0, 0]
    max_depth = 5.0
    min_depth = 0.1
    sdp_median = (sdp_median - min_depth) / (max_depth - min_depth)
    sdp_u8 = (sdp_median.clamp(0, 1).numpy() * 255).astype(np.uint8)
    sdp_img2 = cv2.applyColorMap(sdp_u8, cv2.COLORMAP_JET)
    if rotated:
        sdp_img2 = cv2.rotate(sdp_img2, cv2.ROTATE_90_CLOCKWISE)
    return sdp_img2
