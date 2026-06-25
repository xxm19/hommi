# adapted from `detect_aruco.py` from UMI

import sys
import os

import click
from tqdm import tqdm
import yaml
import av
import numpy as np
import cv2
import pickle
from typing import Dict

from hommi.common import import_umi_source
from umi.common.cv_util import (
    parse_aruco_config, 
    draw_predefined_mask
)

@click.command()
@click.option('-i', '--input', required=True)
@click.option('-o', '--output', required=True)
@click.option('-ij', '--intrinsics_yaml', required=True)
@click.option('-ay', '--aruco_yaml', required=True)
@click.option('-n', '--num_workers', type=int, default=4)
@click.option('-to', '--time_offset', type=float, default=0.0, help='the time offset (in seconds) to apply to the timestamps of the detected tags')
def main(input, output, intrinsics_yaml, aruco_yaml, num_workers, time_offset):
    cv2.setNumThreads(num_workers)

    # load aruco config
    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, 'r')))
    aruco_dict = aruco_config['aruco_dict']
    marker_size_map = aruco_config['marker_size_map']

    # load intrinsics
    with open(intrinsics_yaml, 'r') as f:
        camera_calibration = yaml.safe_load(f)
    K = np.array(camera_calibration['ultrawide']['intrinsics'])

    results = list()
    with av.open(os.path.expanduser(input)) as in_container:
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = "AUTO"
        in_stream.thread_count = num_workers

        in_res = np.array([in_stream.height, in_stream.width])[::-1]

        for i, frame in tqdm(enumerate(in_container.decode(in_stream)), total=in_stream.frames):
            img = frame.to_ndarray(format='rgb24')
            frame_cts_sec = frame.pts * in_stream.time_base
            # avoid detecting tags in the mirror
            img = draw_predefined_mask(img, color=(0,0,0), mirror=True, gripper=False, finger=False)
            tag_dict = detect_localize_aruco_tags_iphone(
                img=img,
                aruco_dict=aruco_dict,
                marker_size_map=marker_size_map,
                K=K,
                refine_subpix=True
            )
            result = {
                'frame_idx': i,
                'time': float(frame_cts_sec) + time_offset,
                'tag_dict': tag_dict
            }
            results.append(result)
    
    # dump
    pickle.dump(results, open(os.path.expanduser(output), 'wb'))


def detect_localize_aruco_tags_iphone(
        img: np.ndarray, 
        aruco_dict: cv2.aruco.Dictionary, 
        marker_size_map: Dict[int, float], 
        K: np.ndarray,
        refine_subpix: bool=True):
    param = cv2.aruco.DetectorParameters()
    if refine_subpix:
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(cv2.aruco, 'detectMarkers'):
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            image=img, dictionary=aruco_dict, parameters=param)
    else:
        detector = cv2.aruco.ArucoDetector(aruco_dict, param)
        corners, ids, rejectedImgPoints = detector.detectMarkers(img)
    if len(corners) == 0:
        return dict()

    tag_dict = dict()
    for this_id, this_corners in zip(ids, corners):
        this_id = int(this_id[0])
        if this_id not in marker_size_map:
            continue
        
        marker_size_m = marker_size_map[this_id]
        if hasattr(cv2.aruco, 'estimatePoseSingleMarkers'):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                this_corners, marker_size_m, K, np.zeros((1,5)))
            rvec = rvec.squeeze()
            tvec = tvec.squeeze()
        else:
            half_marker = marker_size_m / 2
            marker_points = np.array([
                [-half_marker, half_marker, 0],
                [half_marker, half_marker, 0],
                [half_marker, -half_marker, 0],
                [-half_marker, -half_marker, 0],
            ], dtype=np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                marker_points,
                this_corners.squeeze().astype(np.float32),
                K,
                np.zeros((1,5)),
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            rvec = rvec.squeeze()
            tvec = tvec.squeeze()
        tag_dict[this_id] = {
            'rvec': rvec,
            'tvec': tvec,
            'corners': this_corners.squeeze()
        }
    return tag_dict


if __name__ == "__main__":
    main()
