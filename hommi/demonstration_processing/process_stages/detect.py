from omegaconf import DictConfig
import os
import multiprocessing
import pathlib
from tqdm import tqdm
import concurrent.futures
import subprocess
from typing import List
from hommi.common.import_umi_source import get_umi_subprocess_env, get_umi_dir
from hommi.demonstration_processing.utils.generic_util import demonstration_to_display_string, get_demonstration_json_data, get_demonstration_sides_present


GRIPPER_SIDES = {'left', 'right'}


def detect_ar_tag_iphone(demonstration_iterator, cfg: DictConfig):
    """Adapted from 04_detect_aruco.py from UMI"""
    processed_demonstrations = set()
    skipped_demonstrations = set()

    # find mp4 paths for all demonstration videos
    input_video_paths: List[pathlib.Path] = []
    for demonstration_dir in demonstration_iterator(['demonstration', 'grippercalibration']):
        for side in get_demonstration_sides_present(demonstration_dir):
            if side not in GRIPPER_SIDES:
                continue
            video_path = pathlib.Path(demonstration_dir).joinpath(f'{side}_ultrawidergb.mp4').absolute()
            pkl_path = pathlib.Path(demonstration_dir).joinpath(f'{side}_tag_detection.pkl')

            if not pkl_path.exists() or cfg.overwrite:
                print(f'Going to detect AR tag for {demonstration_to_display_string(demonstration_dir, side)}')
                input_video_paths.append(video_path)
                processed_demonstrations.add(demonstration_dir)
                if demonstration_dir in skipped_demonstrations:
                    skipped_demonstrations.remove(demonstration_dir)
            elif demonstration_dir not in processed_demonstrations:
                skipped_demonstrations.add(demonstration_dir)

    # setup paths
    camera_intrinsics = os.path.abspath(cfg.iphone_calibration)
    aruco_yaml = os.path.abspath(cfg.aruco_config)
    assert os.path.isfile(camera_intrinsics)
    assert os.path.isfile(aruco_yaml)

    # setup workers
    num_workers = cfg.num_workers
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    script_path = pathlib.Path('scripts', 'detect_aruco_iphone.py')

    if len(input_video_paths) > 0:
        with tqdm(total=len(input_video_paths)) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = set()
                completed_futures = []
                for video_path in tqdm(input_video_paths):
                    demonstration_dir = str(video_path.parent)
                    # side = 'left' if video_path.name == 'left_ultrawidergb.mp4' else 'right'
                    if video_path.name == 'left_ultrawidergb.mp4':
                        side = 'left'
                    elif video_path.name == 'right_ultrawidergb.mp4':
                        side = 'right'
                    else:
                        continue
                    pkl_path = video_path.parent.joinpath(f'{side}_tag_detection.pkl')

                    # since the ultrawide records at 10Hz, the first frame of the ultrawide may not necessarily be the first frame of the main camera video. Thus we need to offset the timestamps put into the tag detection results, so that they correspond to the actual beginning of the recording (which happens when the first main camera RGB is captured).
                    demonstration_json = get_demonstration_json_data(demonstration_dir, side)
                    video_times = demonstration_json['ultrawideRGBTimes']
                    i = 0
                    try:
                        while video_times[i] == "": # empty time indicates no ultrawide frame captured
                            i += 1
                        time_offset = 1/60 * i # main camera records at 60Hz

                    except IndexError:
                        # skip this demonstration
                        print(f'Skipping {demonstration_to_display_string(demonstration_dir, side)} because no ultrawide frames were captured')
                        skipped_demonstrations.add(demonstration_dir)
                        continue

                    # run tag detection
                    cmd = [
                        'python', script_path.as_posix(),
                        '--input', str(video_path),
                        '--output', str(pkl_path),
                        '--intrinsics_yaml', camera_intrinsics,
                        '--aruco_yaml', aruco_yaml,
                        '--num_workers', '1',
                        '--time_offset', str(time_offset)
                    ]

                    if len(futures) >= num_workers:
                        # limit number of inflight tasks
                        completed, futures = concurrent.futures.wait(futures, 
                            return_when=concurrent.futures.FIRST_COMPLETED)
                        pbar.update(len(completed))
                        completed_futures.extend(completed)

                    futures.add(executor.submit(
                        lambda x: subprocess.run(
                            x, capture_output=True, text=True,
                            env=get_umi_subprocess_env()),
                        cmd))
                    # futures.add(executor.submit(lambda x: print(' '.join(x)), cmd))

                completed, futures = concurrent.futures.wait(futures)            
                pbar.update(len(completed))
                completed_futures.extend(completed)
    
        for future in completed_futures:
            result = future.result()
            if result is not None and result.returncode != 0:
                print(f"Command failed: {' '.join(result.args)}")
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr)
                result.check_returncode()

    print(f'\nProcessed {len(processed_demonstrations)} demonstrations')
    print(f'Skipped {len(skipped_demonstrations)} demonstrations')


def detect_ar_tag_gopro(demonstration_iterator, cfg: DictConfig):
    """Adapted from 04_detect_aruco.py from UMI"""
    processed_demonstrations = set()
    skipped_demonstrations = set()

    # find mp4 paths for all demonstration videos
    input_video_paths: List[pathlib.Path] = []
    for demonstration_dir in demonstration_iterator(['demonstration', 'grippercalibration']):
        for side in ['left', 'right']:
            video_path = pathlib.Path(demonstration_dir).joinpath(f'{side}.mp4').absolute()
            pkl_path = pathlib.Path(demonstration_dir).joinpath(f'{side}_tag_detection.pkl')

            if video_path.exists() and (not pkl_path.exists() or cfg.overwrite):
                print(f'Going to detect AR tag for {demonstration_to_display_string(demonstration_dir, side)}')
                input_video_paths.append(video_path)
                processed_demonstrations.add(demonstration_dir)
                if demonstration_dir in skipped_demonstrations:
                    skipped_demonstrations.remove(demonstration_dir)
            elif demonstration_dir not in processed_demonstrations:
                skipped_demonstrations.add(demonstration_dir)

    # setup paths
    camera_intrinsics = os.path.abspath(cfg.gopro_intrinsics)
    aruco_yaml = os.path.abspath(cfg.aruco_config)
    
    assert os.path.isfile(camera_intrinsics)
    assert os.path.isfile(aruco_yaml)

    # setup workers
    num_workers = cfg.num_workers
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    script_path = pathlib.Path(get_umi_dir()).joinpath('scripts', 'detect_aruco.py')

    if len(input_video_paths) > 0:
        with tqdm(total=len(input_video_paths)) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = set()
                for video_path in tqdm(input_video_paths):
                    demonstration_dir = str(video_path.parent)
                    side = 'left' if video_path.name == 'left.mp4' else 'right'
                    pkl_path = video_path.parent.joinpath(f'tag_detection_{side}.pkl')

                    # run tag detection
                    cmd = [
                        'python', script_path.as_posix(),
                        '--input', str(video_path),
                        '--output', str(pkl_path),
                        '--intrinsics_json', camera_intrinsics,
                        '--aruco_yaml', aruco_yaml,
                        '--num_workers', '1'
                    ]

                    if len(futures) >= num_workers:
                        # limit number of inflight tasks
                        completed, futures = concurrent.futures.wait(futures, 
                            return_when=concurrent.futures.FIRST_COMPLETED)
                        pbar.update(len(completed))

                    futures.add(executor.submit(
                        lambda x: subprocess.run(x, 
                            capture_output=True, env=get_umi_subprocess_env()), 
                        cmd))
                    # futures.add(executor.submit(lambda x: print(' '.join(x)), cmd))

                completed, futures = concurrent.futures.wait(futures)            
                pbar.update(len(completed))

    print(f'\nProcessed {len(processed_demonstrations)} demonstrations')
    print(f'Skipped {len(skipped_demonstrations)} demonstrations')
