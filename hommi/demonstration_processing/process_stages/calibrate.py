from omegaconf import DictConfig
from pathlib import Path
import subprocess
import os
import collections
import pickle
import json
import numpy as np

from hommi.demonstration_processing.utils.generic_util import demonstration_to_display_string, get_demonstration_sides_present
from hommi.common.import_umi_source import get_umi_dir
from hommi.demonstration_processing.utils.gripper_util import get_gripper_width_offset
from hommi.common.contants import IPHUMI_ULTRAWIDE_OFFSET_FROM_FINGER_AR_TAG_Z, IPHUMI_ULTRAWIDE_OFFSET_FROM_CENTER_X


GRIPPER_SIDES = {'left', 'right'}


def calibrate_gripper_range_iphone(demonstration_iterator, cfg: DictConfig):
    """Adapted from 05_run_calibrations.py from UMI"""
    skipped_demonstrations = set()
    processed_demonstrations = set()
    for demonstration_dir in demonstration_iterator('grippercalibration'):
        for side in get_demonstration_sides_present(demonstration_dir):
            if side not in GRIPPER_SIDES:
                continue
            tag_path = Path(demonstration_dir).joinpath(f'{side}_tag_detection.pkl').absolute()
            gripper_range_path = Path(demonstration_dir).joinpath(f'{side}_gripper_range.json').absolute()
            
            if gripper_range_path.exists() and not cfg.overwrite:
                if demonstration_dir not in processed_demonstrations:
                    skipped_demonstrations.add(demonstration_dir)
                continue

            print(f'Processing {demonstration_to_display_string(demonstration_dir, side)} ', end='')

            input = str(tag_path)
            output = str(gripper_range_path)
            tag_det_threshold = cfg.tag_det_threshold

            tag_detection_results = pickle.load(open(input, 'rb'))
    
            # identify gripper hardware id
            n_frames = len(tag_detection_results)
            tag_counts = collections.defaultdict(lambda: 0)
            for frame in tag_detection_results:
                for key in frame['tag_dict'].keys():
                    tag_counts[key] += 1
            tag_stats = collections.defaultdict(lambda: 0.0)
            for k, v in tag_counts.items():
                tag_stats[k] = v / n_frames

            max_tag_id = np.max(list(tag_stats.keys()))
            tag_per_gripper = 6
            max_gripper_id = max_tag_id // tag_per_gripper

            gripper_prob_map = dict()
            for gripper_id in range(max_gripper_id+1):
                left_id = gripper_id * tag_per_gripper
                right_id = left_id + 1
                left_prob = tag_stats[left_id]
                right_prob = tag_stats[right_id]
                gripper_prob = min(left_prob, right_prob)
                if gripper_prob <= 0:
                    continue
                gripper_prob_map[gripper_id] = gripper_prob
            if len(gripper_prob_map) == 0:
                print("No grippers detected!")
                assert False

            gripper_probs = sorted(gripper_prob_map.items(), key=lambda x:x[1])
            gripper_id = gripper_probs[-1][0]
            gripper_prob = gripper_probs[-1][1]
            print(f"Detected gripper id: {gripper_id} with probability {gripper_prob}")
            if gripper_prob < tag_det_threshold:
                print(f"Detection rate {gripper_prob} < {tag_det_threshold} threshold.")
                assert False
                
            # run calibration
            left_id = gripper_id * tag_per_gripper
            right_id = left_id + 1

            gripper_widths = list()
            for i, dt in enumerate(tag_detection_results):
                tag_dict = dt['tag_dict']
                width, _, _ = get_gripper_width_offset(tag_dict, left_id, right_id, nominal_z=IPHUMI_ULTRAWIDE_OFFSET_FROM_FINGER_AR_TAG_Z, offset_x=IPHUMI_ULTRAWIDE_OFFSET_FROM_CENTER_X)
                if width is None:
                    width = float('Nan')
                gripper_widths.append(width)
            gripper_widths = np.array(gripper_widths)
            max_width = np.nanmax(gripper_widths)
            min_width = np.nanmin(gripper_widths)

            result = {
                'gripper_id': gripper_id,
                'left_finger_tag_id': left_id,
                'right_finger_tag_id': right_id,
                'max_width': max_width,
                'min_width': min_width
            }
            json.dump(result, open(output, 'w'), indent=2)

            processed_demonstrations.add(demonstration_dir)
            if demonstration_dir in skipped_demonstrations:
                skipped_demonstrations.remove(demonstration_dir)
    
    print(f"\nProcessed {len(processed_demonstrations)} demonstrations")
    print(f"Skipped {len(skipped_demonstrations)} demonstrations")

def calibrate_gripper_range_gopro(demonstration_iterator, cfg: DictConfig):
    """Adapted from 05_run_calibrations.py from UMI"""
    umi_dir = Path(get_umi_dir()).absolute()
    script_dir = umi_dir.joinpath('scripts')
    script_path = script_dir.joinpath('calibrate_gripper_range.py')
    assert script_path.is_file()

    skipped_demonstrations = set()
    processed_demonstrations = set()
    for demonstration_dir in demonstration_iterator('grippercalibration'):
        for side in get_demonstration_sides_present(demonstration_dir):
            tag_path = Path(demonstration_dir).joinpath(f'{side}_tag_detection.pkl').absolute()
            gripper_range_path = Path(demonstration_dir).joinpath(f'{side}_gripper_range.json').absolute()
            
            if gripper_range_path.exists() and not cfg.overwrite:
                if demonstration_dir not in processed_demonstrations:
                    skipped_demonstrations.add(demonstration_dir)
                continue

            print(f'Processing {demonstration_to_display_string(demonstration_dir, side)} ', end='')

            cmd = [
                'python', str(script_path),
                '--input', str(tag_path),
                '--output', str(gripper_range_path),
                '-t', str(cfg.tag_det_threshold)
            ]
            env={**os.environ, 'PYTHONPATH':';'.join([str(umi_dir)])} # Add UMI dir to sys.path in the subprocess so that it can import umi modules properly; we have to do this because this script is not located in the `universal_manipulation_interface` project, but we want to act like it is 
            subprocess.run(cmd, env=env)

            processed_demonstrations.add(demonstration_dir)
            if demonstration_dir in skipped_demonstrations:
                skipped_demonstrations.remove(demonstration_dir)
    
    print(f"\nProcessed {len(processed_demonstrations)} demonstrations")
    print(f"Skipped {len(skipped_demonstrations)} demonstrations")
