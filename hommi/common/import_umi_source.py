# Since umi is not setup as a python package, we add its sources to the path, so we can import files from the codebase
import os
import sys
from pathlib import Path

def get_umi_dir():
    umi_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'deps', 'universal_manipulation_interface')
    umi_dir = Path(umi_dir).absolute().resolve().as_posix()
    return umi_dir

def get_hommi_dir():
    hommi_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    hommi_dir = Path(hommi_dir).absolute().resolve().as_posix()
    return hommi_dir

sys.path.insert(0, get_umi_dir())

def get_umi_subprocess_env():
    """When you run a subprocess to run script in UMI dir you need to provide path to UMI codebase"""
    pythonpath = [get_umi_dir(), get_hommi_dir()]
    existing_pythonpath = os.environ.get('PYTHONPATH')
    if existing_pythonpath:
        pythonpath.append(existing_pythonpath)
    return {**os.environ, 'PYTHONPATH': os.pathsep.join(pythonpath)}
