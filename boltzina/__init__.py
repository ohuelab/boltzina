from importlib.metadata import version as _v

__version__ = _v("boltzina")

# Heavy imports (torch, boltz, rdkit) are NOT loaded here.
# Import explicitly when needed:
#   from boltzina.engine import Boltzina
#   from boltzina.runner import BoltzinaRunner, RunnerConfig
