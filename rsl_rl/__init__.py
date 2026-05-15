import sys

# Import inner subpackages in dependency order and register them
# in sys.modules as top-level rsl_rl.X to satisfy absolute imports
# (e.g. `from rsl_rl.algorithm import PPO` used inside the inner package).
from .rsl_rl import modules, storage, env

sys.modules.setdefault("rsl_rl.modules", modules)
sys.modules.setdefault("rsl_rl.storage", storage)
sys.modules.setdefault("rsl_rl.env", env)

from .rsl_rl import algorithm

sys.modules.setdefault("rsl_rl.algorithm", algorithm)

from .rsl_rl import runner

sys.modules.setdefault("rsl_rl.runner", runner)

from .rsl_rl import *