# hatch_build.py

import subprocess
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class Hook(BuildHookInterface):
    def initialize(self, version, build_data):
        pkg = "src/mxgraph_component"
        frontend = "src/mxgraph_component/mxgraph_component/frontend"
        subprocess.run(["npm", "install"], cwd=frontend, check=True)
        subprocess.run(["npm", "run", "build"], cwd=frontend, check=True)
