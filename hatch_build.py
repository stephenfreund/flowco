import os
import shutil
import subprocess
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class Hook(BuildHookInterface):
    """A hatchling build hook that runs npm install & build."""

    frontend_dir = "src/flowco/mxgraph_component/mxgraph_component/frontend"

    def initialize(self, version, build_data):
        # 1) npm install
        subprocess.run(
            ["npm", "install"],
            cwd=Hook.frontend_dir,
            check=True,
        )

        # 2) npm run build
        subprocess.run(
            ["npm", "run", "build"],
            cwd=Hook.frontend_dir,
            check=True,
        )
