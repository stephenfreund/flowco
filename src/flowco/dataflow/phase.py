from enum import IntEnum
from typing import Dict, Optional, Tuple


class Phase(IntEnum):
    clean = 0
    requirements = 1
    algorithm = 2
    code = 3
    runnable = 4
    run_checked = 5
    assertions_code = 6
    assertions_checked = 7
    sanity_checks = 8
    unit_tests = 9
    tests_runnable = 10
    tests_checked = 11

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @staticmethod
    def diff(
        map1: Dict[str, "Phase"], map2: Dict[str, "Phase"]
    ) -> Dict[str, Tuple[Optional["Phase"], Optional["Phase"]]]:
        """
        Return a dictionary of the differences between two dictionaries.
        """
        diff = {}
        for key in sorted(map1.keys() | map2.keys()):
            if key not in map1 or key not in map2 or map1[key] != map2[key]:
                diff[key] = (map1.get(key), map2.get(key))
        return diff
