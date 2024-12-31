from __future__ import annotations
from typing import List, Dict, Any
from pydantic import BaseModel

from flowco.dataflow.phase import Phase
from flowco.util.semantic_diff import semantic_diff


class PhaseCacheDescriptor(BaseModel):
    phase: Phase
    in_fields: List[str]
    out_fields: List[str]


class CacheEntry(BaseModel):
    in_values: Dict[str, Any]
    out_values: Dict[str, Any]
    valid: bool = True

    def __init__(self, **data):
        super().__init__(**data)
        # assert self.valid, "Cache entry is invalid"
        # assert set(self.in_values.keys()) == set(self.descriptor.in_fields), "Invalid in values"
        # assert set(self.out_values.keys()) == set(self.descriptor.out_fields), "Invalid out values"

    def matches_in(self, in_node: BaseModel) -> bool:
        in_values = in_node.model_dump(include=self.in_values.keys())
        return self.in_values == in_values and self.valid

    def matches_out(self, in_node: BaseModel) -> bool:
        out_values = in_node.model_dump(include=self.out_values.keys())
        return self.out_values == out_values and self.valid

    def matches_in_and_out(self, in_node: BaseModel) -> bool:
        return self.matches_in(in_node) and self.matches_out(in_node)

    def invalidate(self):
        return self.model_copy(update={"valid": False})

    def diff(self, in_node: BaseModel) -> Dict[str, Any]:
        in_values = in_node.model_dump(include=self.in_values.keys())
        return semantic_diff(self.in_values, in_values)

    def apply(self, in_node: BaseModel) -> BaseModel:
        assert self.valid, "Cache entry is invalid"
        assert self.matches_in(in_node), "Cache entry does not match input node"
        out_node = in_node.update()
        return out_node

    def update(self, node: BaseModel) -> CacheEntry:
        new_entry = CacheEntry(
            in_values=node.model_dump(include=self.in_values.keys()),
            out_values=node.model_dump(include=self.out_values.keys()),
        )
        if new_entry != self:
            return new_entry
        else:
            return self


class BuildCache(BaseModel):
    caches: Dict[Phase, CacheEntry] = {}

    def __init__(self, **data):
        super().__init__(**data)

        self._descriptors = {
            Phase.requirements: PhaseCacheDescriptor(
                phase=Phase.requirements,
                in_fields=["function_parameters", "preconditions", "label", "pill"],
                out_fields=[
                    "requirements",
                    "function_parameters",
                    "function_return_type",
                    "function_computed_value",
                    "description",
                ],
            ),
            Phase.algorithm: PhaseCacheDescriptor(
                phase=Phase.algorithm,
                in_fields=["requirements", "preconditions"],
                out_fields=["algorithm"],
            ),
            Phase.code: PhaseCacheDescriptor(
                phase=Phase.code,
                in_fields=[
                    "signature",
                    "function_parameters",
                    "requirements",
                    "algorithm",
                ],
                out_fields=["code"],
            ),
            Phase.assertions_code: PhaseCacheDescriptor(
                phase=Phase.assertions_code,
                in_fields=[
                    "preconditions",
                    "requirements",
                    "function_parameters",
                    "function_return_type",
                    "assertions",
                ],
                out_fields=["assertion_checks"],
            ),
        }

    def matches_in(self, phase: Phase, node: BaseModel) -> bool:
        if phase not in self.caches:
            return False
        return self.caches[phase].matches_in(node)

    def matches_out(self, phase: Phase, node: BaseModel) -> bool:
        if phase not in self.caches:
            return False
        return self.caches[phase].matches_out(node)

    def matches_in_and_out(self, phase: Phase, node: BaseModel) -> bool:
        if phase not in self.caches:
            return False
        return self.caches[phase].matches_in_and_out(node)

    def diff(self, phase: Phase, node: BaseModel) -> Dict[str, Any]:
        if phase not in self.caches:
            return {}
        return self.caches[phase].diff(node)

    def apply(self, phase: Phase, node: BaseModel) -> BaseModel:
        if phase not in self.caches:
            return node
        return self.caches[phase].apply(node)

    def update(self, phase: Phase, node: BaseModel) -> BaseModel:
        if phase not in self.caches:
            item = CacheEntry(
                in_values=node.model_dump(include=self._descriptors[phase].in_fields),
                out_values=node.model_dump(include=self._descriptors[phase].out_fields),
            )
        else:
            item = self.caches[phase].update(node)

        new_cache = BuildCache(caches=self.caches | {phase: item})
        if new_cache != self:
            return new_cache
        else:
            return self

    def update_all(self, node: BaseModel) -> BaseModel:
        new_caches = {
            phase: CacheEntry(
                in_values=node.model_dump(include=self._descriptors[phase].in_fields),
                out_values=node.model_dump(include=self._descriptors[phase].out_fields),
            )
            for phase in self._descriptors
        }
        new_cache = BuildCache(caches=new_caches)
        if new_cache != self:
            return new_cache
        else:
            return self

    def invalidate(self, target: Phase):
        assert target in [
            Phase.requirements,
            Phase.algorithm,
            Phase.code,
        ], f"Invalid target {target}"

        new_cache = BuildCache(
            caches={
                phase: cache.invalidate() if phase == target else cache
                for phase, cache in self.caches.items()
            }
        )
        if new_cache != self:
            return new_cache
        else:
            return self
