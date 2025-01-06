from flowco.util.semantic_diff import (
    semantic_diff,
)
from pydantic import BaseModel


def test_semantic_diff():
    class A(BaseModel):
        a: int
        b: str

    class B(BaseModel):
        a: int
        b: str

    a = A(a=1, b="hello")
    b = B(a=1, b="world")
    diff = semantic_diff(a, b)
    print(diff)


if __name__ == "__main__":
    test_semantic_diff()
