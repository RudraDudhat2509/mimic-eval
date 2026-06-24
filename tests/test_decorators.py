import pytest
from mimic.decorators import judge, get_registry, clear_registry


def setup_function():
    clear_registry()


def test_decorator_registers_judge_and_stays_callable():
    @judge("checks grounding", optimize="speed", threshold=0.9)
    def grounding_judge(context: str, response: str) -> bool:
        return context in response

    assert grounding_judge("ab", "xaby") is True            # still callable
    cfg = get_registry()["grounding_judge"]
    assert cfg.description == "checks grounding"
    assert cfg.optimize == "speed"
    assert cfg.threshold == 0.9
    assert grounding_judge._mimic_config is cfg


def test_duplicate_name_raises():
    @judge("first")
    def dup() -> bool:
        return True

    with pytest.raises(ValueError, match="already registered"):
        @judge("second")
        def dup() -> bool:  # noqa: F811
            return False
