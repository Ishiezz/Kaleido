from pathlib import Path
import tomllib


def test_gpu_extra_dependencies_are_guarded_for_supported_python_versions() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        pyproject = tomllib.load(handle)

    gpu_deps = pyproject["project"]["optional-dependencies"]["gpu"]

    assert any("python_version < '3.14'" in dep for dep in gpu_deps)
    assert any("vllm" in dep for dep in gpu_deps)
