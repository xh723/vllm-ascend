# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import shutil
import subprocess
from pathlib import Path

import pytest

MAKESELF_MODULE = Path(__file__).resolve().parents[3] / "csrc/cmake/third_party/makeself-fetch.cmake"


@pytest.mark.parametrize("cached_scripts", [(), ("makeself.sh",), ("makeself.sh", "makeself-header.sh")])
def test_makeself_resolves_without_network(tmp_path, cached_scripts):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for the packaging test")

    dependency_dir = tmp_path / "local dependencies"
    cached_makeself = dependency_dir / "makeself"
    cached_makeself.mkdir(parents=True)
    for name in cached_scripts:
        (cached_makeself / name).touch()

    expected = cached_makeself if len(cached_scripts) == 2 else MAKESELF_MODULE.parent / "makeself"
    script = tmp_path / "check-makeself.cmake"
    script.write_text(
        f'include("{MAKESELF_MODULE.as_posix()}")\n'
        f'if(NOT "${{MAKESELF_PATH}}" STREQUAL "{expected.as_posix()}")\n'
        '  message(FATAL_ERROR "Unexpected makeself path: ${MAKESELF_PATH}")\n'
        "endif()\n",
        encoding="utf-8",
    )
    command = [cmake, "-DFETCHCONTENT_FULLY_DISCONNECTED=ON", f"-DCANN_3RD_LIB_PATH={dependency_dir.as_posix()}"]
    subprocess.run([*command, "-P", script], check=True, timeout=15)
