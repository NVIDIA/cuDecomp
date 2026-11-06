# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Label Python versions using the same environment selection passed to CMake."""

import os
from email.parser import Parser
from pathlib import Path


def dynamic_metadata(settings, project):
    value = os.environ.get("CUDECOMP_BUILD_CPU_ONLY", "OFF").upper()
    # Match CMake's false constants so the label agrees with the compiled flag.
    cpu_only = value not in {"", "0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND"} and not value.endswith("-NOTFOUND")
    version = project["version"] + ("+cpu" if cpu_only else "")

    # A source archive's version is fixed. Require the same environment selection
    # when building its wheel rather than changing that version during installation.
    pkg_info = Path("PKG-INFO")
    if pkg_info.is_file():
        archived_version = Parser().parsestr(pkg_info.read_text(encoding="utf-8"))["Version"]
        if archived_version != version:
            raise ValueError(
                f"Source distribution version {archived_version} does not match selected build {version}; "
                "set CUDECOMP_BUILD_CPU_ONLY to match the source distribution"
            )
    return {"version": version}
