# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Preserve a complete local CANN dependency. Otherwise use the pinned
# packaging scripts shipped with the source tree, including on offline hosts.
set(MAKESELF_PATH "${CMAKE_CURRENT_LIST_DIR}/makeself")
if (CANN_3RD_LIB_PATH
    AND EXISTS "${CANN_3RD_LIB_PATH}/makeself/makeself.sh"
    AND EXISTS "${CANN_3RD_LIB_PATH}/makeself/makeself-header.sh")
    set(MAKESELF_PATH "${CANN_3RD_LIB_PATH}/makeself")
endif()
if (NOT EXISTS "${MAKESELF_PATH}/makeself.sh" OR NOT EXISTS "${MAKESELF_PATH}/makeself-header.sh")
    message(FATAL_ERROR "Makeself scripts are missing from ${MAKESELF_PATH}.")
endif()
