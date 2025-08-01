# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "openvino_snippets")

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create static library
add_library(${TARGET_NAME} STATIC
            ${LIBRARY_SRC}
            ${PUBLIC_HEADERS})

add_library(openvino::snippets ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME snippets)
if (ENABLE_CLANG_TIDY AND NOT RISCV64)
    set(clang_tidy_options "--extra-arg=-Wno-unused-command-line-argument")
    if(DEFINED CMAKE_CXX_COMPILER_TARGET)
        list(APPEND clang_tidy_options "--extra-arg=--target=${CMAKE_CXX_COMPILER_TARGET}")
    endif()
    if(DEFINED OV_CLANG_TIDY_TOOLCHAIN_FLAGS)
        list(APPEND clang_tidy_options ${OV_CLANG_TIDY_TOOLCHAIN_FLAGS})
    endif()
    if(ENABLE_CLANG_TIDY_FIX)
        list(APPEND clang_tidy_options "--fix-errors")
    endif()
    set_target_properties(${TARGET_NAME} PROPERTIES
        CXX_CLANG_TIDY "${CLANG_TIDY};${clang_tidy_options}")
endif()


ov_build_target_faster(${TARGET_NAME}
    UNITY
    PCH
)

target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime
                                     PRIVATE openvino::reference openvino::runtime::dev)

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>
                                          PRIVATE $<BUILD_INTERFACE:${SHAPE_INFER_INCLUDE_DIR}>)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

ov_mark_target_as_cc(${TARGET_NAME})

if(ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(tests)
endif()

# LTO

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# install & export

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
                                    INSTALL_INCLUDE_DIRECTORIES "${PUBLIC_HEADERS_DIR}/")
