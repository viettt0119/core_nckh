cmake_minimum_required(VERSION 3.18)
project(gaze_core VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ===========================================================================
# Options
# ===========================================================================
option(GAZE_BUILD_TESTS "Build unit tests" ON)
option(GAZE_BUILD_SHARED "Build shared library" ON)

# ===========================================================================
# Sources
# ===========================================================================
set(GAZE_SOURCES
    src/camera_pipeline.cpp
    src/tflite_inference.cpp
    src/smoothing_filter.cpp
    src/dwell_detector.cpp
    src/gaze_core.cpp
)

set(GAZE_HEADERS
    include/gaze_core.h
    include/camera_pipeline.h
    include/tflite_inference.h
    include/smoothing_filter.h
    include/dwell_detector.h
)

# ===========================================================================
# Library target
# ===========================================================================
if(GAZE_BUILD_SHARED)
    add_library(gaze_core SHARED ${GAZE_SOURCES} ${GAZE_HEADERS})
else()
    add_library(gaze_core STATIC ${GAZE_SOURCES} ${GAZE_HEADERS})
endif()

target_include_directories(gaze_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# ===========================================================================
# Platform-specific dependency resolution
# ===========================================================================

# --- OpenCV ---
if(ANDROID)
    # Android NDK: OpenCV must be provided via OPENCV_SDK_DIR
    if(NOT DEFINED OPENCV_SDK_DIR)
        message(FATAL_ERROR "Set -DOPENCV_SDK_DIR=<path-to-OpenCV-android-sdk>")
    endif()
    set(OpenCV_DIR "${OPENCV_SDK_DIR}/sdk/native/jni")
    find_package(OpenCV REQUIRED COMPONENTS core imgproc)
elseif(IOS)
    # iOS: OpenCV framework must be provided via OPENCV_FRAMEWORK_DIR
    if(NOT DEFINED OPENCV_FRAMEWORK_DIR)
        message(FATAL_ERROR "Set -DOPENCV_FRAMEWORK_DIR=<path-to-opencv2.framework>")
    endif()
    find_library(OPENCV_FW opencv2 PATHS ${OPENCV_FRAMEWORK_DIR} NO_DEFAULT_PATH)
    if(NOT OPENCV_FW)
        message(FATAL_ERROR "opencv2.framework not found in ${OPENCV_FRAMEWORK_DIR}")
    endif()
else()
    # Desktop (Windows/Linux/macOS)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc)
endif()

# --- TensorFlow Lite ---
if(ANDROID)
    # Android: prebuilt TFLite from AAR or manual build
    if(NOT DEFINED TFLITE_ROOT)
        message(FATAL_ERROR "Set -DTFLITE_ROOT=<path-to-tflite-prebuilt>")
    endif()
    set(TFLITE_INCLUDE_DIR "${TFLITE_ROOT}/include")
    find_library(TFLITE_LIB tensorflowlite
        PATHS "${TFLITE_ROOT}/lib/${ANDROID_ABI}"
        NO_DEFAULT_PATH
    )
elseif(IOS)
    # iOS: TensorFlowLiteC.framework
    if(NOT DEFINED TFLITE_FRAMEWORK_DIR)
        message(FATAL_ERROR "Set -DTFLITE_FRAMEWORK_DIR=<path-to-TensorFlowLiteC.framework>")
    endif()
    find_library(TFLITE_LIB TensorFlowLiteC
        PATHS ${TFLITE_FRAMEWORK_DIR}
        NO_DEFAULT_PATH
    )
    set(TFLITE_INCLUDE_DIR "${TFLITE_FRAMEWORK_DIR}/Headers")
else()
    # Desktop: find via pkg-config or manual path
    if(DEFINED TFLITE_ROOT)
        set(TFLITE_INCLUDE_DIR "${TFLITE_ROOT}/include")
        find_library(TFLITE_LIB tensorflowlite
            PATHS "${TFLITE_ROOT}/lib"
            NO_DEFAULT_PATH
        )
    else()
        find_library(TFLITE_LIB tensorflowlite)
        set(TFLITE_INCLUDE_DIR "" CACHE PATH "TFLite include directory")
    endif()
endif()

# Link dependencies
if(ANDROID OR (NOT IOS))
    target_link_libraries(gaze_core PRIVATE ${OpenCV_LIBS})
else()
    target_link_libraries(gaze_core PRIVATE ${OPENCV_FW})
endif()

if(TFLITE_LIB)
    target_link_libraries(gaze_core PRIVATE ${TFLITE_LIB})
    if(TFLITE_INCLUDE_DIR)
        target_include_directories(gaze_core PRIVATE ${TFLITE_INCLUDE_DIR})
    endif()
else()
    message(WARNING "TFLite library not found — building without inference support")
    target_compile_definitions(gaze_core PRIVATE GAZE_NO_TFLITE=1)
endif()

# ===========================================================================
# Android-specific flags
# ===========================================================================
if(ANDROID)
    target_link_libraries(gaze_core PRIVATE log android)
    set_target_properties(gaze_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/libs/${ANDROID_ABI}"
    )
endif()

# ===========================================================================
# iOS-specific flags
# ===========================================================================
if(IOS)
    set_target_properties(gaze_core PROPERTIES
        FRAMEWORK TRUE
        FRAMEWORK_VERSION A
        MACOSX_FRAMEWORK_IDENTIFIER com.eyegaze.core
        XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY ""
    )
endif()

# ===========================================================================
# Export / Install
# ===========================================================================
install(TARGETS gaze_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    FRAMEWORK DESTINATION Frameworks
)
install(FILES ${GAZE_HEADERS} DESTINATION include)
