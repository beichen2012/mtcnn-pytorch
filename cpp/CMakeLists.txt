#cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
#project(example-app)

#list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")
#find_package(Torch REQUIRED)

#add_executable(example-app example-app.cpp)
#target_link_libraries(example-app "${TORCH_LIBRARIES}")
#set_property(TARGET example-app PROPERTY CXX_STANDARD 11)



project(MTCNN_TEST)
cmake_minimum_required(VERSION 3.2)

add_definitions(-std=c++11)

# find Torch
list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")
find_package(Torch REQUIRED)
message(STATUS "TORCH FOUND!")
message(STATUS "Torch inc: " ${TORCH_INCLUDE_DIRS})
message(STATUS "Torch lib: " ${TORCH_LIBRARIES})

set(OPENCV_LIBS "")
find_package(OpenCV 3 QUIET COMPONENTS core highgui imgproc imgcodecs)
if(OpenCV_FOUND)
    message(STATUS "OpenCV3 Found (${OpenCV_CONFIG_PATH})")
    list(APPEND OPENCV_LIBS ${OpenCV_LIBS})
endif()
if(NOT OpenCV_FOUND)
    # OpenCV 2
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
    message(STATUS "OpenCV2 Found (${OpenCV_CONFIG_PATH})")
    list(APPEND OPENCV_LIBS ${OpenCV_LIBS})
endif()
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV libs: (${OPENCV_LIBS})")
else()
    message(WARNING "Not find OpenCV. project maybe invalid ")
endif()

message(STATUS "Notice that, current build type -> " ${CMAKE_BUILD_TYPE})

add_definitions(-DMODEL_PATH="${PROJECT_SOURCE_DIR}/")
add_definitions(-DLOG_OUT)
add_definitions(-DUSE_PNET_MULTI_THREAD)


include_directories(${CMAKE_SOURCE_DIR}/third_party)
include_directories(${CMAKE_SOURCE_DIR}/third_party/eigen)


set(SRC main.cpp MTCNN.cpp torchutils.cpp)
#set(SRC example-app.cpp)

add_executable(mtcnn_test ${SRC})
target_link_libraries(mtcnn_test "${TORCH_LIBRARIES}")
target_link_libraries(mtcnn_test ${OPENCV_LIBS})
set_property(TARGET mtcnn_test PROPERTY CXX_STANDARD 11)
