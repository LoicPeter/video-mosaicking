cmake_minimum_required(VERSION 3.7)


PROJECT(VideoMosaicking)


FIND_PACKAGE( OpenCV REQUIRED )
FIND_PACKAGE( Boost REQUIRED )
FIND_PACKAGE(Eigen3 REQUIRED)

INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(~/Libraries/src/alglib/include)

FILE(GLOB SOURCES_MOSAICKING src/*.cpp)
FILE(GLOB SOURCES_ALGLIB alglib/*.cpp)

ADD_EXECUTABLE(main main.cpp ${SOURCES_MOSAICKING} ${SOURCES_ALGLIB})
TARGET_LINK_LIBRARIES( main ${OpenCV_LIBS})


