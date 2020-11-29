cmake_minimum_required(VERSION 3.7)
PROJECT(InteractiveMosaicking)
FIND_PACKAGE( OpenCV REQUIRED )
FIND_PACKAGE( Boost REQUIRED )
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

file(GLOB SOURCES_MOSAICKING graph.cpp landmark_database.cpp affine_transformation.cpp interaction_loop.cpp position_overlap_model.cpp bow_overlap_model.cpp linear_algebra.cpp unionfind.cpp utilities.cpp agent.cpp data_preparation.cpp)
file(GLOB SOURCES_ALGLIB alglib/alglibinternal.cpp  alglib/diffequations.cpp alglib/linalg.cpp alglib/statistics.cpp alglib/alglibmisc.cpp alglib/fasttransforms.cpp alglib/optimization.cpp alglib/ap.cpp	alglib/integration.cpp	alglib/solvers.cpp alglib/dataanalysis.cpp alglib/interpolation.cpp	alglib/specialfunctions.cpp)

ADD_EXECUTABLE(main main.cpp ${SOURCES_MOSAICKING} ${SOURCES_ALGLIB})
TARGET_LINK_LIBRARIES( main ${OpenCV_LIBS})

INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(~/Libraries/src/alglib/include)
