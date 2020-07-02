cmake_minimum_required(VERSION 3.16)
project(cart_pole)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-g3")

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(autodiff REQUIRED)

add_executable(cart_pole main.cpp)
target_link_libraries(cart_pole nlopt Eigen3::Eigen autodiff::autodiff)

add_executable(test test.cpp)
target_link_libraries(test nlopt Eigen3::Eigen autodiff::autodiff)

add_executable(opt opt.cpp)
target_link_libraries(opt nlopt Eigen3::Eigen autodiff::autodiff)