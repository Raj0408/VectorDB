cmake_minimum_required(VERSION 3.10)
project(VectorDB CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB SOURCES
        "src/*.cpp"
        "src/engine/*.cpp"
)

add_executable(VectorDB ${SOURCES})
target_include_directories(VectorDB PRIVATE src/engine)
