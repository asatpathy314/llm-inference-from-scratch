cmake_minimum_required(VERSION 3.20)
project(inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(inference src/main.cpp)
target_include_directories(inference PRIVATE third_party)
target_compile_options(inference PRIVATE -Wall -Wextra -fsanitize=address,undefined)
target_link_options(inference PRIVATE -fsanitize=address,undefined)
