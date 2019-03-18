cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)

project("jaxup")
if(CMAKE_VERSION VERSION_LESS "3.1")
	if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
		set(CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS}")
	endif()
else()
	set(CMAKE_CXX_STANDARD 11)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	set(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra ${CMAKE_CXX_FLAGS}")
elseif(${MSVC})
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	set(CMAKE_CXX_FLAGS "/W4 /WX /EHsc ${CMAKE_CXX_FLAGS}")
endif()

include_directories(include/)

add_executable(fastParse src/main.cpp)

add_executable(uglify src/uglify.cpp)

add_executable(doubleWriter src/doubleWriter.cpp)

add_executable(nodeCopy src/nodeCopy.cpp)

install(DIRECTORY include/ DESTINATION include/jaxup FILES_MATCHING PATTERN "*.h")
