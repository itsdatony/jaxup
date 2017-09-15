cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)

project("jaxup")
if(CMAKE_VERSION VERSION_LESS "3.1")
	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		set(CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS}")
	endif()
else()
	set(CMAKE_CXX_STANDARD 11)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

add_executable(grank src/main.cpp src/jaxup.cpp)
add_executable(uglify src/uglify.cpp src/jaxup.cpp)

install(TARGETS grank
	RUNTIME DESTINATION bin)

