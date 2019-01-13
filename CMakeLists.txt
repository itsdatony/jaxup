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
	set(CMAKE_CXX_FLAGS "/W4 /WX ${CMAKE_CXX_FLAGS}")
endif()

add_library(jaxup INTERFACE)
target_include_directories(jaxup INTERFACE include/)

add_executable(grank src/main.cpp)
target_link_libraries(grank jaxup)

add_executable(uglify src/uglify.cpp)
target_link_libraries(uglify jaxup)

add_executable(doubleWriter src/doubleWriter.cpp)
target_link_libraries(doubleWriter jaxup)

install(TARGETS jaxup
	RUNTIME DESTINATION bin
	ARCHIVE DESTINATION lib
	PUBLIC_HEADER DESTINATION include)
