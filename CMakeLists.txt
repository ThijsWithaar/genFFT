cmake_minimum_required(VERSION 3.0)
project(genFFT)


# Define a version for the code
if( NOT DEFINED GENFFT_VERSION_MAJOR )
	set( GENFFT_VERSION_MAJOR 1 )
endif( )

if( NOT DEFINED GENFFT_VERSION_MINOR )
	set( GENFFT_VERSION_MINOR 0 )
endif( )

if( NOT DEFINED GENFFT_VERSION_PATCH )
	set( GENFFT_VERSION_PATCH 0 )
endif( )

set(GENFFT_VERSION "${GENFFT_VERSION_MAJOR}.${GENFFT_VERSION_MINOR}.${GENFFT_VERSION_PATCH}")
set(GENFFT_SOVERSION 1)


set(SUFFIX_LIB "")


add_subdirectory(genFFT)
add_subdirectory(sample)



if( WIN32 )
	set( destdir CMake )
else( )
	set( destdir lib${SUFFIX_LIB}/cmake/clFFT )
endif( )

install( FILES
		"genFFT/genFFT.h"
	DESTINATION
		"./include" )
		
install( FILES ${CMAKE_SOURCE_DIR}/FindgenFFT.cmake DESTINATION ${destdir} )

# The following code is setting variables to control the behavior of CPack to generate our
set(CPACK_PACKAGE_NAME "genFFT")
set(CPACK_PACKAGE_CONTACT "a@b.com")
set(CPACK_PACKAGE_DESCRIPTION "A software library containing FFT functions written in OpenCL")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "OpenCL implementation of an FFT library, by Intel")
set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${GENFFT_VERSION}-${CMAKE_HOST_SYSTEM_NAME}" )
set(CPACK_PACKAGE_VENDOR "Neutral")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/license.txt")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

if( WIN32 )
	set( CPACK_SOURCE_GENERATOR "ZIP" )
	set( CPACK_GENERATOR "ZIP" )
else( )
	set( CPACK_DEBIAN_PACKAGE_DEPENDS "opencl-c-headers (>=1.2)")
	set( CPACK_DEBIAN_PACKAGE_RECOMMENDS "build-essential, cmake")

	set( CPACK_SOURCE_GENERATOR "TGZ;DEB" )
	set( CPACK_GENERATOR "TGZ;DEB" )
endif( )

# Define all variables that influence CPack before including CPack, such as install targets
include( CPack )
