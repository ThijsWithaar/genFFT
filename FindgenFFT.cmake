if (GENFFT_LIBRARY)
  set (GENFFT_FIND_QUIETLY TRUE)
endif (GENFFT_LIBRARY)

MARK_AS_ADVANCED(GENFFT_LIBRARY_PATH)
SET(GENFFT_LIBRARY_PATH "/usr/lib" CACHE PATH "The base path of the directory that includes genFFT.lib" )

set(GENFFT_LIB_SEARCHPATH
	${GENFFT_LIBRARY_PATH}
	/usr/lib
	/usr/lib/arm-linux-gnueabihf
)
 
FIND_LIBRARY(GENFFT_LIBRARY NAMES genFFT PATHS ${GENFFT_LIB_SEARCHPATH})

add_library(genFFT::genFFT UNKNOWN IMPORTED)
set_target_properties(genFFT::genFFT PROPERTIES IMPORTED_LOCATION "${OPENGLES_LIBRARY}")

# Handle the QUIETLY and REQUIRED arguments and set OPENGLES_FOUND to TRUE if all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (OpenGLES DEFAULT_MSG GENFFT_LIBRARY)
