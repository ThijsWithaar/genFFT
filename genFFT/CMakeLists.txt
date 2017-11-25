find_package(OpenCL REQUIRED)

add_library(genFFT SHARED
	fftCore.cpp
	genFFT.cpp
)

set_property(TARGET genFFT PROPERTY CXX_STANDARD 11)

target_include_directories(genFFT
	PUBLIC
		"${PROJECT_SOURCE_DIR}/genFFT"
)

target_link_libraries(genFFT
	${OpenCL_LIBRARIES}
)

set_target_properties( genFFT PROPERTIES VERSION ${GENFFT_VERSION} )
set_target_properties( genFFT PROPERTIES SOVERSION ${GENFFT_SOVERSION} )

# CPack configuration; include the executable into the package
install( TARGETS genFFT
        EXPORT Library
        RUNTIME DESTINATION bin${SUFFIX_BIN}
        LIBRARY DESTINATION lib${SUFFIX_LIB}
        ARCHIVE DESTINATION lib${SUFFIX_LIB}/import
)
