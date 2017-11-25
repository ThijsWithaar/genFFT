find_package(OpenCL REQUIRED)

message("OpenCL_INCLUDE_DIR : ${OpenCL_INCLUDE_DIR}")

add_executable(sample
	clutils.cpp
	cpu.cpp
	main.cpp
)

set_property(TARGET sample PROPERTY CXX_STANDARD 11)

target_include_directories(sample
	PRIVATE
		${OpenCL_INCLUDE_DIR}
)

target_link_libraries(sample
	genFFT
	${OpenCL_LIBRARIES}
)
