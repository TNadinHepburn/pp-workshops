#include <iostream>
#include <vector>

#include "Utils.h"

#include <CL/opencl.hpp>

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform" << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -s : choose size of vector (default: 128)" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	int vector_size = 128;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-s") == 0) { vector_size = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
				cl::Program::Sources sources;

		AddSources(sources, "kernels.cl");

		cl::Program program(context, sources);


		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 3 - memory allocation
		//host - input
		std::vector<int> A = { 1, 2, 3, 4, 5};
		std::vector<int> B = { 6, 7 };
		size_t vector_A_elements = A.size();//number of elements
		size_t vector_B_elements = B.size();//number of elements
		size_t vector_A_size = A.size() * sizeof(int);//size in bytes
		size_t vector_B_size = B.size() * sizeof(int);//size in bytes


		//host - output
		std::vector<int> C(vector_A_elements*vector_B_elements);
		size_t vector_C_elements = C.size();//number of elements
		size_t vector_C_size = C.size() * sizeof(int);//size in bytes

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_A_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_C_size);

		//Part 4 - device operations

		//4.1 Copy arrays A and B to device memory
		cl::Event A_event;
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_A_size, &A[0], NULL, &A_event);
		cl::Event B_event;
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_B_size, &B[0], NULL, &B_event);

		//4.2 Setup and execute the kernel (i.e. device code)

		cl::Kernel kernel = cl::Kernel(program, "add2D");
		kernel.setArg(0, buffer_A);
		kernel.setArg(1, buffer_B);
		kernel.setArg(2, buffer_C);

		cl::Event prof_event;

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(5, 2), cl::NullRange, NULL, &prof_event);

		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0]; // get device
		cerr << kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << endl; // get info
		cerr << kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << endl; // get info

		//4.3 Copy the result from device to host
		cl::Event C_event;
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_C_size, &C[0], NULL, &C_event);

		//std::cout << "Arr Size = " << A.size() << std::endl;
		std::cout << "Vector A = " << A << std::endl;
		std::cout << "Vector B = " << B << std::endl;
		std::cout << "Vector C = " << C << std::endl;

		std::cout << "Kernel execution time [ns]:" <<
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl;
		std::cout << "Overall execution time[ns]:" <<
			A_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			A_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() +
			B_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			B_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;


	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}