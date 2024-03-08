//a simple OpenCL kernel which adds two integer vectors A and B together into a third vector C
kernel void add(global const int* A, global const int* B, global int* C) {
	int id = get_global_id(0);
	if (id == 0) { // perform this part only once i.e. for work item 0
		printf("work group size %d\n", get_local_size(0), "\n");
	}
	int loc_id = get_local_id(0);
	printf("global id = %d, local id = %d\n", id, "  ", loc_id);
	C[id] = A[id] + B[id];
}

//a simple OpenCL kernel which adds two float vectors A and B together into a third vector C
kernel void addf(global const float* A, global const float* B, global float* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
}

//a simple OpenCL kernel which adds two double vectors A and B together into a third vector C
kernel void addd(global const double* A, global const double* B, global double* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
}

//a simple OpenCL kernel which multiplies two vectors A and B together into a third vector C
kernel void mult(global const int* A, global const int* B, global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] * B[id];
}

//a simple OpenCL kernel which multiplies two vectors A and B together into a third vector C, vector B is then added to vector C
kernel void multadd(global const int* A, global const int* B, global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] * B[id];
	C[id] = C[id] + B[id];
}

//a simple smoothing kernel averaging values in a local window (radius 2)
kernel void avg_filter(global const int* A, global int* B) {
	int id = get_global_id(0);
	int length = get_global_size(0);
	if (id == 0) {
		B[id] = (A[id] + A[id + 1] + A[id + 2]) / 3;
	}
	else if (id == 1) {
		B[id] = (A[id - 1] + A[id] + A[id + 1] + A[id + 2]) / 4;
	}
	else if (id == length - 1) {
		B[id] = (A[id - 2] + A[id - 1] + A[id]) / 3;
	}
	else if (id == length - 2) {
		B[id] = (A[id - 2] + A[id - 1] + A[id] + A[id + 1]) / 4;
	}
	else {
		B[id] = (A[id - 2] + A[id - 1] + A[id] + A[id + 1] + A[id + 2])/5;
	};
}

//a simple 2D kernel
kernel void add2D(global const int* A, global const int* B, global int* C) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	int id = x + y * width;

	//printf("id = %d x = %d y = %d w = %d h = %d\n", id, x, y, width, height);

	C[id] = A[x] + B[y];
}
