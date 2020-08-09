#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"

__global__ void feelScreenGPU(int* screen, int ScreenWidth, int ScreenHeight, double leftB, double downB, double pWidth, double pHeight, int iterations) {
	int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
	bool found = false;
	int color = 0;
	bool print = false;
	if (threadNum == 529469) {
		print = true;
	}

	if (threadNum < ScreenWidth * ScreenHeight) {
		int count = 0;
		double r1 = 0;
		double r2 = leftB + pWidth * (double)(threadNum % ScreenWidth);
		double c1 = 0;
		double c2 = downB + pHeight * (double)(threadNum / ScreenHeight);

		


		while (count < iterations)
		{	
			if (print & screen[threadNum] == 0) printf("Count: %d, r1: %f, c1: %f, r2: %f, c2: %f2 \n", count, r1, c1, r2, c2);
			r1 = r1 * r1 - c1 * c1 + r2;
			c1 = 2 * r1 * c1 + c2;


			if ((r1 * r1 + c1 * c1) < 4) {
				count++;
			}
			else{
				screen[threadNum] = count;
				return;
			}
		}
		screen[threadNum] = -1;
	}
}

int* CalculateScreen(int ScreenWidth, int ScreenHeight, double leftB, double downB, double pWidth, double pHeight, int iterations, int Blocks, int Threads) {
	
	int* screen;
	cudaMallocManaged(&screen, ScreenHeight * ScreenWidth * sizeof(int));
	
	for (int i = 0; i < ScreenWidth * ScreenHeight; i++) {
		screen[i] = 0;
	}
	

	feelScreenGPU <<<Blocks, Threads>>> (screen, ScreenWidth, ScreenHeight, leftB, downB, pWidth, pHeight, iterations);
	cudaDeviceSynchronize();


	return screen;
}

void FreeMem(int* screen) {
	cudaFree(screen);
}

void AllocateMem(int* screen, int memSize) {
	cudaMallocManaged(&screen, memSize * sizeof(int));
}