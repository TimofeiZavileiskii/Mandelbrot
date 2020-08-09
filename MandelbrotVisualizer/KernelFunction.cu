#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"

__global__ void feelScreenGPU(int* screen, int ScreenWidth, int ScreenHeight, double leftB, double downB, double pWidth, double pHeight, int iterations) {
	
	int threadNum = blockIdx.x * blockDim.x + threadIdx.x;


	//if (threadNum < ScreenWidth * ScreenHeight) {
		int count = 0;
		double r1 = 0;
		double r2 = leftB + pWidth * (threadNum % ScreenWidth);
		double c1 = 0;
		double c2 = downB + pHeight * (threadNum / ScreenHeight);


		while (count < iterations)
		{	
			double r1Temp = r1;
			r1 = r1 * r1 - c1 * c1 + r2;
			c1 = 2 * r1Temp * c1 + c2;


			if ((r1 * r1 + c1 * c1) > 4) {
				break;
			}
			count++;
		}
		screen[threadNum] = count;
	//}
}

void CalculateScreen(int* screen, int ScreenWidth, int ScreenHeight, double leftB, double downB, double pWidth, double pHeight, int iterations, int Blocks, int Threads) {
	
	
	
	feelScreenGPU <<<Blocks, Threads>>> (screen, ScreenWidth, ScreenHeight, leftB, downB, pWidth, pHeight, iterations);
	cudaDeviceSynchronize();
	
}

void FreeMem(int* screen) {
	cudaFree(screen);
}

int* AllocateMem(int* screen, int memSize) {
	cudaMallocManaged(&screen, memSize * sizeof(int));
	return screen;
}