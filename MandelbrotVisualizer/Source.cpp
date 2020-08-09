#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include "math.h"
#include "KernelHeader.cuh"
#include <thread>;



// Override base class with your custom functionality
class Visualizer : public olc::PixelGameEngine
{
public:
	const int ZoomRes = 150;
	int iterations;
	double rightB;
	double leftB;
	double topB;
	double downB;
	double pWidth;
	double pHeight;
	int blocks;
	int threads;
	int MousePosX;
	int MousePosY;
	bool dragging;

	int* screen;

	Visualizer()
	{
		sAppName = "Mandelbrot";
	}

public:

	void drawScreen(int* screen) {
		for (int i = 0; i < ScreenHeight() * ScreenWidth(); i++) {
				
			int red = std::floor(255 * screen[i] / iterations);


				int colour = std::floor(255 - 255 * ((double)screen[i] / double(iterations)));
				
				Draw(i % ScreenWidth(), i / ScreenHeight(), olc::Pixel(colour, colour, colour));
		}
	}
	

	//Function callled at the beggining
	bool OnUserCreate() override
	{
		rightB = 2;
		leftB = -2;
		topB = 2;
		downB = -2;
		iterations = 5000;
		dragging = false;

		threads = 128;
		blocks = (ScreenHeight() * ScreenWidth()) / threads + 1;

		screen = AllocateMem(screen, ScreenWidth() * ScreenHeight());
		
		return true;
	}

	void CalculateMouseOffset() {
		if (GetMouse(0).bHeld) {
			if (!dragging) {
				MousePosX = GetMouseX();
				MousePosY = GetMouseY();
				dragging = true;
			}

			int MousePosDX = MousePosX - GetMouseX();
			int MousePosDY = MousePosY - GetMouseY();
			MousePosX = GetMouseX();
			MousePosY = GetMouseY();

			topB += MousePosDY * pHeight;
			downB += MousePosDY * pHeight;
			leftB += MousePosDX * pWidth;
			rightB += MousePosDX * pWidth;
		}
		else {
			dragging = false;
		}
		
	}

	double GetMouseXInSpace(int xPos) {
		double Wide = rightB - leftB;
		return leftB + Wide * ((double)xPos / (double)ScreenWidth());
	}

	double GetMouseYInSpace(int yPos) {
		double Height = topB - downB;
		return downB + Height * ((double)yPos / (double)ScreenHeight());
	}

	void CalculateZoom() {
		if (GetKey(olc::Key::Q).bPressed) {
			double xPos = GetMouseXInSpace(GetMouseX());
			double yPos = GetMouseYInSpace(GetMouseY());
			
 			topB = yPos + ZoomRes * pHeight;
			downB = yPos - ZoomRes * pHeight;
			leftB = xPos - ZoomRes * pWidth;
			rightB = xPos + ZoomRes * pWidth;

		}
	}

	void ChangePixelSize() {
		pWidth = (rightB - leftB) / ScreenWidth();
		pHeight = (topB - downB) / ScreenHeight();
	}

	void Reset() {
		if (GetKey(olc::Key::R).bHeld) {
			rightB = 2;
			leftB = -2;
			topB = 2;
			downB = -2;
		}
	}

	void AdjustIterations() {
		if (GetKey(olc::Key::W).bHeld) {
			iterations = iterations + 1500;
		}
		else if (GetKey(olc::Key::E).bHeld && iterations > 1500) {
			iterations = iterations - 1500;
		}
	}

	void DrawIterations() {
		std::stringstream ss;
		ss << iterations;
		std::string str;
		ss >> str;
		std::string text = "Iterations: " + str;


		DrawString(5, 5, text, olc::RED, 2);
	}


	//Function called every farame 
	bool OnUserUpdate(float fElapsedTime) override
	{
			ChangePixelSize();
			CalculateMouseOffset();
			CalculateZoom();
			Reset();
			AdjustIterations();


			CalculateScreen(screen, ScreenWidth(), ScreenHeight(), leftB, downB, pWidth, pHeight, iterations, blocks, threads);

			drawScreen(screen);
			DrawIterations();
		
		return true;
	}
};

int main()
{
	Visualizer demo;
	if (demo.Construct(1200, 1200, 1, 1))
		demo.Start();
	return 0;
}