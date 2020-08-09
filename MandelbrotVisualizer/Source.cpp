#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include "math.h"
#include "KernelHeader.cuh"


struct complexNum {
public:
	double realPart;
	double imagenaryPart;

	complexNum(double r = 0, double i = 0) {
		realPart = r;
		imagenaryPart = i;
	}

	
	complexNum operator+(complexNum& z) const {
		return complexNum(realPart + z.realPart, imagenaryPart + z.imagenaryPart);
	}

	complexNum operator-(complexNum& z) const {
		return complexNum(realPart - z.realPart, imagenaryPart - z.imagenaryPart);
	}

	complexNum operator*(complexNum& z) const {
		return complexNum(realPart * z.realPart - imagenaryPart * z.imagenaryPart, imagenaryPart * z.realPart + realPart * z.imagenaryPart );
	}

	complexNum operator*(const int& z) const {
		return complexNum(realPart * z, imagenaryPart * z);
	}
	

	double absoluteSqr() const{
		return realPart * realPart + imagenaryPart * imagenaryPart;
	}

};

// Override base class with your custom functionality
class Visualizer : public olc::PixelGameEngine
{
public:
	const int ZoomRes = 40;
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
	bool scrolling;

	int* screen;

	Visualizer()
	{
		sAppName = "Mandelbrot";
		scrolling = false;
	}

public:
	int CheckPartOfSet(complexNum z) {
		int count = 0;
		complexNum var (0, 0);
		while (count < iterations)
		{
			var = (var * var) + z;

				if (var.absoluteSqr() < 4) {
					count++;
				}
				else {
					return count;
					break;
				}
		}
		return -1;
	}

	void feelScreen(int* screen) {
		for (int i = 0; i < ScreenHeight() * ScreenWidth(); i++) {
				complexNum z((leftB + (i % ScreenWidth()) * pWidth), (downB + (i/ScreenHeight()) * pHeight));
				screen[i] = CheckPartOfSet(z);
		}
	}
	
	void drawScreen(int* screen) {
		for (int i = 0; i < ScreenHeight() * ScreenWidth(); i++) {
				int colour;

				if (screen[i] == -1) {
					colour = 0;
				}
				else {
					colour = std::floor(255 - 255 * ((double)screen[i] / double(iterations)));
				}
				Draw(i % ScreenWidth(), i / ScreenHeight(), olc::Pixel(colour, colour, colour));
		}
	}
	

	void FeelAndDrawScreen() {
		for (int i = 0; i < ScreenHeight(); i++) {
			for (int ii = 0; ii < ScreenWidth(); ii++) {

				complexNum z((leftB + ii * pWidth), (downB + i * pHeight));
				int numIterations = CheckPartOfSet(z);
				int color;

				if (numIterations == -1) {
					color = 0;
				}
				else {
					color = std::floor((double)255 * ((double)1 - (double)numIterations / (double)iterations));
				}
				Draw(ii, i, olc::Pixel(color, color, color));
			}
		}
	}

	//Function callled at the beggining
	bool OnUserCreate() override
	{
		rightB = 2;
		leftB = -2;
		topB = 2;
		downB = -2;
		iterations = 128;
		dragging = false;

		threads = 128;
		blocks = (ScreenHeight() * ScreenWidth()) / threads + 1;
		screen = new int[ScreenWidth() * ScreenHeight() * sizeof(int) * 4];
		for (int i = 0; i < ScreenWidth() * ScreenHeight(); i++) {
			screen[i] = i;
		}

	//	AllocateMem(screen, ScreenWidth() * ScreenHeight());
		
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
		if (GetKey(olc::Key::Q).bHeld) {
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
			iterations = iterations * 2;
		}
		else if (GetKey(olc::Key::E).bHeld && iterations >= 16) {
			iterations = iterations / 2;
		}
	}

	void DrawIterations() {
		std::stringstream ss;
		ss << iterations;
		std::string str;
		ss >> str;
		std::string text = "Iterations are " + str;


		DrawString(5, 5, text, olc::RED, 2);
	}

	void DrawCoordinates() {


		std::stringstream ss1;
		ss1 << GetMouseX();
		std::string str1;
		ss1 >> str1;

		std::stringstream ss2;
		ss2 << GetMouseY();
		std::string str2;
		ss2 >> str2;
		std::string text = "X: " + str1 + " Y: " + str2;

		DrawString(5, 20, text, olc::RED, 2);
	}

	//Function called every farame 
	bool OnUserUpdate(float fElapsedTime) override
	{
		//if (!scrolling) {
			ChangePixelSize();


			CalculateMouseOffset();

			CalculateZoom();
			Reset();

			AdjustIterations();

			feelScreen(screen);
			//screen = CalculateScreen(ScreenWidth(), ScreenHeight(), leftB, downB, pWidth, pHeight, iterations, blocks, threads);

			drawScreen(screen);
			//FreeMem(screen);

			DrawIterations();
			DrawCoordinates();

			scrolling = true;
		
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