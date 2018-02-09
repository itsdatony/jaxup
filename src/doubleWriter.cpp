#include <chrono>
#include <iostream>
#include <jaxup_grisu.h>

using namespace jaxup::grisu;

void showResult(char buff[], double d) {
	int len = fastDoubleToString(buff, d);
	std::string str(buff, len);
	std::cout << "str: " << str << ", len: " << len << std::endl;
}

int main(int /*argc*/, char* /*argv*/ []) {
	auto start = std::chrono::high_resolution_clock::now();
	char buff[36];
	double testVals[] = {0, 1.2, 500999123, 0.000012, 0.1234, 5e30, 5.123456789e-20, 0.001, 0.002, 0.003, 0.007};
	for (unsigned int i = 0; i < sizeof(testVals) / sizeof(testVals[0]); ++i) {
		showResult(buff, testVals[i]);
		showResult(buff, -testVals[i]);
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
						end - start)
						.count();
	std::cout << "Microseconds: " << duration << std::endl;

	return 0;
}
