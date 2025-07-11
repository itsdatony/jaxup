// The MIT License (MIT)
//
// Copyright (c) 2017-2025 Kyle Hawk
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include <chrono>
#include <iostream>
#include <string>
#include <jaxup_numeric.h>

using namespace jaxup::numeric;

void showResult(char buff[], double d) {
	int len = ryu(d, buff);
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
