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
#include <fstream>
#include <iostream>
#include <jaxup.h>

using namespace jaxup;

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cerr << "Expected format: " << argv[0] << " inputFile" << std::endl;
		return 1;
	}
	int error = 0;
	auto start = std::chrono::high_resolution_clock::now();
	//std::ifstream inputFile(argv[1]);
	FILE* inputFile = fopen(argv[1], "r");
	setbuf(inputFile, nullptr);

	JsonToken token;
	int i = 0;
	try {
		JsonFactory factory;
		std::shared_ptr<JsonParser<FILE*>> parser = factory.createJsonParser(
			inputFile);
		while ((token = parser->nextToken()) != JsonToken::NOT_AVAILABLE) {
			++i;
		}
	} catch (const JsonException& e) {
		std::cerr << "Failed to parse file: " << e.what() << std::endl;
		error = 1;
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
						end - start)
						.count();
	std::cout << "Microseconds: " << duration << std::endl;
	std::cout << "Total token count: " << i << std::endl;

	return error;
}
