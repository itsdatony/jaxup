// The MIT License (MIT)
//
// Copyright (c) 2017-2022 Kyle Hawk
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

int streamingCopy(FILE* inputFile, FILE* outputFile, bool prettify) {
	JsonFactory factory;
	auto parser = factory.createJsonParser(inputFile);
	auto generator = factory.createJsonGenerator(outputFile, prettify);
	JsonNode node;
	int i = 0;
	while (parser->nextToken() != JsonToken::NOT_AVAILABLE) {
		node.read(*parser);
		node.write(*generator);
		++i;
	}

	return i;
}

int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::cerr << "Expected format: " << argv[0] << " inputFile outputFile" << std::endl;
		return 1;
	}
	auto start = std::chrono::high_resolution_clock::now();
	//std::ifstream inputFile(argv[1]);
	//std::ofstream outputFile(argv[2]);
	FILE* inputFile = fopen(argv[1], "r");
	FILE* outputFile = fopen(argv[2], "w");
	bool prettify = false;
	if (argc > 3 && std::string("--prettify") == argv[3]) {
		prettify = true;
	}

	int numRootNodes = 0;
	try {
		numRootNodes = streamingCopy(inputFile, outputFile, prettify);
	} catch (const JsonException& e) {
		std::cerr << "Failed to uglify file: " << e.what() << std::endl;
		return 1;
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
						end - start)
						.count();
	std::cout << "Microseconds: " << duration << std::endl;
	std::cout << "Total root node count: " << numRootNodes << std::endl;

	return 0;
}
