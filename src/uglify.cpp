// The MIT License (MIT)
//
// Copyright (c) 2017-2020 Kyle Hawk
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
	JsonToken token;
	int i = 0;
	while ((token = parser->nextToken()) != JsonToken::NOT_AVAILABLE) {
		switch (token) {
		case JsonToken::END_ARRAY:
			generator->endArray();
			break;
		case JsonToken::END_OBJECT:
			generator->endObject();
			break;
		case JsonToken::FIELD_NAME:
			generator->writeFieldName(parser->getCurrentName());
			break;
		case JsonToken::START_ARRAY:
			generator->startArray();
			break;
		case JsonToken::START_OBJECT:
			generator->startObject();
			break;
		case JsonToken::VALUE_FALSE:
			generator->write(false);
			break;
		case JsonToken::VALUE_NULL:
			generator->write(nullptr);
			break;
		case JsonToken::VALUE_NUMBER_FLOAT:
			generator->write(parser->getDoubleValue());
			break;
		case JsonToken::VALUE_NUMBER_INT:
			generator->write(parser->getIntegerValue());
			break;
		case JsonToken::VALUE_STRING:
			generator->write(parser->getText());
			break;
		case JsonToken::VALUE_TRUE:
			generator->write(true);
			break;
		case JsonToken::NOT_AVAILABLE:
			break;
		}
		++i;
	}

	return i;
}

int main(int argc, char* argv[]) {
	auto start = std::chrono::high_resolution_clock::now();
	//std::ifstream inputFile(argv[1]);
	//std::ofstream outputFile(argv[2]);
	FILE* inputFile = fopen(argv[1], "r");
	FILE* outputFile = fopen(argv[2], "w");
	bool prettify = false;
	if (argc > 3 && std::string("--prettify") == argv[3]) {
		prettify = true;
	}

	int numTokens = 0;
	try {
		numTokens = streamingCopy(inputFile, outputFile, prettify);
	} catch (const JsonException& e) {
		std::cerr << "Failed to uglify file: " << e.what() << std::endl;
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
						end - start)
						.count();
	std::cout << "Microseconds: " << duration << std::endl;
	std::cout << "Total token count: " << numTokens << std::endl;

	return 0;
}
