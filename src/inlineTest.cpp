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

template<typename ... Args>
JsonNode toNode(Args&& ... args) {
	JsonFactory factory;
	auto parser = factory.createJsonParser(std::forward<Args>(args) ...);
	JsonNode node;
	node.read(*parser);
	return node;
}

template<typename ... Args>
bool testSuccess(const char* label, Args&& ... args) {
	auto start = std::chrono::steady_clock::now();
	auto node = toNode(std::forward<Args>(args) ...);
	bool success = node.getBoolean("success", false);
	auto end = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cerr << label << ": " << (success? "true" : "false") << " - " << duration << "μs" << std::endl;
	return success;
}

int main(int /*argc*/, char* /*argv*/[]) {
	bool success = true;
	const char* pointer = "{ \"different stuff\" : -1, \"success\" : true }";
	std::string string = "{ \"other stuff\" : 1.2, \"success\" : true }";
	success &= testSuccess("Character array reference", "{ \"stuff\" : 5, \"success\" : true }");
	success &= testSuccess("Character pointer with size", pointer, 44);
	success &= testSuccess("Character pointer", pointer);
	success &= testSuccess("String", string);

	return success? 0 : -1;
}
