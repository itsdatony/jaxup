#include <chrono>
#include <fstream>
#include <iostream>
#include <jaxup.h>

using namespace jaxup;

int main(int /*argc*/, char* argv[]) {
	auto start = std::chrono::high_resolution_clock::now();
	std::ifstream inputFile(argv[1]);

	JsonToken token;
	int i = 0;
	try {
		JsonFactory factory;
		std::shared_ptr<JsonParser> parser = factory.createJsonParser(
			inputFile);
		while ((token = parser->nextToken()) != JsonToken::NOT_AVAILABLE) {
			++i;
		}
	} catch (const JsonException& e) {
		std::cerr << "Failed to parse file: " << e.what() << std::endl;
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
						end - start)
						.count();
	std::cout << "Microseconds: " << duration << std::endl;
	std::cout << "Total token count: " << i << std::endl;

	return 0;
}
