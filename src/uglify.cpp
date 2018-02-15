#include <iostream>
#include <chrono>
#include <fstream>
#include <jaxup.h>

using namespace jaxup;

int streamingCopy(std::ifstream& inputFile, std::ofstream& outputFile, bool prettify) {
	JsonFactory factory;
	std::shared_ptr<JsonParser> parser = factory.createJsonParser(inputFile);
	std::shared_ptr<JsonGenerator> generator = factory.createJsonGenerator(
			outputFile, prettify);
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
	std::ifstream inputFile(argv[1]);
	std::ofstream outputFile(argv[2]);
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
			end - start).count();
	std::cout << "Microseconds: " << duration << std::endl;
	std::cout << "Total token count: " << numTokens << std::endl;

	return 0;
}

