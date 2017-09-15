#include <iostream>
#include <chrono>
#include <fstream>
#include "jaxup.h"

using namespace jaxup;

int main(int argc, char* argv[]) {
	auto start = std::chrono::high_resolution_clock::now();
	std::ifstream inputFile(argv[1]);
	std::ofstream outputFile(argv[2]);

	JsonFactory factory;
	std::shared_ptr<JsonParser> parser = factory.createJsonParser(inputFile);
	std::shared_ptr<JsonGenerator> generator = factory.createJsonGenerator(outputFile);
	JsonToken token;
	int i = 0;
	while ((token = parser->nextToken()) != JsonToken::NOT_AVAILABLE) {
		switch(token) {
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
			generator->write(parser->getLongValue());
			break;
		case JsonToken::VALUE_STRING:
			generator->write(parser->getText());
			break;
		case JsonToken::VALUE_TRUE:
			generator->write(true);
			break;
		}
		++i;
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "Microseconds: " << duration << std::endl;
	std::cout << "Total token count: " << i << std::endl;
}

