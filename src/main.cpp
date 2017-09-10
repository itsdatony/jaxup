#include <iostream>
#include <sstream>
#include "jaxup.h"

static const std::string trueText = "true";
static const std::string falseText = "false";
static const std::string nullText = " \t\r\nnull,";
static const std::string doubleText = " \t\r\n[1012e0, {\"hey\" : 1.2} ]";

using namespace jaxup;

int main(int argc, char* argv[]) {
	std::istringstream iss(doubleText);

	JsonFactory factory;
	std::shared_ptr<JsonParser> parser = factory.createJsonParser(iss);
	JsonToken token;
	while ((token = parser->nextToken()) != JsonToken::NOT_AVAILABLE) {
		if (token == JsonToken::VALUE_NUMBER_FLOAT) {
			std::cout << "Double value: " << parser->getDoubleValue() << std::endl;
			std::cout << "Long value: " << parser->getLongValue() << std::endl;
		} else if (token == JsonToken::VALUE_STRING) {
			std::cout << "String value: " << parser->getText() << std::endl;
		} else {
			std::cout << "Other token: " << (int)parser->currentToken() << std::endl;
		}
	}
	std::cout << "Hello world" << std::endl;
}

