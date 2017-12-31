#ifndef JAXUP_H
#define JAXUP_H

#include "jaxup_parser.h"
#include "jaxup_generator.h"
#include <memory>

namespace jaxup {

class JsonFactory {
public:
	std::shared_ptr<JsonParser> createJsonParser(std::istream& inputStream) {
		return std::make_shared<JsonParser>(inputStream);
	}
	std::shared_ptr<JsonGenerator> createJsonGenerator(
			std::ostream& outputStream, bool prettyPrint = false) {
		return std::make_shared<JsonGenerator>(outputStream, prettyPrint);
	}
};

}

#endif
