#ifndef JAXUP_H
#define JAXUP_H

#include <iostream>
#include <string>
#include <memory>

namespace jaxup {

enum class JsonToken {
	NOT_AVAILABLE,
	START_OBJECT,
	END_OBJECT,
	START_ARRAY,
	END_ARRAY,
	FIELD_NAME,
	VALUE_STRING,
	VALUE_NUMBER_INT,
	VALUE_NUMBER_FLOAT,
	VALUE_TRUE,
	VALUE_FALSE,
	VALUE_NULL
};

class JsonException: public std::exception {
public:
	JsonException(const char* what) :
			text(what) {
	}
	const char* what() const noexcept override {
		return text;
	}
private:
	const char* text;
};

class JsonParser {
public:
	virtual ~JsonParser() = default;
	virtual JsonToken currentToken() = 0;
	virtual JsonToken nextToken() = 0;
	virtual const std::string& getCurrentName() = 0;
	virtual long getLongValue() = 0;
	virtual double getDoubleValue() = 0;
	virtual bool getBooleanValue() = 0;
	virtual const std::string& getText() = 0;
	virtual JsonToken nextValue() = 0;
	virtual JsonParser& skipChildren() = 0;
};

class JsonFactory {
public:
	std::shared_ptr<JsonParser> createJsonParser(std::istream& inputStream);
};

}

#endif
