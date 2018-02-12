#ifndef JAXUP_COMMON_H
#define JAXUP_COMMON_H

#include <exception>

namespace jaxup {

static const unsigned int initialBuffSize = 8196*4;

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

class JsonException : public std::exception {
public:
	JsonException(const char* what) : text(what) {
	}
	const char* what() const noexcept override {
		return text;
	}

private:
	const char* text;
};
}

#endif
