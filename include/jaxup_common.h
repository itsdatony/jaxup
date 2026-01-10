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

#ifndef JAXUP_COMMON_H
#define JAXUP_COMMON_H

#include <exception>
#include <string>

namespace jaxup {

static const unsigned int initialBuffSize = 8192*4;

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

static inline std::string getTokenAsString(JsonToken t) {
	switch(t) {
	case JsonToken::NOT_AVAILABLE:
		return "Not Available";
	case JsonToken::START_OBJECT:
		return "Start Object ({)";
	case JsonToken::END_OBJECT:
		return "End Object (})";
	case JsonToken::START_ARRAY:
		return "Start Array ([)";
	case JsonToken::END_ARRAY:
		return "End Array (])";
	case JsonToken::FIELD_NAME:
		return "Field Name";
	case JsonToken::VALUE_STRING:
		return "String";
	case JsonToken::VALUE_NUMBER_INT:
		return "Integer";
	case JsonToken::VALUE_NUMBER_FLOAT:
		return "Double";
	case JsonToken::VALUE_TRUE:
		return "True";
	case JsonToken::VALUE_FALSE:
		return "False";
	case JsonToken::VALUE_NULL:
		return "Null";
	default:
		return "Unknown";
	}
}

#if (defined(__EXCEPTIONS) || defined(_HAS_EXCEPTIONS)) && !defined(JAXUP_NO_EXCEPTIONS)

class JsonException : public std::exception {
public:
	JsonException(const std::string& text) : text(text) {
	}

	template<typename... Args>
	JsonException(const std::string& first, Args... rest) : text(getCombinedString(first, rest...)) {}

	const char* what() const noexcept override {
		return text.c_str();
	}

private:
	const std::string text;

	template<typename... Args>
	static std::string getCombinedString(const std::string& first, Args... rest) {
		return first + getCombinedString(rest...);
	}
	static std::string getCombinedString(const std::string& first) {
		return first;
	}
};

#define JAXUP_THROW(...) throw JsonException(__VA_ARGS__)
#define JAXUP_TRY try
#define JAXUP_CATCH(condition) catch(condition)

#else

class JsonException {
public:
	const char* what() const { return ""; }
};

static JsonException JAXUP_DUMMY_EXCEPTION = {};

#define JAXUP_THROW(...) std::exit(1);
#define JAXUP_TRY
#define JAXUP_CATCH(condition) for(condition = JAXUP_DUMMY_EXCEPTION; false;)

#endif

}

#endif
