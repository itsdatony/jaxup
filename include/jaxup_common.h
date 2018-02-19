// The MIT License (MIT)
//
// Copyright (c) 2017-2018 Kyle Hawk
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
