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

#ifndef JAXUP_GENERATOR_H
#define JAXUP_GENERATOR_H

#include <cstring>
#include <iostream>
#include <vector>

#include "jaxup_common.h"
#include "jaxup_numeric.h"

namespace jaxup {

template <class source, size_t size>
class JsonDestination {
};

template <size_t size>
class JsonDestination<std::ostream, size> {
public:
	JsonDestination(std::ostream& output) : output(output) {
	}
	inline void write(char bytes[size], size_t count) {
		output.write(bytes, count);
	}

private:
	std::ostream& output;
};

template <size_t size>
class JsonDestination<FILE*, size> {
public:
	JsonDestination(FILE* output) : output(output) {
	}
	inline void write(char bytes[size], size_t count) {
		fwrite(bytes, 1, count, output);
	}

private:
	FILE* output;
};

template <class dest>
class JsonGenerator {
private:
	alignas(8) char unicodeBuff[8] = {'\\', 'u', '0', '0', '0', '0', 0, 0};
	alignas(8) char doubleBuff[36];
	char outputBuffer[initialBuffSize];
	std::size_t outputSize = 0;
	char* const doubleBuffEndMarker = doubleBuff + sizeof(doubleBuff);
	JsonDestination<dest, initialBuffSize> output;
	JsonToken token = JsonToken::NOT_AVAILABLE;
	std::vector<JsonToken> tagStack;
	std::string prettyBuff = "\n";
	bool prettyPrint;

	inline void writeBuff(char c) {
		if (outputSize >= initialBuffSize) {
			flush();
		}
		outputBuffer[outputSize++] = c;
	}

	inline void writeBuff(const char* c, std::size_t length) {
		if (outputSize + length <= initialBuffSize) {
			std::memcpy(&outputBuffer[outputSize], c, length);
			outputSize += length;
		} else {
			std::size_t first = initialBuffSize - outputSize;
			std::memcpy(&outputBuffer[outputSize], c, first);
			outputSize = initialBuffSize;
			flush();
			length -= first;
			std::memcpy(outputBuffer, c + first, length);
			outputSize += length;
		}
	}

	inline void writePrettyBuff() {
		writeBuff(prettyBuff.c_str(), prettyBuff.length());
	}

	inline void prepareWriteValue() {
		if (!tagStack.empty()) {
			JsonToken parent = tagStack.back();
			if (parent == JsonToken::START_OBJECT && token != JsonToken::FIELD_NAME) {
				throw JsonException("Tried to write a value without giving it a field name");
			}
			if (parent == JsonToken::START_ARRAY && token != JsonToken::START_ARRAY) {
				writeBuff(',');
			}
			if (prettyPrint && parent == JsonToken::START_ARRAY) {
				writePrettyBuff();
			}
		}
	}

	inline void encodeString(const char* value, std::size_t length) {
		writeBuff('"');
		std::size_t run = 0;
		std::size_t runStart = 0;
		for (std::size_t i = 0; i < length; ++i) {
			char c = value[i];
			if ((c >= ' ' || (signed char)c < 0) && c != '"' && c != '\\') {
				if (run == 0) {
					runStart = i;
				}
				++run;
				continue;
			}
			if (run > 0) {
				writeBuff(&value[runStart], run);
				run = 0;
			}

			switch (c) {
			case '"':
				writeBuff("\\\"", 2);
				break;
			case '\\':
				writeBuff("\\\\", 2);
				break;
			case '\b':
				writeBuff("\\b", 2);
				break;
			case '\f':
				writeBuff("\\f", 2);
				break;
			case '\n':
				writeBuff("\\n", 2);
				break;
			case '\r':
				writeBuff("\\r", 2);
				break;
			case '\t':
				writeBuff("\\t", 2);
				break;
			default:
				unicodeBuff[4] = (c >> 4) + '0'; // '0' or '1'
				c = c & 0xF;
				if (c < 10) {
					unicodeBuff[5] = c + '0';
				} else {
					unicodeBuff[5] = c - 10 + 'A';
				}
				writeBuff(unicodeBuff, 6);
			}
		}
		if (run > 0) {
			writeBuff(&value[runStart], run);
		}
		writeBuff('"');
	}

	inline int writeDoubleToBuff(double value, char* buff) {
		int len = numeric::ryu(value, buff);
		if (len < 0) {
			throw JsonException("Failed to serialize double");
		}
		if ((unsigned int)len > sizeof(doubleBuff)) {
			len = sizeof(doubleBuff);
		}
		return len;
	}

public:
	JsonGenerator(dest& output, bool prettyPrint) : output(output), prettyPrint(prettyPrint) {
		tagStack.reserve(32);
	}

	~JsonGenerator() {
		flush();
	}

	void flush() {
		if (outputSize > 0) {
			output.write(outputBuffer, outputSize);
			outputSize = 0;
		}
	}

	void write(double value) {
		prepareWriteValue();
		token = JsonToken::VALUE_NUMBER_FLOAT;
		if (sizeof(doubleBuff) <= initialBuffSize - outputSize) {
			int len = writeDoubleToBuff(value, &outputBuffer[outputSize]);
			outputSize += len;
		} else {
			int len = writeDoubleToBuff(value, doubleBuff);
			writeBuff(doubleBuff, len);
		}
	}

	void write(int64_t value) {
		prepareWriteValue();
		token = JsonToken::VALUE_NUMBER_INT;
		char* start = numeric::writeIntegerToBuff(value, doubleBuffEndMarker);
		writeBuff(start, doubleBuffEndMarker - start);
	}

	inline void write(int32_t value) {
		write(static_cast<int64_t>(value));
	}

	void write(bool value) {
		prepareWriteValue();
		if (value) {
			token = JsonToken::VALUE_TRUE;
			writeBuff("true", 4);
		} else {
			token = JsonToken::VALUE_FALSE;
			writeBuff("false", 5);
		}
	}

	void write(std::nullptr_t) {
		prepareWriteValue();
		token = JsonToken::VALUE_NULL;
		writeBuff("null", 4);
	}

	void write(const char* value) {
		prepareWriteValue();
		if (value == nullptr) {
			token = JsonToken::VALUE_NULL;
			writeBuff("null", 4);
			return;
		}
		token = JsonToken::VALUE_STRING;
		encodeString(value, std::strlen(value));
	}

	void write(const std::string& value) {
		prepareWriteValue();
		token = JsonToken::VALUE_STRING;
		encodeString(value.c_str(), value.length());
	}

	void writeFieldName(const std::string& field) {
		if (tagStack.empty() || tagStack.back() != JsonToken::START_OBJECT) {
			throw JsonException("Tried to write a field name outside of an object: ", field);
		}
		if (token != JsonToken::START_OBJECT) {
			writeBuff(',');
		}
		if (prettyPrint) {
			writePrettyBuff();
		}
		token = JsonToken::FIELD_NAME;
		encodeString(field.c_str(), field.length());
		if (!prettyPrint) {
			writeBuff(':');
		} else {
			writeBuff(" : ", 3);
		}
	}

	void startObject() {
		prepareWriteValue();
		token = JsonToken::START_OBJECT;
		tagStack.push_back(token);
		writeBuff('{');
		if (prettyPrint) {
			prettyBuff.push_back('\t');
		}
	}

	inline void startObject(const std::string& field) {
		writeFieldName(field);
		startObject();
	}

	void endObject() {
		if (tagStack.empty() || tagStack.back() != JsonToken::START_OBJECT) {
			throw JsonException("Tried to close an object while outside of an object");
		}
		token = JsonToken::END_OBJECT;
		tagStack.pop_back();
		if (prettyPrint) {
			prettyBuff.pop_back();
			writePrettyBuff();
		}
		writeBuff('}');
	}

	void startArray() {
		prepareWriteValue();
		token = JsonToken::START_ARRAY;
		tagStack.push_back(token);
		writeBuff('[');
		if (prettyPrint) {
			prettyBuff.push_back('\t');
		}
	}

	inline void startArray(const std::string& field) {
		writeFieldName(field);
		startArray();
	}

	void endArray() {
		if (tagStack.empty() || tagStack.back() != JsonToken::START_ARRAY) {
			throw JsonException("Tried to close an array while outside of an array");
		}
		token = JsonToken::END_ARRAY;
		tagStack.pop_back();
		if (prettyPrint) {
			prettyBuff.pop_back();
			writePrettyBuff();
		}
		writeBuff(']');
	}

	template <class T>
	inline void writeField(const std::string& field, T value) {
		writeFieldName(field);
		write(value);
	}
};
}

#endif
