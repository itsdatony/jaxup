#ifndef JAXUP_GENERATOR_H
#define JAXUP_GENERATOR_H

#include <cstring>
#include <iostream>
#include <vector>

#include "jaxup_common.h"
#include "jaxup_grisu.h"

namespace jaxup {

class JsonGenerator {
private:
	unsigned int outputSize = 0;
	char outputBuffer[initialBuffSize];
	std::ostream& output;
	JsonToken token = JsonToken::NOT_AVAILABLE;
	std::vector<JsonToken> tagStack;
	std::string prettyBuff = "\n";
	bool prettyPrint;
	alignas(8) char unicodeBuff[6] = {'\\', 'u', '0', '0', '0', '0'};
	alignas(8) char doubleBuff[36];

	void flush() {
		if (outputSize > 0) {
			output.write(outputBuffer, outputSize);
			outputSize = 0;
		}
	}

	inline void writeBuff(char c) {
		if (outputSize >= initialBuffSize) {
			flush();
		}
		outputBuffer[outputSize++] = c;
	}

	inline void writeBuff(const char* c, unsigned long length) {
		if (outputSize + length <= initialBuffSize) {
			std::memcpy(&outputBuffer[outputSize], c, length);
			outputSize += length;
		} else {
			long first = initialBuffSize - outputSize;
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

	inline void encodeString(const char* value, long length = -1) {
		writeBuff('"');
		long run = 0;
		long runStart = -1;
		for (long i = 0; value[i] != 0 || i < length; ++i) {
			char c = value[i];
			if ((c >= ' ' || c < 0) && c != '"' && c != '\\') {
				if (runStart < 0) {
					runStart = i;
				}
				++run;
				continue;
			}
			if (run > 0) {
				writeBuff(&value[runStart], run);
				run = 0;
				runStart = -1;
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

	inline int writeDoubleToBuff(double value, char* buff, int /*size*/) {
		//int len = std::snprintf(buff, size, "%.16g", value);
		int len = grisu::fastDoubleToString(buff, value);
		if (len < 0) {
			throw JsonException("Failed to serialize double");
		}
		if ((unsigned int)len > sizeof(doubleBuff)) {
			len = sizeof(doubleBuff);
		}
		return len;
	}

	inline int writeLongToBuff(long value, char* buff, int size) {
		int len = std::snprintf(buff, size, "%ld", value);
		if (len < 0) {
			throw JsonException("Failed to serialize long");
		}
		if ((unsigned int)len > sizeof(doubleBuff)) {
			len = sizeof(doubleBuff);
		}
		return len;
	}

public:
	JsonGenerator(std::ostream& outputStream, bool prettyPrint) : output(outputStream), prettyPrint(prettyPrint) {
		tagStack.reserve(32);
	}

	~JsonGenerator() {
		flush();
	}

	void write(double value) {
		prepareWriteValue();
		token = JsonToken::VALUE_NUMBER_FLOAT;
		if (sizeof(doubleBuff) <= initialBuffSize - outputSize) {
			int len = writeDoubleToBuff(value, &outputBuffer[outputSize], sizeof(doubleBuff));
			outputSize += len;
		} else {
			int len = writeDoubleToBuff(value, doubleBuff, sizeof(doubleBuff));
			writeBuff(doubleBuff, len);
		}
	}

	void write(long value) {
		prepareWriteValue();
		token = JsonToken::VALUE_NUMBER_INT;
		if (sizeof(doubleBuff) <= initialBuffSize - outputSize) {
			int len = writeLongToBuff(value, &outputBuffer[outputSize], sizeof(doubleBuff));
			outputSize += len;
		} else {
			int len = writeLongToBuff(value, doubleBuff, sizeof(doubleBuff));
			writeBuff(doubleBuff, len);
		}
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
		encodeString(value);
	}

	void write(const std::string& value) {
		prepareWriteValue();
		token = JsonToken::VALUE_STRING;
		encodeString(value.c_str(), value.length());
	}

	void writeFieldName(const std::string& field) {
		if (tagStack.empty() || tagStack.back() != JsonToken::START_OBJECT) {
			throw JsonException("Tried to write a field name outside of an object");
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
