// The MIT License (MIT)
//
// Copyright (c) 2017-2020 Kyle Hawk
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

#ifndef JAXUP_PARSER_H
#define JAXUP_PARSER_H

#include <cstdint>
#include <iostream>
#include <vector>

#include "jaxup_common.h"
#include "jaxup_grisu.h"

namespace jaxup {

template <class source, size_t size>
class JsonSource {
};

template <size_t size>
class JsonSource<std::istream, size> {
public:
	JsonSource(std::istream& input) : input(input) {
	}
	inline size_t loadMore(char inputBuffer[size]) {
		if (input.eof() || input.bad()) {
			return 0;
		}
		input.read(&inputBuffer[0], size);
		return static_cast<size_t>(input.gcount());
	}

private:
	std::istream& input;
};

template <size_t size>
class JsonSource<FILE*, size> {
public:
	JsonSource(FILE* input) : input(input) {
	}
	inline size_t loadMore(char inputBuffer[size]) {
		if (input == nullptr) {
			return 0;
		}
		return fread(&inputBuffer[0], 1, size, input);
	}

private:
	FILE* input;
};

static inline int getIntFromChar(char c) {
	return c - '0';
}

template <class source>
class JsonParser {
private:
	int64_t int64Value = 0;
	double doubleValue = 0.0;
	JsonToken token = JsonToken::NOT_AVAILABLE;
	int inputOffset = 0;
	int inputSize = 0;
	char inputBuffer[initialBuffSize];
	std::string currentName, currentString;
	std::vector<JsonToken> tagStack;
	JsonSource<source, initialBuffSize> input;

public:
	JsonParser(source& input) : currentName(""), currentString(""), input(input) {
		currentName.reserve(32);
		currentString.reserve(initialBuffSize);
		tagStack.reserve(32);
	}
	~JsonParser() = default;

	JsonToken currentToken() const {
		return this->token;
	}

	const std::string& getCurrentName() const {
		return this->currentName;
	}

	int64_t getIntegerValue() const {
		if (this->token == JsonToken::VALUE_NUMBER_INT) {
			return this->int64Value;
		} else if (this->token == JsonToken::VALUE_NUMBER_FLOAT) {
			return static_cast<int64_t>(this->doubleValue);
		}
		throw JsonException("Attempted to parse a ", getTokenAsString(this->token), " token as an Integer");
	}

	double getDoubleValue() const {
		if (this->token == JsonToken::VALUE_NUMBER_FLOAT) {
			return this->doubleValue;
		} else if (this->token == JsonToken::VALUE_NUMBER_INT) {
			return static_cast<double>(this->int64Value);
		}
		throw JsonException("Attempted to parse a ", getTokenAsString(this->token), " token as a Double");
	}

	bool getBooleanValue() const {
		if (this->token == JsonToken::VALUE_TRUE) {
			return true;
		} else if (this->token == JsonToken::VALUE_FALSE) {
			return false;
		}
		throw JsonException("Attempted to parse a ", getTokenAsString(this->token), " token as a Boolean");
	}

	const std::string& getText() const {
		return this->currentString;
	}

	JsonToken nextValue() {
		while (this->nextToken() == JsonToken::FIELD_NAME)
			;
		return this->token;
	}

	JsonParser& skipChildren() {
		if (this->token == JsonToken::START_OBJECT) {
			skipPair(JsonToken::START_OBJECT, JsonToken::END_OBJECT);
		} else if (this->token == JsonToken::START_ARRAY) {
			skipPair(JsonToken::START_ARRAY, JsonToken::END_ARRAY);
		}
		return *this;
	}

	JsonToken nextToken() {
		char c;
		bool comma = false;
		if (this->token == JsonToken::FIELD_NAME) {
			getNextSignificantCharacter(&c);
			if (c != ':') {
				throw JsonException("Expected a colon, but none was found");
			}
		} else if (!this->tagStack.empty() && this->token != JsonToken::START_ARRAY && this->token != JsonToken::START_OBJECT) {
			// Expect a comma or a close array/object
			getNextSignificantCharacter(&c);
			switch (c) {
			case ']':
				return parseCloseArray();
			case '}':
				return parseCloseObject();
			case ',':
				comma = true;
				break;
			default:
				throw JsonException(
					"Expected a comma before the next value, but none was found");
			}
		}

		if (this->token != JsonToken::FIELD_NAME && !this->tagStack.empty() && this->tagStack.back() == JsonToken::START_OBJECT) {
			getNextSignificantCharacter(&c);
			if (c == '}') {
				return parseCloseObject(comma);
			}
			// Expect a field name next
			if (c != '"') {
				throw JsonException("Expected a quoted string value");
			}
			parseString(currentName);
			return foundToken(JsonToken::FIELD_NAME);
		}

		while (readNextCharacter(&c)) {
			if (isInsignificantWhitespace(c))
				continue;
			switch (c) {
			case '-':
				return parseNegativeNumber();
			case '0':
			case '1':
			case '2':
			case '3':
			case '4':
			case '5':
			case '6':
			case '7':
			case '8':
			case '9':
				return parsePositiveNumber(c);
			case '"':
				parseString(currentString);
				return foundToken(JsonToken::VALUE_STRING);
			case 't':
				if (!nextEquals("rue", 3)) {
					throw JsonException("Invalid token beginning with t");
				}
				return foundToken(JsonToken::VALUE_TRUE);
			case 'f':
				if (!nextEquals("alse", 4)) {
					throw JsonException("Invalid token beginning with f");
				}
				return foundToken(JsonToken::VALUE_FALSE);
			case 'n':
				if (!nextEquals("ull", 3)) {
					throw JsonException("Invalid token beginning with n");
				}
				return foundToken(JsonToken::VALUE_NULL);
			case '{':
				tagStack.push_back(JsonToken::START_OBJECT);
				return foundToken(JsonToken::START_OBJECT);
			case '}':
				return parseCloseObject(comma);
			case '[':
				tagStack.push_back(JsonToken::START_ARRAY);
				return foundToken(JsonToken::START_ARRAY);
			case ']':
				return parseCloseArray(comma);
			default:
				throw JsonException("Invalid token beginning with character: ", std::string(&c, 1));
			}
		}
		if (!this->tagStack.empty()) {
			if (tagStack.back() == JsonToken::START_OBJECT) {
				throw JsonException("Failed to close object at end of stream");
			} else {
				throw JsonException("Failed to close array at end of stream");
			}
		}
		return foundToken(JsonToken::NOT_AVAILABLE);
	}

private:
	void parseString(std::string& buff) {
		buff.clear();
		long code;
		char c = 0;
		int runStart;
		for (;;) {
			runStart = inputOffset;
			while (inputOffset < inputSize) {
				c = inputBuffer[inputOffset];
				if ((c < ' ' && (signed char)c >= 0) || c == '"' || c == '\\') {
					break;
				}
				++inputOffset;
			}

			if (inputOffset > runStart) {
				buff.append(&inputBuffer[runStart], inputOffset - runStart);
			}

			if (inputOffset > inputSize - 1) {
				if (!loadMore()) {
					throw JsonException("String was not terminated");
				}
				continue;
			}

			++inputOffset;
			if (c == '"') {
				if (!nextIsDelimiter()) {
					throw JsonException("Invalid string");
				}
				return;
			} else if (c == '\\') {
				readNextCharacter(&c);
				switch (c) {
				case '"':
				case '\\':
				case '/':
					buff.push_back(c);
					break;
				case 'b':
					buff.push_back('\b');
					break;
				case 'f':
					buff.push_back('\f');
					break;
				case 'n':
					buff.push_back('\n');
					break;
				case 'r':
					buff.push_back('\r');
					break;
				case 't':
					buff.push_back('\t');
					break;
				case 'u':
					code = parseHexcode();
					if (code < 0x80) {
						buff.push_back((char)code);
					} else if (code < 0x800) {
						buff.push_back(0xC0 | (char)(code >> 6));
						buff.push_back(0x80 | (char)(code & 0x3F));
					} else {
						buff.push_back(0xE0 | (char)(code >> 12));
						buff.push_back(0x80 | (char)((code >> 6) & 0x3F));
						buff.push_back(0x80 | (char)(code & 0x3F));
					}
					break;
				default:
					throw JsonException("Invalid escape code: \\", std::string(&c, 1));
				}
			} else {
				char intRep[5] = { 0, 0, 0, 0, 0 };
				grisu::writeSmallInteger(intRep, (int) c);
				throw JsonException("Unescaped control character with value: ", intRep);
			}
		}
	}

	inline long parseHexcode() {
		long code = 0;
		char c;
		for (unsigned int i = 0; i < 4; ++i) {
			readNextCharacter(&c);
			if (isdigit(c)) {
				code = code * 16 + (c - '0');
			} else {
				c = c & ~' '; // To uppercase
				if (c < 'A' || c > 'F') {
					throw JsonException("Invalid hex digit");
				}
				code = code * 16 + c - 'A' + 10;
			}
		}
		return code;
	}

	inline JsonToken parseCloseArray(bool comma = false) {
		if (comma) {
			throw JsonException("Invalid trailing comma in array");
		}
		if (tagStack.empty()) {
			throw JsonException("Tag underflow");
		}
		if (tagStack.back() != JsonToken::START_ARRAY) {
			throw JsonException("Unexpected end array");
		}
		tagStack.pop_back();
		return foundToken(JsonToken::END_ARRAY);
	}

	inline JsonToken parseCloseObject(bool comma = false) {
		if (comma) {
			throw JsonException("Invalid trailing comma in object");
		}
		if (tagStack.empty()) {
			throw JsonException("Tag underflow");
		}
		if (tagStack.back() != JsonToken::START_OBJECT) {
			throw JsonException("Unexpected end object");
		}
		tagStack.pop_back();
		return foundToken(JsonToken::END_OBJECT);
	}

	JsonToken parseNegativeNumber() {
		char c;
		if (!readNextCharacter(&c) || !isDigit(c)) {
			throw JsonException("Invalid number");
		}

		JsonToken output = parsePositiveNumber(c);
		this->doubleValue *= -1.0;
		this->int64Value *= -1;
		return output;
	}

	JsonToken parsePositiveNumber(char c) {
		static const uint64_t bigInt = (uint64_t)(INT64_MAX) / 10;
		if (c == '0') {
			if (peekNextCharacter(&c) && isDigit(c)) {
				throw JsonException("Leading zeroes are not allowed");
			}
			c = '0'; // Undo peek
		}

		bool rounded = false;
		uint64_t significand = getIntFromChar(c);
		int decimalExponent = 0;
		while (peekNextCharacter(&c) && isDigit(c)) {
			if (significand >= bigInt) {
				if (significand != bigInt || c > '7') {
					rounded = true;
					++decimalExponent;
					if (c > '5' || (c == '5' && (significand & 1))) {
						++significand;
					}
					break;
				}
			}
			significand = significand * 10 + getIntFromChar(c);
			++inputOffset;
		}
		// Eat remaining digits
		while (isDigit(c)) {
			advanceAndPeekNextCharacter(&c);
		}

		if (c == '.') {
			advanceAndPeekNextCharacter(&c);
			if (!isDigit(c)) {
				throw JsonException("Expected digit after decimal point");
			}
			if (!rounded) {
				do {
					if (significand >= bigInt) {
						if (significand != bigInt || c > '7') {
							rounded = true;
							if (c > '5' || (c == '5' && (significand & 1))) {
								++significand;
							}
							break;
						}
					}
					significand = significand * 10 + getIntFromChar(c);
					--decimalExponent;
					advanceAndPeekNextCharacter(&c);
				} while (isDigit(c));
			}
			// Eat remaining digits
			while (isDigit(c)) {
				advanceAndPeekNextCharacter(&c);
			}
		}

		if (c == 'e' || c == 'E') {
			advanceAndPeekNextCharacter(&c);

			bool isNegativeExponent = false;
			if (c == '+') {
				advanceAndPeekNextCharacter(&c);
			} else if (c == '-') {
				isNegativeExponent = true;
				advanceAndPeekNextCharacter(&c);
			}

			if (!isDigit(c)) {
				throw JsonException("Invalid exponent");
			}

			int tempExponent = 0;
			do {
				tempExponent = tempExponent * 10 + (c - '0');
			} while (advanceAndPeekNextCharacter(&c) && isDigit(c));

			if (isNegativeExponent) {
				tempExponent = -tempExponent;
			}

			decimalExponent += tempExponent;
		}

		if (c != 0 && !isDelimiter(c)) {
			throw JsonException("Invalid JSON number");
		}

		if (!rounded) {
			// Decide if number fits in a int64_t
			if (decimalExponent == 0 && significand <= static_cast<uint64_t>(INT64_MAX)) {
				this->int64Value = significand;
				return foundToken(JsonToken::VALUE_NUMBER_INT);
			}
			if (decimalExponent > 0 && decimalExponent < 20) {
				uint64_t power = grisu::getIntegerPowTen(decimalExponent);
				uint64_t mul = power * significand;
				if (mul == 0 || ((mul / power == significand) && mul <= static_cast<uint64_t>(INT64_MAX))) {
					this->int64Value = mul;
					return foundToken(JsonToken::VALUE_NUMBER_INT);
				}
			}
			// Fall through to floating point handling
		}
		this->doubleValue = grisu::raiseToPowTen((double)significand, decimalExponent);

		return foundToken(JsonToken::VALUE_NUMBER_FLOAT);
	}

	void skipPair(JsonToken start, JsonToken end) {
		int count = 1;
		while (count > 0) {
			JsonToken t = this->nextToken();
			if (t == start) {
				++count;
			} else if (t == end) {
				--count;
			} else if (t == JsonToken::NOT_AVAILABLE) {
				break;
			}
		}
	}

	inline bool loadMore() {
		inputOffset = 0;
		inputSize = static_cast<int>(input.loadMore(inputBuffer));
		return inputSize > 0;
	}

	inline bool readNextCharacter(char* c) {
		if (inputOffset > inputSize - 1) {
			if (!loadMore()) {
				*c = 0;
				return false;
			}
		}
		*c = inputBuffer[inputOffset++];
		return true;
	}

	inline bool peekNextCharacter(char* c) {
		if (inputOffset > inputSize - 1) {
			if (!loadMore()) {
				*c = 0;
				return false;
			}
		}
		*c = inputBuffer[inputOffset];
		return true;
	}

	inline bool advanceAndPeekNextCharacter(char* c) {
		++inputOffset;
		return peekNextCharacter(c);
	}

	inline void getNextSignificantCharacter(char* c) {
		while (readNextCharacter(c) && isInsignificantWhitespace(*c))
			;
	}

	inline JsonToken foundToken(JsonToken found) {
		this->token = found;
		return found;
	}

	static inline bool isInsignificantWhitespace(char c) {
		return c == ' ' || c == '\t' || c == '\r' || c == '\n';
	}

	static inline bool isDigit(char c) {
		return '0' <= c && c <= '9';
	}

	static inline bool isDelimiter(char c) {
		return c == ',' || c == ':' || c == ']' || c == '}' || isInsignificantWhitespace(c);
	}

	inline bool nextIsDelimiter() {
		char c;
		return !peekNextCharacter(&c) || isDelimiter(c);
	}

	inline bool nextEquals(const char chars[], unsigned int len) {
		char c;
		for (unsigned int i = 0; i < len; ++i) {
			if (!readNextCharacter(&c)) {
				return false;
			}
			if (c != chars[i]) {
				return false;
			}
		}
		return nextIsDelimiter();
	}
};
}

#endif
