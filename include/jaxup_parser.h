#ifndef JAXUP_PARSER_H
#define JAXUP_PARSER_H

#include <iostream>
#include <vector>

#include "jaxup_common.h"

namespace jaxup {

//template <class source, size_t size>
//class JsonSource {
//public:
//	JsonSource(source& input) : input(&input) {
//	}
//	int inputOffset = 0;
//	int inputSize = 0;
//	char inputBuffer[size];
//	void reset(source& newSource) {
//		this->inputOffset = 0;
//		this->inputSize = 0;
//		this->input = &newSource;
//	}
//	bool loadMore() {
//		return false;
//	}
//
//private:
//	source* input;
//};
//
//template <size_t size>
//class JsonSource<std::istream, size> {
//public:
//	//	JsonSource(std::istream& input) :
//	//			input(&input) {
//	//	}
//	//	int inputOffset = 0;
//	//	int inputSize = 0;
//	//	char inputBuffer[size];
//	bool loadMore() {
//		this->inputOffset = 0;
//		if (this->input->eof()) {
//			return false;
//		}
//		this->input->read(&this->inputBuffer[0], size);
//		this->inputSize = this->input->gcount();
//		return this->inputSize > 0;
//	}
//
//private:
//	std::istream* input;
//};

static inline double getDoubleFromChar(char c) {
	static const double DIGITS[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
									8.0, 9.0};
	return DIGITS[c - '0'];
}

class JsonParser {
private:
	long longValue = 0;
	JsonToken token = JsonToken::NOT_AVAILABLE;
	double doubleValue = 0.0;
	int inputOffset = 0;
	int inputSize = 0;
	char inputBuffer[initialBuffSize];
	std::string currentName, currentString;
	std::vector<JsonToken> tagStack;
	std::istream& input;

public:
	JsonParser(std::istream& inputStream) : currentName(""), currentString(""), input(inputStream) {
		currentName.reserve(initialBuffSize);
		currentString.reserve(initialBuffSize);
		tagStack.reserve(32);
	}
	~JsonParser() = default;

	JsonToken currentToken() {
		return this->token;
	}

	const std::string& getCurrentName() {
		return this->currentName;
	}

	long getLongValue() {
		if (this->token == JsonToken::VALUE_NUMBER_INT) {
			return this->longValue;
		} else if (this->token == JsonToken::VALUE_NUMBER_FLOAT) {
			return (long)this->doubleValue;
		}
		//TODO:
		throw JsonException("Invalid type");
	}

	double getDoubleValue() {
		if (this->token == JsonToken::VALUE_NUMBER_FLOAT) {
			return this->doubleValue;
		} else if (this->token == JsonToken::VALUE_NUMBER_INT) {
			return (double)this->longValue;
		}
		//TODO:
		throw JsonException("Invalid type");
	}

	bool getBooleanValue() {
		if (this->token == JsonToken::VALUE_TRUE) {
			return true;
		} else if (this->token == JsonToken::VALUE_FALSE) {
			return false;
		}
		//TODO:
		throw JsonException("Invalid type");
	}

	const std::string& getText() {
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
				break;
			default:
				throw JsonException(
					"Expected a comma before the next value, but none was found");
			}
		}

		if (this->token != JsonToken::FIELD_NAME && !this->tagStack.empty() && this->tagStack.back() == JsonToken::START_OBJECT) {
			// Expect a field name next
			getNextSignificantCharacter(&c);
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
				return parseCloseObject();
			case '[':
				tagStack.push_back(JsonToken::START_ARRAY);
				return foundToken(JsonToken::START_ARRAY);
			case ']':
				return parseCloseArray();
			default:
				throw JsonException("Invalid token");
			}
		}
		return foundToken(JsonToken::NOT_AVAILABLE);
	}

private:
	void parseString(std::string& buff) {
		buff.clear();
		long code;
		char c;
		int runStart;
		while (true) {
			runStart = inputOffset;
			while (inputOffset < inputSize) {
				c = inputBuffer[inputOffset];
				if ((c < ' ' && c >= 0) || c == '"' || c == '\\') {
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
					throw JsonException("Invalid escape code");
				}
			} else {
				std::cerr << "Unexpected value " << (int)c << std::endl;
				throw JsonException("Unescaped control character");
			}
		}
		throw JsonException("String was not terminated");
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

	inline JsonToken parseCloseArray() {
		if (tagStack.empty()) {
			throw JsonException("Tag underflow");
		}
		if (tagStack.back() != JsonToken::START_ARRAY) {
			throw JsonException("Unexpected end array");
		}
		tagStack.pop_back();
		return foundToken(JsonToken::END_ARRAY);
	}

	inline JsonToken parseCloseObject() {
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

		JsonToken token = parsePositiveNumber(c);
		this->doubleValue *= -1.0;
		this->longValue *= -1;
		return token;
	}

	JsonToken parsePositiveNumber(char c) {
		if (c == '0') {
			if (peekNextCharacter(&c) && isDigit(c)) {
				throw JsonException("Leading zeroes are not allowed");
			}
			c = '0'; // Undo peek
		}
		bool isFloatingPoint = false;

		this->doubleValue = getDoubleFromChar(c);
		while (peekNextCharacter(&c) && isDigit(c)) {
			this->doubleValue = this->doubleValue * 10.0 + getDoubleFromChar(c);
			++this->inputOffset;
		}

		if (c == '.') {
			isFloatingPoint = true;

			double fraction = 1.0;
			while (advanceAndPeekNextCharacter(&c) && isDigit(c)) {
				fraction *= 0.1;
				this->doubleValue += getDoubleFromChar(c) * fraction;
			}
		}

		if (c == 'e' || c == 'E') {
			isFloatingPoint = true;

			++this->inputOffset;
			peekNextCharacter(&c);

			double base = 10.0;
			if (c == '+') {
				advanceAndPeekNextCharacter(&c);
			} else if (c == '-') {
				base = 0.1;
				advanceAndPeekNextCharacter(&c);
			}

			if (!isDigit(c)) {
				throw JsonException("Invalid exponent");
			}

			unsigned long exponent = 0;
			do {
				exponent = exponent * 10 + (c - '0');
			} while (advanceAndPeekNextCharacter(&c) && isDigit(c));

			double power = 1.0;
			while (exponent > 0) {
				if (exponent & 1) {
					power *= base;
				}
				exponent >>= 1;
				base *= base;
			}

			this->doubleValue *= power;
		}

		if (c != 0 && !isDelimiter(c)) {
			throw JsonException("Invalid JSON number");
		}

		if (isFloatingPoint) {
			return foundToken(JsonToken::VALUE_NUMBER_FLOAT);
		}

		this->longValue = this->doubleValue;
		return foundToken(JsonToken::VALUE_NUMBER_INT);
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
		if (input.eof()) {
			return false;
		}
		input.read(&inputBuffer[0], initialBuffSize);
		inputSize = input.gcount();
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

	inline JsonToken foundToken(JsonToken token) {
		this->token = token;
		return token;
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
